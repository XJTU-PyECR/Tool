"""Cancellable DCT execution; numerical contracts and algorithms are unchanged.

GUI services opt into a fresh spawn process per numerical operation. Only the
owned Process handle can be terminated. Inputs/results cross as versioned JSON,
never a live session, callback, mutable array, GUI object or pickle result.
"""
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json,multiprocessing,os,threading,time,traceback


@dataclass(frozen=True)
class _Scope:
    check: object = lambda: None
    isolated: bool = False
    phase: object = lambda *args: None

_scope=ContextVar('pyecr_dct_execution',default=_Scope())
_records=deque(maxlen=256)
_records_lock=threading.Lock()


def execution_records():
    """Bounded execution telemetry, separate from scientific result identities."""
    with _records_lock:return tuple(dict(x,events=tuple(x['events'])) for x in _records)


@contextmanager
def execution_scope(check,*,isolated=False,phase=None):
    token=_scope.set(_Scope(check,isolated,phase or (lambda *args:None)))
    try:yield
    finally:_scope.reset(token)


def run_numeric(isolated,check,function,*args,**kwargs):
    record=dict(kind='thread',operation=function.__name__,pid=os.getpid(),start_ns=time.perf_counter_ns(),
        cancel_ns=None,exit_ns=None,exitcode=None,terminated=False,status='running',events=[])
    registered=False
    def checked():
        try:check()
        except CancelledError:
            record['cancel_ns']=record['cancel_ns'] or time.perf_counter_ns();record['status']='cancelled';raise
    def phase(event,name,ns):
        nonlocal registered
        if not registered:
            with _records_lock:_records.append(record)
            registered=True
        record['events'].append(dict(event=event,phase=name,ns=ns))
    try:
        with execution_scope(checked,isolated=isolated,phase=phase):result=function(*args,**kwargs)
        record['status']='completed';return result
    except Exception:
        if record['status']!='cancelled':record['status']='error'
        raise
    finally:record['exit_ns']=time.perf_counter_ns()


def check_cancel():_scope.get().check()


@contextmanager
def numeric_phase(name):
    s=_scope.get();s.check();s.phase('enter',name,time.perf_counter_ns())
    try:yield
    finally:s.phase('exit',name,time.perf_counter_ns())
    s.check()


def _encode(kind,args,kwargs):
    if kind=='convex':
        return dict(source=args[0].to_dict(),config=args[1].to_dict(),options=args[2].to_dict(),diagnostic_config=args[3].to_dict())
    if kind=='monotonic':
        return dict(t=list(args[0]),y=list(args[1]),kwargs=kwargs)
    if kind=='signed':
        from pyecr.contracts.signed_analysis import default_diagnostic_config
        diagnostic=args[2] if len(args)>2 else kwargs.get('diagnostic_config')
        if diagnostic is None:diagnostic=default_diagnostic_config(args[1].grid)
        return dict(source=args[0].to_dict(),config=args[1].to_dict(),diagnostic_config=diagnostic.to_dict(),
                    terminal_sources=[x.to_dict() for x in kwargs.get('terminal_sources',())])
    if kind=='component':
        return dict(source=args[0].to_dict(),parent=args[1].to_dict(),config=args[2].to_dict(),parent_rmse=kwargs['parent_rmse'])
    raise ValueError('Unknown DCT execution operation')


def _decode(kind,value):
    if kind=='convex':
        from pyecr.contracts.convex_dct import ConvexDctAnalysis
        return ConvexDctAnalysis.from_dict(value)
    if kind=='monotonic':
        from pyecr.dct.core import DCTResult
        return DCTResult.from_dict(value)
    from pyecr.contracts import read_contract
    return read_contract(value)


def _pipe_ready(connection,timeout=0):
    # On Windows a writer that has closed normally can raise WinError 109.
    # A completed packet is drained before interpreting process exit.
    try:return connection.poll(timeout)
    except (EOFError,OSError):return False


def isolated_numeric(kind):
    def decorate(function):
        @wraps(function)
        def execute(*args,**kwargs):
            scope=_scope.get()
            if not scope.isolated or not _needs_isolation(kind,args,kwargs):return function(*args,**kwargs)
            # JSON is created before spawn; the child never reads current GUI state.
            payload=json.dumps(_encode(kind,args,kwargs),ensure_ascii=False,allow_nan=True)
            return _isolated(kind,payload,scope.check,kwargs.get('progress_callback'))
        return execute
    return decorate


def _needs_isolation(kind,args,kwargs):
    """Execution policy only: keep small cancellable solves in their thread.

    Spawn/import overhead is several seconds in the bundled Windows runtime.
    Large dense calls get their own process; numerical inputs stay identical.
    """
    if kind=='convex':return args[1].grid.n_tau>=241 or len(args[0].time_s)*args[1].grid.n_tau>=250000
    if kind=='monotonic':rows=len(args[0]);grid=int(kwargs.get('n_tau',121))
    else:
        rows=len(args[0].time_s);grid=args[1 if kind=='signed' else 2].grid.n_tau
        if kind=='signed':
            from pyecr.contracts.signed_analysis import default_diagnostic_config
            policy=args[2] if len(args)>2 else kwargs.get('diagnostic_config')
            if policy is None:policy=default_diagnostic_config(args[1].grid)
            grid=max((grid,*policy.grid_sizes))
    return grid>=241 or rows*grid>=250000


def _worker(kind,payload,connection,cancel):
    def check():
        if cancel.is_set():raise CancelledError('DCT cancellation acknowledged')
    def phase(event,name,ns):connection.send(('phase',dict(event=event,phase=name,ns=ns)))
    def progress(f,m):check();connection.send(('progress',(f,m)));check()
    try:
        p=json.loads(payload)
        with execution_scope(check,phase=phase):
            check()
            if kind=='monotonic':
                from pyecr.dct.core import fit_dct
                result=fit_dct(p['t'],p['y'],**p['kwargs'])
            elif kind=='signed':
                from pyecr.contracts import read_contract
                from pyecr.dct.signed_analysis import analyze_signed_dct
                result=analyze_signed_dct(read_contract(p['source']),read_contract(p['config']),read_contract(p['diagnostic_config']),
                    terminal_sources=tuple(read_contract(x) for x in p['terminal_sources']),cancel_check=check,progress_callback=progress)
            elif kind=='convex':
                from pyecr.contracts import DctSourceSnapshot,SignedDctConfig,SignedDctDiagnosticConfig
                from pyecr.contracts.convex_dct import ConvexDctOptions
                from pyecr.dct.convex_gui import analyze_convex_gui
                result=analyze_convex_gui(DctSourceSnapshot.from_dict(p['source']),SignedDctConfig.from_dict(p['config']),ConvexDctOptions.from_dict(p['options']),SignedDctDiagnosticConfig.from_dict(p['diagnostic_config']),cancel_check=check,progress_callback=progress)
            elif kind=='component':
                from pyecr.contracts import read_contract
                from pyecr.dct.components import fit_component_spectrum
                result=fit_component_spectrum(read_contract(p['source']),read_contract(p['parent']),read_contract(p['config']),parent_rmse=p['parent_rmse'])
            else:raise ValueError('Unknown DCT operation')
            check();connection.send(('result',json.dumps(result.to_dict(),ensure_ascii=False,allow_nan=True)))
    except BaseException as exc:
        connection.send(('error',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
    finally:connection.close()


def _isolated(kind,payload,check,progress):
    check();ctx=multiprocessing.get_context('spawn')
    parent,child=ctx.Pipe(duplex=False);cancel=ctx.Event()
    process=ctx.Process(target=_worker,args=(kind,payload,child,cancel),name='pyecr-dct-'+kind,daemon=False)
    record=dict(kind=kind,pid=None,start_ns=time.perf_counter_ns(),cancel_ns=None,exit_ns=None,exitcode=None,
                terminated=False,status='starting',events=[])
    result=None;error=None;started=False;cancelled=False
    try:
        process.start();started=True;record['pid']=process.pid;record['status']='running';child.close()
        with _records_lock:_records.append(record)
        while True:
            if not cancelled:
                try:check()
                except CancelledError:
                    cancelled=True;record['cancel_ns']=time.perf_counter_ns();record['status']='cancelling';cancel.set()
            if cancelled and not record['terminated'] and process.is_alive() and time.perf_counter_ns()-record['cancel_ns']>=150_000_000:
                # This exact Process was created above for this one operation.
                # Never enumerate/kill by name or touch the GUI/unrelated workers.
                process.terminate();record['terminated']=True
            if _pipe_ready(parent,.01):
                try:tag,value=parent.recv()
                except (EOFError,OSError):tag,value='eof',None
                if tag=='phase':record['events'].append(value)
                elif tag=='progress' and not cancelled and progress is not None:progress(*value)
                elif tag=='result':result=value
                elif tag=='error':error=value
            if not process.is_alive():
                # Drain already-written result before closing the pipe.
                while _pipe_ready(parent):
                    try:tag,value=parent.recv()
                    except (EOFError,OSError):break
                    if tag=='phase':record['events'].append(value)
                    elif tag=='result':result=value
                    elif tag=='error':error=value
                break
        process.join();record['exit_ns']=time.perf_counter_ns();record['exitcode']=process.exitcode
        if cancelled:
            record['status']='cancelled';raise CancelledError('DCT worker exited; previous complete result retained')
        check()
        if error is not None:
            record['status']='error'
            if error['type']=='CancelledError':raise CancelledError(error['message'])
            from pyecr.contracts import ContractError
            cls={'ValueError':ValueError,'TypeError':TypeError,'RuntimeError':RuntimeError,'ContractError':ContractError}.get(error['type'],RuntimeError)
            raise cls(error['type']+': '+error['message']+'\n'+error['traceback'])
        if result is None or process.exitcode!=0:
            record['status']='error';raise RuntimeError(f'DCT worker exited without a complete result (code {process.exitcode})')
        decoded=_decode(kind,json.loads(result));check();record['status']='completed';return decoded
    except CancelledError:
        record['status']='cancelled';record['cancel_ns']=record['cancel_ns'] or time.perf_counter_ns()
        raise
    finally:
        if started:
            if process.is_alive():
                cancel.set();process.join(.15)
                if process.is_alive():process.terminate();record['terminated']=True
                process.join()
            record['exit_ns']=record['exit_ns'] or time.perf_counter_ns();record['exitcode']=process.exitcode
            process.close()
        parent.close();child.close()
