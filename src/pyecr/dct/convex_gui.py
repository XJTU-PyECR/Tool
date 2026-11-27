"""GUI adapter around the unchanged frozen experimental convex engine."""
import json
from pyecr.contracts.convex_dct import ConvexDctAnalysis
from pyecr.dct.execution import isolated_numeric,check_cancel


@isolated_numeric('convex')
def analyze_convex_gui(source,config,options,diagnostic_config,*,progress_callback=None,cancel_check=lambda:None):
    from pyecr.dct.signed_convex import solve_convex,search_convex
    check_cancel();cancel_check();spec=options.prior_spec()
    if progress_callback:progress_callback(.05,'Running '+options.mode+' with '+options.selection+' lambda')
    if options.selection=='manual':
        trial=solve_convex(source,config,spec)
        report=dict(version='convex-gui-manual/1',source=source.to_dict(),config=config.to_dict(),spec=spec.to_dict(),
            trials=[trial],selected_lambda=trial['lambda_value'] if trial['status']=='CONVERGED' else None,
            seconds=trial['seconds'],noise_evidence=options.noise_evidence,manual_noise_sd_physical=options.noise_sd_physical)
    else:
        report=search_convex(source,config,spec,noise_sd_physical=(options.noise_sd_physical,)*len(source.signal),noise_evidence=options.noise_evidence)
    check_cancel();cancel_check()
    result=ConvexDctAnalysis(options,diagnostic_config,json.dumps(report,ensure_ascii=False,allow_nan=False))
    if progress_callback:progress_callback(1.,'Stored '+options.mode+' spectrum; checking owner before publication')
    return result
