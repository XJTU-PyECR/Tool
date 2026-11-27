"""Pure presentation of task telemetry; no scientific evaluations."""
from __future__ import annotations

from pyecr.fitting.progress import FitProgressEvent
from pyecr.model_display import display_fit_worker_text


def progress_label(snapshot: FitProgressEvent | None, *, cancelling: bool = False,
                   elapsed_s: float | None = None) -> str:
    if snapshot is None:
        return "Cancelling | awaiting worker exit" if cancelling else "Preparing | awaiting worker"
    elapsed = snapshot.elapsed_s if elapsed_s is None else elapsed_s
    phase = ("Cancelling" if cancelling else snapshot.phase
             if snapshot.phase in {"Completed", "Cancelled", "Failed"} else snapshot.message)
    label = f"{phase} | elapsed {elapsed:.1f} s"
    if snapshot.fraction is not None and not cancelling:
        label += f" | {snapshot.completed_starts}/{snapshot.total_starts} starts finished"
    if snapshot.best_raw_sse is not None:
        label += f" | best raw SSE {snapshot.best_raw_sse:.6g}"
    return display_fit_worker_text(label)


def progress_details(snapshot: FitProgressEvent | None) -> str:
    if snapshot is None:
        return "Telemetry unavailable: no worker snapshot yet (or loaded result has no live task)."
    s = snapshot
    r = s.revision
    lines = [f"Job {s.job_id} | sequence {s.sequence}",
             f"Revisions: dataset={r.dataset_revision}, config={r.config_revision}, physical={r.physical_input_revision}",
             "Elapsed: monotonic seconds since worker scope started; no ETA.",
             "Best raw SSE: minimum finite completed candidate in the last searched stage.",
             "This stage best may differ from the finally selected M2/M3 FitResult.",
             f"Best raw SSE: {s.best_raw_sse:.17g}" if s.best_raw_sse is not None else
             "Best raw SSE unavailable: no finite completed candidate yet.",
             f"Optimizer cost (SciPy half loss sum): {s.optimizer_cost:.17g}; loss={s.loss}" if s.optimizer_cost is not None else
             "Optimizer cost unavailable: no completed candidate yet.",
             "Cost belongs to the same candidate as best raw SSE; robust cost is not raw SSE.",
             "Production optimizer residual = prediction - observed; cost depends on the selected loss.",
             f"Forward calls: {s.forward_calls}; residual calls: {s.residual_calls}" if s.forward_calls is not None else
             "Forward/residual counts unavailable: hooks have not started or this backend is not instrumented.",
             "Forward: evaluator predictions plus dual fitter forwards; residual: optimizer residual calls."]
    c = s.root_cache
    if c is None:
        lines.append("Root cache statistics unavailable: no analysis cache snapshot yet.")
    else:
        lines.extend([
            f"Root requests: {c.hits+c.misses}; original solver batches: {c.misses}",
            f"Cache enabled={c.enabled}; hits={c.hits}; misses={c.misses}; evictions={c.evictions}",
            f"Entries={c.entries}/{c.max_entries}; estimated bytes={c.estimated_bytes}/{c.max_bytes}",
            f"Peak entries={c.peak_entries}; peak estimated bytes={c.peak_estimated_bytes}",
            "Estimated bytes are not RSS. Final entries describe occupancy before scope disposal."])
    lines.append("M2/M3 use seeded random starts; GUI initial values do not seed this staged MultiStart.")
    lines.append("Configured bounds, signs, baseline, seed and start counts still control the fit.")
    return display_fit_worker_text("\n".join(lines))
