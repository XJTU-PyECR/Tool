"""Capture row provenance once, without changing P53 or running inference."""
from datetime import datetime, timezone
import numpy as np
from pyecr.preprocessing import PreprocessConfig
from pyecr.report.snapshot import dataset_sha256, _sha256_json


def capture_export_context(session, snapshot):
    source = session.preprocessing.source
    recipe = session.preprocessing.config
    previous = source
    reasons = {str(int(i)): {"reason": "initial_mask_source_not_recorded", "step_ids": []}
               for i in np.flatnonzero(~source.mask)}
    for position, step in enumerate(recipe.steps, 1):
        future = tuple(s for s in recipe.steps[position:] if s.operation == "unmask_points")
        current = PreprocessConfig((*recipe.steps[:position], *future)).replay(source)
        before, after = set(np.flatnonzero(previous.mask)), set(np.flatnonzero(current.mask))
        for i in after - before:
            reasons.pop(str(int(i)), None)
        for i in before - after:
            reasons[str(int(i))] = {"reason": step.operation, "step_ids": [position]}
        previous = current
    for name in ("mask", "processed_time", "processed_signal"):
        if not np.array_equal(getattr(previous, name), getattr(snapshot.dataset, name)):
            raise ValueError("captured row audit/recipe disagrees with snapshot data")
    return {
        "version": "p71-export-context/1.0.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": snapshot.dataset_sha256,
        "config_sha256": _sha256_json(snapshot.analysis_config.to_dict()),
        "fit_result_sha256": _sha256_json(snapshot.fit_result.to_dict()),
        "free_parameter_order": list(snapshot.fit_result.params),
        "source_dataset_sha256": dataset_sha256(source),
        "revision": {"data": session.data_revision, "config": session.config_revision, "physical": session.physical_input_revision},
        "project_metadata": dict(session.project_metadata),
        "preprocess_recipe": recipe.to_dict(),
        "row_reasons": reasons,
        "row_audit_status": "incomplete" if any(v["reason"] == "initial_mask_source_not_recorded" for v in reasons.values()) else "captured_recipe_exact",
    }


def validate_export_context(context, snapshot):
    if not context:
        return
    if context.get("version") != "p71-export-context/1.0.0":
        raise ValueError("unsupported export_context version")
    from pyecr.report.scientific_evidence import HISTORY_KEY,validate_execution_history
    metadata=context.get('project_metadata',{})
    if HISTORY_KEY in metadata:validate_execution_history(metadata[HISTORY_KEY])
    if (context.get("dataset_sha256") != snapshot.dataset_sha256
            or context.get("config_sha256") != _sha256_json(snapshot.analysis_config.to_dict())
            or context.get("fit_result_sha256") != _sha256_json(snapshot.fit_result.to_dict())):
        raise ValueError("export_context source binding mismatch")
    excluded = set(map(int, np.flatnonzero(~snapshot.dataset.mask)))
    if set(map(int, context.get("row_reasons", {}))) != excluded:
        raise ValueError("export_context row audit binding mismatch")
    names = context.get("free_parameter_order", ())
    if len(names) != len(snapshot.fit_result.params) or set(names) != set(snapshot.fit_result.params):
        raise ValueError("export_context parameter column binding mismatch")
