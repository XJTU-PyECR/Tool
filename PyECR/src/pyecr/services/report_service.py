"""Thin orchestration service for reproducible analysis reports.

The service owns session-validity and seed-selection policy.  Snapshot
serialization, figures, report files, hashing, and atomic export remain in the
``pyecr.report`` core.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Mapping

from pyecr.report import AnalysisSnapshot, PdfRenderer, ReportBundle
from pyecr.services.workflow import AnalysisSession
from pyecr.services.m1_deep_uncertainty import ExperimentalM1UncertaintyResult

__all__ = [
    "ReportArtifactStatus",
    "ReportExportResult",
    "ReportService",
]


@dataclass(frozen=True, slots=True)
class ReportArtifactStatus:
    """One exported artifact after ReportBundle integrity verification."""

    relative_path: str
    sha256: str
    size_bytes: int
    status: str = "verified"


@dataclass(frozen=True, slots=True)
class ReportExportResult:
    """Verified report output suitable for GUI presentation."""

    bundle: ReportBundle
    artifacts: tuple[ReportArtifactStatus, ...]
    verified: bool
    verification_status: str
    pdf_status: str

    @property
    def root(self) -> Path:
        return self.bundle.root

    @property
    def dataset_sha256(self) -> str:
        return self.bundle.dataset_sha256


class ReportService:
    """Capture current analysis state and delegate report export to the core."""

    def capture_snapshot(
        self,
        session: AnalysisSession,
        *,
        seed: int | None = None,
        include_export_context: bool = False,
        execution_evidence=None,
        control_request=None,
    ) -> AnalysisSnapshot:
        """Capture a reproducible snapshot from one current, non-stale fit."""

        if not isinstance(session, AnalysisSession):
            raise TypeError("session must be an AnalysisSession")
        dataset = session.dataset
        config = session.current_analysis_config
        fit_result = session.fit_result
        if dataset is None or config is None or fit_result is None:
            raise RuntimeError(
                "current analysis report requires a dataset, AnalysisConfig, and fit"
            )
        if session.fit_stale:
            raise RuntimeError(
                "current analysis report cannot be generated from an Out of date Fit"
            )
        if session.fit_revision is None:
            raise RuntimeError(
                "current analysis fit has no recorded workflow revision"
            )
        if session.fit_revision != session.current_revision:
            raise RuntimeError(
                "current analysis fit revision does not match dataset/config revision"
            )

        diagnostics: object
        if session.diagnostics is None:
            diagnostics = _diagnostics_unavailable(
                "Diagnostics have not been run for this fit."
            )
        elif (
            session.diagnostics_stale
            or session.diagnostics_fit_revision != session.fit_revision
        ):
            diagnostics = _diagnostics_unavailable(
                "Diagnostics are Out of date or belong to another Fit revision."
            )
        else:
            diagnostics = session.diagnostics

        map_context = session.error_map_context
        map_current = bool(
            session.error_map_result is not None
            and session.error_map_config is not None
            and not session.error_map_stale
            and map_context is not None
            and map_context.dataset is dataset
            and map_context.analysis_config is config
            and map_context.fit_result is fit_result
            and map_context.revision == (
                session.current_revision.data_revision,
                session.current_revision.config_revision,
                session.current_revision.physical_input_revision,
            )
        )

        # P45's current-result gate is the only source of truth. A retained
        # stale H/O map remains viewable in the GUI but is never captured here.
        from pyecr.services.dual_error_map_service import DualErrorMapService

        dual_result = DualErrorMapService.current_result(session)
        dual_config = session.dual_error_map_config if dual_result is not None else None

        snapshot = AnalysisSnapshot.capture(
            dataset,
            config,
            fit_result,
            diagnostics,
            seed=_resolve_seed(seed, fit_result),
            error_map_result=session.error_map_result if map_current else None,
            error_map_config=session.error_map_config if map_current else None,
            dual_error_map_result=dual_result if dual_config is not None else None,
            dual_error_map_config=dual_config,
        )
        if include_export_context:
            from pyecr.report.export_capture import capture_export_context
            snapshot = replace(snapshot, export_context=capture_export_context(session, snapshot))
        declared=fit_result.diagnostics.get('analysis_artifacts',{}).get('model_name',fit_result.diagnostics.get('model_id'))
        has_source=session.preprocessing is not None and declared is not None
        if has_source:
            from pyecr.report.scientific_evidence import capture_scientific
            evidence=capture_scientific(session,snapshot,execution=execution_evidence,control_request=control_request)
            snapshot=replace(snapshot,schema_version=3,scientific_evidence=evidence)
        elif any(v is not None for v in (session.selected_local_intervals,session.noise_reference,session.execution_evidence,execution_evidence,control_request)):
            raise ValueError('Scientific result evidence requires its original preprocessing recipe/source')
        # Historical API sessions may lack a recipe entirely. Keep their old
        # snapshot schema and endpoints; do not invent a recipe/model identity.
        return snapshot

    def export_report_bundle(
        self,
        snapshot: AnalysisSnapshot,
        destination: str | Path,
        *,
        pdf_renderer: PdfRenderer | None = None,
        experimental_uncertainty: ExperimentalM1UncertaintyResult | None = None,
        export_options=None,
        export_scope: str = "captured",
        batch_result=None,
        cancel_check=None,
        before_publish=None,
        progress_callback=None,
    ) -> ReportExportResult:
        """Export through ReportBundle and verify hashes before returning."""

        if not isinstance(snapshot, AnalysisSnapshot):
            raise TypeError("snapshot must be an AnalysisSnapshot")
        bundle = ReportBundle.export(
            snapshot,
            destination,
            pdf_renderer=pdf_renderer,
            experimental_uncertainty=experimental_uncertainty,
            export_options=export_options, export_scope=export_scope, batch_result=batch_result,
            cancel_check=cancel_check, before_publish=before_publish, progress_callback=progress_callback,
        )
        bundle.verify()
        artifacts = tuple(
            ReportArtifactStatus(
                relative_path=relative_path,
                sha256=sha256,
                size_bytes=(bundle.root / relative_path).stat().st_size,
            )
            for relative_path, sha256 in sorted(bundle.artifacts.items())
        )
        return ReportExportResult(
            bundle=bundle,
            artifacts=artifacts,
            verified=True,
            verification_status="verified",
            pdf_status=(
                "optional/not configured"
                if pdf_renderer is None and not (export_options is not None and export_options.preset.value == "publication")
                else "generated and verified"
            ),
        )

    def export_current(self, session, destination, *, export_options, snapshot=None, **kwargs):
        """Validate current result both at capture and immediately before commit."""
        captured = self.capture_snapshot(session, include_export_context=True) if snapshot is None else snapshot
        revision = session.current_revision
        def guard():
            if (session.fit_stale or session.fit_revision != session.current_revision
                    or session.current_revision != revision or session.dataset is not captured.dataset
                    or session.current_analysis_config != captured.analysis_config or session.fit_result is not captured.fit_result):
                raise RuntimeError("Out of date current analysis; export was not published")
        guard()
        return self.export_report_bundle(captured, destination, export_options=export_options,
                                         export_scope="current", before_publish=guard, **kwargs)


def _diagnostics_unavailable(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "warnings": [],
        "gate_failures": [],
    }


def _resolve_seed(explicit, fit_result):
    from pyecr.fitting.reproducibility import require_reproducibility
    return require_reproducibility(fit_result,explicit)
