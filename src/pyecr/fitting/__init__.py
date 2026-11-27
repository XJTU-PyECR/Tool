"""Parameter estimation workflows."""

from pyecr.fitting.base import FitCandidate, FitResult, ModelProtocol
from pyecr.fitting.objective import ObjectiveContext, ObjectiveEvaluation, SingleObjectiveEvaluator
from pyecr.fitting.dual_objective import (
    DualHOObjectiveContext,
    DualHOObjectiveEvaluation,
    DualHOObjectiveEvaluator,
    DualObjectiveRevision,
    OptimizerLossMetadata,
)
from pyecr.fitting.dual_ho import (
    DualHOFitConfig,
    DualHOFitResult,
    FitCandidateSummary,
    LabelConstraint,
    M3TriggerEvidence,
    fit_dual_ho,
)
from pyecr.fitting.global_fit import (
    GlobalCurve,
    GlobalFitConfig,
    GlobalFitResult,
    fit_global_ecr,
)
from pyecr.fitting.single import fit_single_ecr, prepare_single_objective

__all__ = [
    "DualHOFitConfig",
    "DualHOFitResult",
    "DualHOObjectiveContext",
    "DualHOObjectiveEvaluation",
    "DualHOObjectiveEvaluator",
    "DualObjectiveRevision",
    "FitCandidateSummary",
    "FitCandidate",
    "FitResult",
    "GlobalCurve",
    "GlobalFitConfig",
    "GlobalFitResult",
    "LabelConstraint",
    "M3TriggerEvidence",
    "ModelProtocol",
    "ObjectiveContext",
    "ObjectiveEvaluation",
    "OptimizerLossMetadata",
    "SingleObjectiveEvaluator",
    "fit_dual_ho",
    "fit_global_ecr",
    "fit_single_ecr",
    "prepare_single_objective",
]
