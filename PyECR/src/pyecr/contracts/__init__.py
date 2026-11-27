"""P01 public interfaces. Data contracts only; no scientific algorithms run here."""
from .common import ContractError, IdentityConflictError, ResultState, SourceIdentity, canonical_sha256
from .normalization import SignedNormalizationSpec, SignedNormalizationResult
from .eligibility import FeatureEligibility, EligibilityReason
from .signed_solve import SignedDctSolveDiagnostics
from .signed_priors import SignedDctPriorSpec,SignedPriorIteration,SignedDctPriorEvidence
from .dct import SourceKind, DctSourceSnapshot, SignedDctConfig, SignedDctResult, ComponentDctBundle
from .simulation import SimulationRequest, SimulationResult, model_support_matrix, SUPPORT_MATRIX_VERSION
from .simulation_units import SimulationUnitInputs
from .component_source import ComponentSourceCapture
from .simulation_parent import SimulationSendReceipt,SimulationParentEvidence
from .component_spectrum import ComponentDctConfig, ComponentSpectrumResult
from .signed_analysis import (SignedDctDiagnosticConfig, SignedPeakDiagnostic, SignedLambdaCandidate,
                              SignedDctComparison, SignedDctAnalysis)

__all__ = ['ContractError', 'IdentityConflictError', 'ResultState', 'SourceIdentity',
           'canonical_sha256', 'SignedNormalizationSpec', 'SignedNormalizationResult',
           'SourceKind', 'DctSourceSnapshot', 'SignedDctConfig', 'SignedDctResult',
           'ComponentDctBundle', 'SimulationRequest', 'SimulationResult', 'SignedDctSolveDiagnostics',
           'model_support_matrix', 'SUPPORT_MATRIX_VERSION', 'read_contract', 'read_contract_json', 'read_dct_payload']
__all__ += ['SignedDctDiagnosticConfig', 'SignedPeakDiagnostic', 'SignedLambdaCandidate',
            'SignedDctComparison', 'SignedDctAnalysis']

_CONTRACTS = {cls.KIND: cls for cls in (ResultState, SourceIdentity, SignedNormalizationSpec,
    SignedNormalizationResult, DctSourceSnapshot, SignedDctConfig, SignedDctResult,
    ComponentDctBundle, SimulationRequest, SimulationResult, SignedDctSolveDiagnostics)}
__all__ += ['SignedDctPriorSpec','SignedPriorIteration','SignedDctPriorEvidence']
_CONTRACTS.update({c.KIND:c for c in (SignedDctPriorSpec,SignedPriorIteration,SignedDctPriorEvidence)})
_CONTRACTS.update({cls.KIND: cls for cls in (SignedDctDiagnosticConfig, SignedPeakDiagnostic,
                   SignedLambdaCandidate, SignedDctComparison, SignedDctAnalysis)})
_CONTRACTS[ComponentSourceCapture.KIND] = ComponentSourceCapture
_CONTRACTS[SimulationUnitInputs.KIND]=SimulationUnitInputs
__all__ += ['SimulationUnitInputs']
_CONTRACTS.update({c.KIND:c for c in (FeatureEligibility,EligibilityReason)})
__all__ += ['FeatureEligibility','EligibilityReason']
_CONTRACTS.update({c.KIND:c for c in (SimulationSendReceipt,SimulationParentEvidence)})
__all__ += ['SimulationSendReceipt','SimulationParentEvidence']
__all__ += ['ComponentSourceCapture']
__all__ += ['ComponentDctConfig','ComponentSpectrumResult']
_CONTRACTS.update({cls.KIND:cls for cls in (ComponentDctConfig,ComponentSpectrumResult)})

# R01 metadata only. Existing Config/Fit/gate/solver dispatch is unchanged.
from .fit_controls import (ConfidencePolicy, ReliabilityPolicy, TerminationPolicy,
                           FitControlSpec, FitControlBinding, parse_confidence_percent,
                           read_fit_controls)
from .fit_trace import AcceptedIterate, AcceptedTrace
__all__ += ['ConfidencePolicy','ReliabilityPolicy','TerminationPolicy','FitControlSpec',
            'FitControlBinding','parse_confidence_percent','read_fit_controls','AcceptedIterate','AcceptedTrace']
_CONTRACTS.update({cls.KIND:cls for cls in (ConfidencePolicy,ReliabilityPolicy,TerminationPolicy,
                   FitControlSpec,FitControlBinding,AcceptedIterate,AcceptedTrace)})
from pyecr.fitting.solver_strategy import SolverBudget as _SolverBudget
_CONTRACTS['solver_budget'] = _SolverBudget

# R10 independent post-stop diagnostic evidence; never an optimizer/gate policy.
from .noise import NoiseReference,NoiseComparison
__all__ += ['NoiseReference','NoiseComparison']
_CONTRACTS.update({cls.KIND:cls for cls in (NoiseReference,NoiseComparison)})

from .transport import TransportProcess,TransportAxis,TransportInterval,TransportSummary
__all__ += ['TransportProcess','TransportAxis','TransportInterval','TransportSummary']
_CONTRACTS.update({cls.KIND:cls for cls in (TransportProcess,TransportAxis,TransportInterval,TransportSummary)})
from .dct_context import DctContextReview
__all__ += ['DctContextReview']
_CONTRACTS[DctContextReview.KIND] = DctContextReview


def read_contract(payload):
    from collections.abc import Mapping
    if (not isinstance(payload, Mapping) or not isinstance(payload.get('contract_kind'), str)
            or payload['contract_kind'] not in _CONTRACTS):
        raise ContractError('unknown/missing contract_kind; no fallback to a legacy mode')
    return _CONTRACTS[payload['contract_kind']].from_dict(payload)


def read_dct_payload(payload):
    """Explicit standalone dispatch. Project/GUI persistence integration is P10."""
    from collections.abc import Mapping
    from pyecr.services.dct_service import DctPresentation
    if not isinstance(payload, Mapping):
        raise ContractError('DCT payload must be an object')
    if 'contract_kind' in payload:
        if payload['contract_kind'] != SignedDctResult.KIND:
            raise ContractError('expected signed_dct_result; do not reinterpret another contract')
        return SignedDctResult.from_dict(payload)
    if 'mode' in payload:
        raise ContractError('untagged new DCT mode is rejected; explicit versioned contract required')
    return DctPresentation.from_dict(payload)


def read_contract_json(payload):
    from .common import strict_json_loads
    return read_contract(strict_json_loads(payload))
