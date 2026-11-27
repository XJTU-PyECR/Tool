"""Purpose-specific permission evidence; never a replacement for reliability/CI."""
from dataclasses import dataclass
from typing import ClassVar
from .common import Contract, SourceIdentity, number, require, strings, text

COMPONENT_POLICY = 'component-dct-eligibility/1'
COMPONENT_POLICY_V2 = 'component-dct-eligibility/2'
AUTOCORRELATION = 'RESIDUAL_AUTOCORRELATION_EXCEEDS_LIMIT'


@dataclass(frozen=True, slots=True, kw_only=True)
class EligibilityReason(Contract):
    KIND: ClassVar[str] = 'eligibility_reason'
    code: str
    origin: str
    message: str
    disposition: str
    causes: tuple[str, ...] = ()

    def __post_init__(self):
        Contract.__post_init__(self)
        for n in ('code','origin','message'):text(getattr(self,n),n)
        require(self.disposition in {'warning','blocking'},'unknown eligibility disposition')
        object.__setattr__(self,'causes',strings(self.causes,'causes'))


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureEligibility(Contract):
    KIND: ClassVar[str] = 'feature_eligibility'
    source: SourceIdentity | None
    can_run: bool
    warnings: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    reasons: tuple[EligibilityReason, ...]
    original_reliable: bool
    original_best_reliable: str | None
    original_gate_failures: tuple[str, ...]
    residual_lag1: float | None
    policy_version: str = COMPONENT_POLICY
    feature: str = 'component_dct'
    residual_limit: float | None = .30
    gate_confidence_level: float = .95

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.policy_version in {COMPONENT_POLICY, COMPONENT_POLICY_V2} and self.feature=='component_dct','unknown eligibility policy/feature')
        require(type(self.can_run) is bool and type(self.original_reliable) is bool,'eligibility booleans required')
        require(self.source is None or isinstance(self.source,SourceIdentity),'eligibility source required')
        if self.original_best_reliable is not None:text(self.original_best_reliable,'original_best_reliable')
        require(self.residual_limit==(.30 if self.policy_version==COMPONENT_POLICY else None)
                and self.gate_confidence_level==.95,'eligibility policy/threshold conflict')
        if self.residual_lag1 is not None:number(self.residual_lag1,'residual_lag1')
        for n in ('warnings','blocking_reasons','original_gate_failures'):
            object.__setattr__(self,n,strings(getattr(self,n),n))
        object.__setattr__(self,'reasons',tuple(self.reasons))
        require(all(isinstance(r,EligibilityReason) for r in self.reasons),'typed eligibility reasons required')
        for disposition,n in [('warning','warnings'),('blocking','blocking_reasons')]:
            require(getattr(self,n)==tuple(dict.fromkeys(r.message for r in self.reasons if r.disposition==disposition)),
                    'eligibility reasons/summary conflict')
        require(self.can_run==(not bool(self.blocking_reasons)),'eligibility decision conflicts with blockers')
        if self.can_run:
            require(self.source is not None,'usable eligibility requires source')
            self.source.require_parent()
            if not self.original_reliable or self.original_best_reliable!=self.source.model_id:
                if self.policy_version==COMPONENT_POLICY_V2:
                    require(not self.original_reliable and self.original_best_reliable is None
                            and not self.original_gate_failures,'v2 component inference contains blockers')
                    require(any(r.code=='LOCAL_CI_ASSUMPTIONS_UNVERIFIED' and r.origin=='diagnostics.evaluation'
                                and r.disposition=='warning' for r in self.reasons),
                            'v2 component permission needs traced CI-assumption evidence')
                    return
                require(not self.original_reliable and self.original_best_reliable is None
                        and self.residual_lag1 is not None and abs(self.residual_lag1)>.30,
                        'unreliable parent needs traced autocorrelation evidence')
                require(len(self.original_gate_failures)==1 and
                        self.original_gate_failures[0].partition(':')[0]==AUTOCORRELATION,
                        'unreliable parent contains an independent/unknown gate')
                for origin in ('diagnostics.gate_failures','core.gate_failures'):
                    require(any(r.code==AUTOCORRELATION and r.origin==origin and r.disposition=='warning'
                                for r in self.reasons),'autocorrelation origin missing')

    @property
    def parent_reliability(self):
        return 'reliable' if self.original_reliable else 'unreliable'

    def require_component(self, identity, parent_reliability):
        require(self.can_run,'; '.join(self.blocking_reasons) or 'component DCT unavailable')
        self.source.assert_matches(identity)
        require(parent_reliability==self.parent_reliability,'original parent reliability conflicts with eligibility')
