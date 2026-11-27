"""Explicit Send and simulation-derived parent evidence. No forward or Fit."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from pyecr.config import _freeze_json
from .common import Contract,require,text,digest,canonical_sha256,exact_fields
from .simulation import SimulationResult

CHANNEL_KINDS={'clean':'simulation_clean','observed':'simulation_observed'}
ORIGIN_ID_FIELDS=('version','source_kind','request_sha256','result_sha256','raw_indices',
                  'bundle_manifest_sha256','bundle_table')

@dataclass(frozen=True,slots=True,kw_only=True)
class SimulationSendReceipt(Contract):
    KIND: ClassVar[str]='simulation_send_receipt'
    review_id: str
    channel: str
    target_action: str
    target_before_sha256: str
    origin_sha256: str
    request_sha256: str
    result_sha256: str
    raw_sha256: str
    send_identity: str

    def identity_fields(self):
        return dict(version='simulation-send/1',review_id=self.review_id,channel=self.channel,
            target_action=self.target_action,target_before_sha256=self.target_before_sha256,
            origin_sha256=self.origin_sha256,request_sha256=self.request_sha256,
            result_sha256=self.result_sha256,raw_sha256=self.raw_sha256)

    def __post_init__(self):
        Contract.__post_init__(self);text(self.review_id,'Send review id')
        require(self.channel in CHANNEL_KINDS,'Send requires clean or observed')
        require(self.target_action in {'require_empty','replace_current'},'Send requires explicit target action')
        for name in ('target_before_sha256','origin_sha256','request_sha256','result_sha256','raw_sha256','send_identity'):
            digest(getattr(self,name),name)
        require(self.send_identity==canonical_sha256(self.identity_fields()),'Send identity/content conflict')

@dataclass(frozen=True,slots=True,kw_only=True)
class SimulationParentEvidence(Contract):
    KIND: ClassVar[str]='simulation_component_parent'
    result: SimulationResult
    send: SimulationSendReceipt
    origin_identity: Mapping[str,object]
    interpretation: str='simulation-derived estimated parent; components are model-derived, not independent observations or noise'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.result,SimulationResult) and self.result.state.ready,'Complete Generate result required')
        require(isinstance(self.send,SimulationSendReceipt),'Explicit reviewed Send receipt required')
        o=self.origin_identity;exact_fields(o,ORIGIN_ID_FIELDS,'simulation parent origin')
        require(o['version']=='simulation-analysis-source/1','Unknown simulation origin version')
        kind=CHANNEL_KINDS[self.send.channel]
        require(o['source_kind']==kind,'Simulation parent channel conflict')
        require(o['request_sha256']==self.send.request_sha256==self.result.request_sha256
            and o['result_sha256']==self.send.result_sha256==self.result.content_sha256,'Generate/Send result identity conflict')
        table='simulation_truth.csv' if self.send.channel=='clean' else 'simulation_observations.csv'
        require(o['bundle_table']=='tables/'+table,'Simulation parent source table conflict')
        require(tuple(o['raw_indices'])==tuple(range(len(self.result.time_s)))
            and all(type(i) is int for i in o['raw_indices']),'Simulation parent requires all raw rows')
        digest(o['bundle_manifest_sha256'],'Simulation manifest hash')
        require(canonical_sha256(o)==self.send.origin_sha256,'Simulation origin/Send conflict')
        require(self.interpretation=='simulation-derived estimated parent; components are model-derived, not independent observations or noise',
            'Simulation derived interpretation conflict')
        object.__setattr__(self,'origin_identity',_freeze_json(dict(o),path='simulation_parent_origin'))
