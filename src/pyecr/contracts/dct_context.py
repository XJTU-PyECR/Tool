"""Explicit spectrum-bound interpretation, separate from frozen inversion input.

The older signed service DctPhysicalContext remains run-time metadata. This
record is a reviewed R11 snapshot, with its own version and application history.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import ClassVar
from pyecr.config import ModelConfig, SolverConfig, BoundaryConfig, _freeze_json
from .common import Contract, require, digest, text, integer, exact_fields, canonical_sha256, number, vector, close_vectors
from .dct import DctSourceSnapshot, SourceKind
from .transport import TransportSummary


@dataclass(frozen=True,slots=True,kw_only=True)
class DctContextReview(Contract):
    KIND:ClassVar[str]='dct_context_review'
    target_sha256:str
    target_source:DctSourceSnapshot
    target_kind:str
    transport:TransportSummary
    model:ModelConfig
    solver:SolverConfig
    boundary:BoundaryConfig
    basis:str
    declared_given:bool
    interpretation_version:str='dct-physical-context/1'
    sequence:int=1
    modal_reference:Mapping[str,object]

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.interpretation_version=='dct-physical-context/1','Unknown interpretation version')
        digest(self.target_sha256,'spectrum identity');integer(self.sequence,'sequence',minimum=1)
        require(self.target_kind in ('whole','components','monotonic'),'Unknown DCT context target')
        require(isinstance(self.target_source,DctSourceSnapshot),'Missing frozen spectrum source')
        self.target_source.require_usable()
        require(isinstance(self.transport,TransportSummary),'Missing R11 transport snapshot')
        require(isinstance(self.model,ModelConfig) and isinstance(self.solver,SolverConfig) and isinstance(self.boundary,BoundaryConfig),'Missing physical configuration')
        self.model.validate();self.solver.validate();self.boundary.validate()
        require(self.solver.surface_mode==self.model.surface_mode,'Physical solver/model surface convention conflict')
        require(self.model.model_id.value==self.transport.model_id and self.model.geometry==self.transport.geometry,'Physical model/summary conflict')
        require(canonical_sha256(self.boundary.to_dict())==canonical_sha256(self.transport.parameter_evidence['boundary']) and self.model.boundary_mode==self.boundary.mode,'Physical boundary conflict')
        require(isinstance(self.declared_given,bool),'Declaration must be boolean')
        source=self.target_source.identity;parent=self.transport.source
        if parent is not None:
            require((source.dataset_sha256,source.recipe_sha256)==(parent.dataset_sha256,parent.recipe_sha256),'Physical context dataset/recipe identity conflict')
            require(source.config_sha256 is None or (source.config_sha256,source.model_id)==(parent.config_sha256,parent.model_id),'Physical context config identity conflict')
            if source.fit_id is not None:source.assert_matches(parent)
            if self.target_kind=='components':
                source.require_parent();require(self.transport.parent_reliability=='supported','Component context requires original reliable parent')
            require(not self.declared_given,'Fitted background cannot be labelled given')
        else:
            require(self.target_kind!='components','Independent given context cannot replace component parent gate')
            text(self.basis,'Given context basis');require(self.declared_given,'Explicit known/given background declaration required')
            if self.target_source.source_kind in (SourceKind.SIMULATION_CLEAN,SourceKind.SIMULATION_OBSERVED):
                require(self.target_source.simulation_request_sha256==self.transport.simulation_request_sha256,'Simulation request identity conflict')
        require(isinstance(self.modal_reference,Mapping),'Modal evidence must be a mapping')
        exact_fields(self.modal_reference,('algorithm_version','status','reasons','boundary','geometry','per_axis_budget','time_s','processes','response_scope','deconvolved','interpretation'),'modal evidence')
        m=self.modal_reference
        require(m['algorithm_version']=='production-rectangular-modes/1' and m['status'] in ('available','unsupported'),'Unknown modal method/status')
        require(tuple(m['time_s'])==self.target_source.time_s,'Modal observations differ from frozen spectrum source')
        require(canonical_sha256(m['geometry'])==canonical_sha256(self.model.geometry.to_dict()) and canonical_sha256(m['boundary'])==canonical_sha256(self.boundary.to_dict()),'Modal geometry/boundary conflict')
        require(m['response_scope']=='observed_effective_response' and m['deconvolved'] is False,'Unsupported material deconvolution claim')
        integer(m['per_axis_budget'],'modal budget',minimum=1);require(m['per_axis_budget']<=8,'Modal budget exceeds limit')
        require(tuple(p['process'] for p in m['processes'])==tuple(p.process for p in self.transport.processes) if m['status']=='available' else not m['processes'],'Modal process role conflict')
        for p,parent in zip(m['processes'],self.transport.processes):
            exact_fields(p,('process','label','D_cm2_s','k_cm_s','axes','modes','total_retained_mass',
                'production_remaining','truncated_remaining','truncation_difference','production_axis_terms',
                'zero_time_note','production_term_tol','production_max_terms','prefix_probe_time_s','production_axis_evidence'),'modal process')
            require(p['production_term_tol']==self.solver.relative_tolerance and p['production_max_terms']==self.solver.max_terms,'Production numerical settings conflict')
            require(len(p['production_axis_evidence'])==len(p['production_axis_terms'])==3,'Full production geometry evidence missing')
            for on,a,terms in zip(self.model.geometry.active_dimensions,p['production_axis_evidence'],p['production_axis_terms'],strict=True):
                require((a is not None)==on and (terms is not None)==on,'Production active-axis evidence conflict')
                if a is None:continue
                exact_fields(a,('n_terms_used','converged','coefficient_sum','coefficient_sum_error','truncation_error','asymptotic_roots_used'),'production axis evidence')
                require(a['n_terms_used']==terms and isinstance(a['converged'],bool) and isinstance(a['asymptotic_roots_used'],bool),'Production convergence evidence conflict')
                for k in ('coefficient_sum','coefficient_sum_error','truncation_error'):number(a[k],k,nonnegative=True)
            n=len(self.target_source.time_s)
            require(all(len(p[k])==n for k in ('production_remaining','truncated_remaining','truncation_difference')),'Modal full observation count conflict')
            require(p['D_cm2_s']==parent.D_cm2_s and p['k_cm_s']==parent.k_cm_s,'Modal kinetic parameter conflict')
            for k in ('production_remaining','truncated_remaining','truncation_difference'):vector(p[k],k)
            close_vectors(p['truncation_difference'],tuple(a-b for a,b in zip(p['production_remaining'],p['truncated_remaining'])),'Modal truncation difference')
            require(tuple(a['axis'] for a in p['axes'])==tuple(a for a,on in zip('xyz',self.model.geometry.active_dimensions) if on),'Modal active-axis conflict')
            for a in p['axes']:
                exact_fields(a,('axis','roots','coefficients','rates_s_inv','Bi','n_terms_used','coefficient_sum','coefficient_sum_error','converged','root_residual_max_abs','asymptotic_roots_used'),'modal axis')
                roots=vector(a['roots'],'roots',positive=True);rates=vector(a['rates_s_inv'],'rates',positive=True)
                coeff=vector(a['coefficients'],'coefficients',positive=True)
                require(len(roots)==len(rates)==len(coeff)==a['n_terms_used'] and len(roots)<=m['per_axis_budget'],'Modal prefix/budget conflict')
                half=self.model.geometry.full_dims_cm['xyz'.index(a['axis'])]/2
                close_vectors(rates,tuple(r*r*parent.D_cm2_s/half**2 for r in roots),'Modal rate/root/full geometry conflict')
            from itertools import product
            from math import prod,log10,exp,isclose
            combinations=list(product(*(range(a['n_terms_used']) for a in p['axes'])))
            require(len(p['modes'])==len(combinations) and {tuple(mode['indices']) for mode in p['modes']}==set(combinations),'Modal Cartesian prefix conflict')
            for mode in p['modes']:
                exact_fields(mode,('indices','rate_s_inv','tau_s','log10_tau','mass'),'modal term')
                rate=sum(a['rates_s_inv'][i] for a,i in zip(p['axes'],mode['indices'],strict=True))
                mass=prod(a['coefficients'][i] for a,i in zip(p['axes'],mode['indices'],strict=True))
                close_vectors((mode['rate_s_inv'],mode['tau_s'],mode['log10_tau'],mode['mass']),(rate,1/rate,log10(1/rate),mass),'Modal rate sum / mass product')
            require(isclose(number(p['total_retained_mass'],'retained mass',positive=True),sum(mode['mass'] for mode in p['modes']),rel_tol=1e-12),'Retained modal mass conflict')
            expected=tuple(sum(mode['mass']*exp(-t*mode['rate_s_inv']) for mode in p['modes']) for t in self.target_source.time_s)
            close_vectors(p['truncated_remaining'],expected,'Truncated modal reconstruction')
        require(bool(m['processes']) if m['status']=='available' else bool(m['reasons']),'Modal status/evidence conflict')
        object.__setattr__(self,'modal_reference',_freeze_json(m,path='modal_reference'))
