"""Freeze whole experimental observations without applying Analysis operations."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from pyecr.contracts import (DctSourceSnapshot, SignedNormalizationSpec,
    SignedNormalizationResult, SignedDctDiagnosticConfig, SourceIdentity)
from pyecr.contracts.common import ContractError, digest, integer, require
from pyecr.dct.signed_analysis import prepare_terminal_sensitivity
from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing import PreprocessConfig, normalize_signed_dct
from pyecr.preprocessing.signed_normalization import (applied_signed_normalization,
    cumulative_t0_s, raw_observations_sha256)


@dataclass(frozen=True, slots=True)
class PreparedSignedDctInput:
    """Existing contracts composed with the actual immutable input recipe."""
    processed: DctSourceSnapshot
    recipe: PreprocessConfig
    raw_sha256: str
    normalization: SignedNormalizationResult
    source: DctSourceSnapshot | None
    terminal_sources: tuple[DctSourceSnapshot, ...] = ()
    terminal_preparation_failure: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version') == 1, 'unknown prepared-input version')
        require(isinstance(self.processed, DctSourceSnapshot) and isinstance(self.recipe, PreprocessConfig)
                and isinstance(self.normalization, SignedNormalizationResult), 'prepared input types required')
        digest(self.raw_sha256, 'raw_sha256')
        object.__setattr__(self, 'terminal_sources', tuple(self.terminal_sources))
        require(self.source is None or isinstance(self.source, DctSourceSnapshot), 'source snapshot required')

    @property
    def mask(self):
        retained = set(self.processed.raw_indices)
        return tuple(i in retained for i in range(self.processed.raw_count))

    def to_dict(self):
        return dict(schema_version=self.schema_version, processed=self.processed.to_dict(),
            recipe=self.recipe.to_dict(), raw_sha256=self.raw_sha256, mask=list(self.mask),
            normalization=self.normalization.to_dict(),
            source=None if self.source is None else self.source.to_dict(),
            terminal_sources=[s.to_dict() for s in self.terminal_sources],
            terminal_preparation_failure=self.terminal_preparation_failure)


def prepare_signed_experiment(dataset: EcrDataset, recipe: PreprocessConfig, *,
        normalization_spec: SignedNormalizationSpec | None = None,
        diagnostic_config=SignedDctDiagnosticConfig()) -> PreparedSignedDctInput:
    """Prepare a DCT copy using P02, or verify an already accepted P02 profile.

    A missing declaration never selects a suggested tail. Default terminal rows
    come exclusively from a current, explicitly accepted final stability window.
    All processed observations and their exact raw mapping are retained.
    """
    require(isinstance(dataset, EcrDataset) and isinstance(recipe, PreprocessConfig),
            'whole signed-DCT requires an actual dataset and recipe')
    require(isinstance(diagnostic_config, SignedDctDiagnosticConfig), 'diagnostic config required')
    require(dataset.metadata.get('time_unit') == 's',
            'signed-DCT requires processed time in seconds; convert explicitly in preprocessing')
    identity = SourceIdentity.capture(dataset, recipe)  # No physical config or Fit prerequisite.
    t0 = cumulative_t0_s(dataset)
    terminal_failure = None
    applied = dataset.metadata.get('normalization_mode') == 'signed_terminal'
    if applied:
        require(normalization_spec is None,
                'already signed-normalized: Undo/Reset explicitly before changing its terminal')
        norm = applied_signed_normalization(dataset, recipe=recipe)
        processed = DctSourceSnapshot.capture_experiment(dataset, recipe, normalization=norm, t0_s=t0)
        source = processed
        # No implicit Undo/inverse reconstruction to invent missing input context.
        variants = ()
    else:
        processed = DctSourceSnapshot.capture_experiment(dataset, recipe, t0_s=t0)
        if normalization_spec is None:
            windows = dataset.metadata.get('stability_windows', {})
            window = windows.get('final') if isinstance(windows, Mapping) else None
            require(isinstance(window, Mapping),
                    'accept a final stability window or explicitly declare an external known terminal')
            normalization_spec = SignedNormalizationSpec(identity=identity,
                raw_count=len(dataset.raw_time), raw_indices=processed.raw_indices,
                signal_type=processed.signal_type, signal_unit=processed.signal_unit,
                reference_kind='accepted_final_window', reference_rows=tuple(window['raw_indices']),
                reference_evidence='Current explicitly accepted final stability window; DCT copy only', t0_s=t0)
        identity.assert_matches(normalization_spec.identity)
        norm = normalize_signed_dct(dataset, normalization_spec, recipe=recipe)
        source = (DctSourceSnapshot.capture_experiment(dataset, recipe, normalization=norm, t0_s=t0)
                  if norm.state.ready else None)
        variants = ()
        if source is not None and diagnostic_config.terminal_offsets_in_S:
            try:
                variants = prepare_terminal_sensitivity(dataset, recipe, norm,
                    offsets_in_S=diagnostic_config.terminal_offsets_in_S)
            except ContractError as exc:
                # A failed optional hypothesis is evidence about that control;
                # it does not disqualify an otherwise valid base observation.
                terminal_failure = f'NOT_RUN: terminal controls could not be prepared: {exc}'
    require(tuple(np.flatnonzero(dataset.mask)) == processed.raw_indices
            and processed.time_s == tuple(dataset.processed_time), 'complete observation mapping conflict')
    return PreparedSignedDctInput(processed=processed, recipe=recipe,
        raw_sha256=raw_observations_sha256(dataset), normalization=norm, source=source,
        terminal_sources=variants, terminal_preparation_failure=terminal_failure)
