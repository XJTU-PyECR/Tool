"""Observation ledger validation only; reading never executes a random generator."""
from .common import canonical_sha256, exact_fields, require
from .simulation import SIMULATION_OBSERVATION_VERSION, SIMULATION_POINT_BUDGET

FIELDS = ('version', 'source_kind', 'clean_source_kind', 'request_sha256', 'job_id',
    'name', 'sequence_number', 'scientific_time_sha256', 'sampling_kind', 'point_count',
    'point_budget', 'display_grid_used', 'noise_model', 'noise_sd', 'signal_unit',
    'actual_seed', 'rng_algorithm', 'seed_sequence_spawn_key', 'draw_count', 'numpy_version',
    'component_noise_policy')


def observation_ledger(request, numpy_version):
    """Metadata shared by producer and reader, never a second noise algorithm."""
    noisy = request.noise_sd > 0
    return dict(version=SIMULATION_OBSERVATION_VERSION, source_kind='simulation_observed',
        clean_source_kind='simulation_clean', request_sha256=request.content_sha256,
        job_id=request.job_id, name=request.name, sequence_number=request.sequence_number,
        scientific_time_sha256=canonical_sha256(request.time_s), sampling_kind=request.sampling_kind,
        point_count=len(request.time_s), point_budget=SIMULATION_POINT_BUDGET, display_grid_used=False,
        noise_model='gaussian_iid' if noisy else 'none', noise_sd=request.noise_sd,
        signal_unit=request.signal_unit, actual_seed=request.seed, rng_algorithm=request.rng_algorithm,
        seed_sequence_spawn_key=(request.sequence_number,) if noisy else (),
        draw_count=len(request.time_s) if noisy else 0, numpy_version=numpy_version,
        component_noise_policy='clean_model_contributions_only; one_noise_vector_added_to_total')


def validate_observation_evidence(result):
    e = result.observation_evidence
    exact_fields(e, FIELDS, 'simulation v4 observation evidence')
    expected = observation_ledger(result.request, result.numerical_evidence['software_versions']['numpy'])
    require(canonical_sha256(e) == canonical_sha256(expected),
        'simulation observation identity/grid/seed/units ledger conflict')
    require(result.signal_observed == tuple(c+n for c,n in zip(result.signal_clean, result.noise)),
        'simulation observed must equal clean + actual noise at every scientific point')
