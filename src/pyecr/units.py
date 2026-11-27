"""Canonical model units and one registry for imported signal units."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnitConvention:
    """Names of the canonical units accepted by public APIs."""

    time: str = "s"
    length: str = "cm"
    diffusion_coefficient: str = "cm^2/s"
    surface_exchange_coefficient: str = "cm/s"
    signal: str = "signal"
    dimensionless: str = "1"


DEFAULT_UNITS = UnitConvention()


@dataclass(frozen=True, slots=True)
class SignalUnit:
    """Stable serialized key, GUI label, and scale to its signal reference unit."""

    key: str
    label: str
    factor: float | None
    aliases: tuple[str, ...] = ()


_SIGNAL_UNITS: dict[str, tuple[SignalUnit, ...]] = {
    "resistance": (
        SignalUnit("ohm", "Ω", 1.0, ("ohm", "ohms", "Ω", "ω")),
        SignalUnit("kiloohm", "kΩ", 1.0e3, ("kohm", "kOhm", "kΩ", "kω")),
        SignalUnit("megaohm", "MΩ", 1.0e6, ("Mohm", "megohm", "MΩ", "Mω")),
    ),
    "conductance": (
        SignalUnit("siemens", "S", 1.0, ("S", "siemens", "Ω^-1", "ohm^-1")),
        SignalUnit("millisiemens", "mS", 1.0e-3, ("mS", "millisiemens")),
        SignalUnit("microsiemens", "µS", 1.0e-6, ("µS", "μS", "uS", "microsiemens")),
    ),
    "conductivity": (
        SignalUnit("S_per_cm", "S/cm", 1.0, ("S/cm", "S cm^-1", "Ω^-1 cm^-1", "ohm^-1 cm^-1")),
        SignalUnit("mS_per_cm", "mS/cm", 1.0e-3, ("mS/cm", "mS cm^-1")),
        SignalUnit("uS_per_cm", "µS/cm", 1.0e-6, ("µS/cm", "μS/cm", "uS/cm")),
        SignalUnit("S_per_m", "S/m", 1.0e-2, ("S/m",)),
    ),
    "normalized": (SignalUnit("dimensionless", "1", 1.0, ("1", "dimensionless")),),
    "arbitrary": (
        SignalUnit("arbitrary", "arbitrary", None, ("arbitrary",)),
        SignalUnit("custom", "Custom…", None, ("Custom…",)),
    ),
}
SIGNAL_TYPES = tuple(_SIGNAL_UNITS)


def _unit_token(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("signal unit must be a string")
    token = "".join(value.strip().split()).replace("μ", "µ")
    if not token:
        raise ValueError("signal unit must not be empty")
    return token


_BY_KEY = {unit.key: unit for options in _SIGNAL_UNITS.values() for unit in options}
_BY_ALIAS: dict[str, SignalUnit] = {}
for _unit in _BY_KEY.values():
    for _alias in (_unit.key, _unit.label, *_unit.aliases):
        _token = _unit_token(_alias)
        if _token in _BY_ALIAS and _BY_ALIAS[_token] != _unit:
            raise RuntimeError(f"ambiguous signal unit alias: {_alias}")
        _BY_ALIAS[_token] = _unit
        # English names are case-insensitive, unlike SI symbols (mS versus MS).
        if _token.isascii() and _token.isalpha():
            _BY_ALIAS.setdefault(_token.lower(), _unit)


def get_units(signal_type: str) -> tuple[SignalUnit, ...]:
    """Return ordered compatible units; `unknown` is a legacy arbitrary type."""

    kind = str(signal_type).strip().lower()
    if kind == "unknown":
        kind = "arbitrary"
    try:
        return _SIGNAL_UNITS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown signal type: {signal_type!r}") from exc


def get_default_unit(signal_type: str) -> SignalUnit:
    return get_units(signal_type)[0]


def normalize_alias(unit: str) -> str:
    """Map a canonical key, display label, or known legacy alias to its key."""

    token = _unit_token(unit)
    descriptor = _BY_ALIAS.get(token)
    if descriptor is None and token.isascii() and token.isalpha():
        descriptor = _BY_ALIAS.get(token.lower())
    if descriptor is None:
        raise ValueError(f"unknown signal unit: {unit!r}")
    return descriptor.key


def unit_label(unit: str) -> str:
    return _BY_KEY[normalize_alias(unit)].label


def validate_pair(signal_type: str, unit: str) -> str:
    """Return canonical key or reject a known but dimensionally wrong unit.

    Custom labels are allowed only for arbitrary/unknown signals. They have no
    physical conversion factor and must be explicitly selected in the GUI.
    """

    allowed = {descriptor.key for descriptor in get_units(signal_type)}
    try:
        key = normalize_alias(unit)
    except ValueError:
        if "custom" in allowed and _unit_token(unit):
            return "custom"
        raise
    if key not in allowed:
        raise ValueError(f"signal unit {unit!r} is incompatible with signal type {signal_type!r}")
    return key


def factor_to_reference(unit: str) -> float:
    """Scale to Ω, S, or S/cm as appropriate; no factor for arbitrary units."""

    descriptor = _BY_KEY[normalize_alias(unit)]
    if descriptor.factor is None:
        raise ValueError(f"signal unit {unit!r} has no physical conversion factor")
    return descriptor.factor


# Physical input units are separate from source signal declarations. These
# factors are used at the Simulation input boundary, never on signal arrays.
@dataclass(frozen=True, slots=True)
class PhysicalUnit:
    key: str
    label: str
    factor: float


_PHYSICAL_UNITS = {
    'length': (PhysicalUnit('cm','cm',1.), PhysicalUnit('mm','mm',.1), PhysicalUnit('m','m',100.)),
    'diffusion': (PhysicalUnit('cm^2/s','cm²/s',1.), PhysicalUnit('mm^2/s','mm²/s',.01), PhysicalUnit('m^2/s','m²/s',1e4)),
    'exchange': (PhysicalUnit('cm/s','cm/s',1.), PhysicalUnit('mm/s','mm/s',.1), PhysicalUnit('m/s','m/s',100.)),
}
PRESSURE_UNIT_CHOICES = ('bar','Pa','kPa','atm')  # source declaration, not automatic conversion


def get_physical_units(dimension: str) -> tuple[PhysicalUnit, ...]:
    if dimension == 'time':
        # Reuse the exact existing Analysis/data import factors and aliases.
        # Lazy import avoids a units <-> dataset import cycle.
        from pyecr.io.dataset import _TIME_FACTORS_TO_SECONDS
        return tuple(PhysicalUnit(k,k,_TIME_FACTORS_TO_SECONDS[k]) for k in ('s','ms','min','h','us'))
    try:return _PHYSICAL_UNITS[dimension]
    except KeyError as exc:raise ValueError('Unsupported physical dimension: '+str(dimension)) from exc


def physical_unit(dimension: str, unit: str) -> PhysicalUnit:
    if dimension == 'time':
        from pyecr.io.dataset import _canonical_time_unit
        unit = _canonical_time_unit(unit)
    if not isinstance(unit,str):raise TypeError('Physical unit must be text')
    token=''.join(unit.split()).replace('²','^2')
    for option in get_physical_units(dimension):
        if token==option.key:return option
    raise ValueError(f'Unit {unit!r} is incompatible with {dimension}; choose a registered unit')


def to_canonical(value: float, dimension: str, unit: str) -> float:
    from math import isfinite
    from numbers import Real
    if isinstance(value,bool) or not isinstance(value,Real) or not isfinite(value):
        raise ValueError('Physical input must be finite and numeric')
    from fractions import Fraction
    ratio=Fraction(str(physical_unit(dimension,unit).factor))
    # The registry ratios are exact decimal powers / integer seconds. Division
    # by 10, for example, avoids an extra rounding from a binary 0.1 factor.
    result=float(value)*ratio.numerator/ratio.denominator
    if not isfinite(result):raise ValueError('Physical unit conversion overflow')
    return result


def parameter_dimension(unit: str) -> str | None:
    return {'cm^2/s':'diffusion','cm/s':'exchange'}.get(unit)
