"""Axis titles shared by GUI and captured scientific figures."""


def signal_unit_axis_label(unit):
    """For stored views that declare a unit but carry no dataset metadata."""
    return 'Normalized signal' if str(unit).strip() in ('1','dimensionless') else 'Signal ('+str(unit)+')'


def signal_axis_label(metadata, *, raw=False):
    kind = str(metadata.get('raw_signal_type' if raw else 'signal_type', 'signal')).strip().lower()
    if kind == 'normalized':
        return 'Signed-DCT Y' if not raw and metadata.get('normalization_mode') == 'signed_terminal' else 'Normalized signal'
    unit = str(metadata.get('raw_signal_unit' if raw else 'signal_unit', 'arbitrary')).strip() or 'arbitrary'
    return ('Raw signal' if raw else 'Signal') + ' (' + unit + ')'
