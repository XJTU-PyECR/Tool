"""Column-oriented figure arrays; never resample, sort, join or infer values."""
from dataclasses import dataclass
import numpy as np
from pyecr.report.export_tables import ExportTable

TABLE_LAYOUT_VERSION = 'series-columns/1'
LEGACY_TABLE_LAYOUT_VERSION = 'series-rows/1'


@dataclass(frozen=True)
class SeriesExportTable(ExportTable):
    coordinate_layout: str = 'shared_x'
    series_columns: tuple = ()

    def info(self):
        return {**super().info(), 'layout': TABLE_LAYOUT_VERSION,
                'coordinate_layout': self.coordinate_layout,
                'series_columns': [dict(item) for item in self.series_columns]}


def build_series_table(figure, name):
    """Use one shared coordinate only for exactly identical captured arrays.

    Different grids are independent X/Y column pairs, padded with empty cells
    after each array's own last sample. Horizontal references have no scientific
    X array: store their single captured constant once, not an invented curve.
    Original labels/IDs/kinds/sampling and point counts accompany the columns.
    """
    curves = [s for s in figure.series if s.kind != 'hline']
    shared = bool(curves) and all(np.array_equal(curves[0].x, s.x, equal_nan=True)
                                  for s in curves[1:])
    # Preserve signed zero in a shared axis as well as the numerical coordinate.
    if shared:
        shared = all(np.array_equal(np.signbit(curves[0].x), np.signbit(s.x))
                     for s in curves[1:])
    axis = ('time_' + figure.x_unit if 'time' in figure.xlabel.casefold()
            else 'X: ' + figure.xlabel if figure.xlabel else 'X')
    columns, arrays, units, bindings = [], [], {}, []
    used = set()

    def add(header, values, unit):
        # Prefixes below also prevent user labels from becoming Excel formulas.
        # Full unmodified labels remain in series_columns and catalog.json.
        clean = ''.join(f'\\u{ord(c):04x}' if ord(c)<32 and not c.isspace() else c
                        for c in header)
        base = ' '.join(clean.split())[:180]
        candidate, suffix = base, 2
        while candidate.casefold() in used:
            candidate = f'{base} ({suffix})'; suffix += 1
        used.add(candidate.casefold()); columns.append(candidate)
        arrays.append(values); units[candidate] = unit
        return candidate

    shared_column = add(axis, curves[0].x, figure.x_unit) if shared else None
    for index, series in enumerate(figure.series):
        label = series.label or series.series_id or f'Series {index+1}'
        reference = series.kind == 'hline'
        x_column = None if reference else shared_column if shared else add(
            axis + ': ' + label, series.x, figure.x_unit)
        y_column = add(('Reference: ' if reference else 'Value: ') + label,
                       series.y, figure.y_unit)
        bindings.append(dict(series_index=index, label=series.label,
            series_id=series.series_id, sampling=series.sampling, kind=series.kind,
            x_column=x_column, y_column=y_column, points=len(series.y)))
    count = max(map(len, arrays), default=0)

    def rows():
        for i in range(count):
            yield tuple(values[i] if i < len(values) else None for values in arrays)

    return SeriesExportTable(name, tuple(columns), ('float64?',)*len(columns),
        units, count, rows, coordinate_layout='shared_x' if shared else 'per_series_x',
        series_columns=tuple(bindings))
