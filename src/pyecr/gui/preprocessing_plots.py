"""Coordinate and row identity rules for read-only preprocessing plots."""
import numpy as np


def window_rows(dataset, window):
    """Map recorded original rows into the current processed observation view."""
    return np.isin(np.flatnonzero(dataset.mask), tuple(window.get('raw_indices', ())))


def preview_uses_separate_axes(current, proposed):
    if any(current.metadata.get(k) != proposed.metadata.get(k)
           for k in ('time_unit', 'signal_unit', 'signal_type')):
        return True
    if np.array_equal(current.mask,proposed.mask):
        return not np.array_equal(current.processed_time,proposed.processed_time)
    _, left, right = np.intersect1d(np.flatnonzero(current.mask), np.flatnonzero(proposed.mask), return_indices=True)
    # Same time unit does not mean same origin (manual/suggested t0).
    return not np.array_equal(current.processed_time[left], proposed.processed_time[right])


def window_is_current(dataset, kind, window):
    try:
        dataset._validate_stability_window_is_current(kind, window, allow_normalized_signal=True,
                                                      require_unchanged_signal=True)
    except (ValueError, TypeError, KeyError):
        return False
    return True


def plateau_reference_summary(dataset):
    """Separate calibration evidence from rows retained in the fit view."""
    if dataset is None:
        return ''
    try:
        reference = dataset.normalization_reference()
        windows = dataset.metadata.get('stability_windows', {})
        lines = []
        for kind in ('initial', 'final'):
            window = windows.get(kind)
            if not window:
                continue
            start, end = dataset.stability_window_bounds(kind, window)
            retained = int(window_rows(dataset, window).sum())
            status = 'fixed normalization reference' if reference else (
                'accepted' if window_is_current(dataset, kind, window) else 'stale; reselect before normalization')
            lines.append(f"{kind.title()}: [{start:.6g}, {end:.6g}] {dataset.metadata['time_unit']}; "
                         f"{retained}/{window['n_points']} rows retained; {status}.")
        if reference:
            lines.append('Plateau references remain valid outside the fit interval. No need to reselect or re-normalize after t0 / crop.')
        return '\n'.join(lines)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return 'Reference needs review: ' + str(exc)
