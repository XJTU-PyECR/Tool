"""Export-only legend placement against visible artists in final pixel space."""
from dataclasses import replace

import numpy as np
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.transforms import Bbox

from .view_state import export_legend_position


def avoid_curves(axis, extent, state):
    """Prefer a clear inside position; reserve outside-right if all are occupied.

    Work in renderer coordinates so logarithmic axes, sparse line segments and
    scatter markers are treated as drawn. Never change a source array or limit.
    """
    renderer = axis.figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer)
    margin = axis.figure.dpi * 6 / 72
    paths, points, boxes = [], [], []
    if any(image.get_visible() for image in axis.images):
        boxes.append(bounds)  # Heatmap cells have no empty curve background.
    for line in axis.lines:
        if not line.get_visible():
            continue
        path = line.get_transform().transform_path(line.get_path())
        if line.get_linestyle() not in ('None', 'none', '', ' '):
            paths.append(path)
        if line.get_marker() not in ('None', 'none', '', ' '):
            points.append((path.vertices, axis.figure.dpi * line.get_markersize() / 144))
    for collection in axis.collections:
        if not collection.get_visible():
            continue
        if isinstance(collection, PathCollection):
            offsets = np.ma.asarray(collection.get_offsets()).filled(np.nan)
            sizes = collection.get_sizes()
            radius = axis.figure.dpi * np.sqrt(max(sizes, default=0)) / 144
            points.append((collection.get_offset_transform().transform(offsets), radius))
        elif isinstance(collection, LineCollection):
            paths.extend(collection.get_transform().transform_path(p) for p in collection.get_paths())
        elif isinstance(collection, QuadMesh):
            boxes.append(bounds)
    for patch in axis.patches:
        if patch.get_visible():
            boxes.append(patch.get_window_extent(renderer))
    for text in axis.texts:
        if text.get_visible() and text.get_text():
            boxes.append(text.get_window_extent(renderer))

    def score(box):
        padded = Bbox.intersection(box.padded(margin), bounds)
        value = sum(path.intersects_bbox(padded, filled=False) for path in paths)
        value += sum(box.overlaps(padded) for box in boxes)
        for values, radius in points:
            box = padded.padded(radius)
            value += int(np.count_nonzero(
                (values[:, 0] >= box.x0) & (values[:, 0] <= box.x1) &
                (values[:, 1] >= box.y0) & (values[:, 1] <= box.y1)))
        return int(value)

    scores = {}
    for position in ('right', 'top_right', 'bottom_right', 'top_left', 'bottom_left',
                     'top', 'bottom', 'left', 'center'):
        candidate = replace(state, legend_mode='preset', legend_position=position)
        x, top = export_legend_position((bounds.x0, 0., bounds.width, bounds.height),
                                       (extent.width, extent.height), candidate, axis.figure.dpi)
        box = Bbox.from_bounds(x, bounds.y1 - top - extent.height, extent.width, extent.height)
        scores[position] = score(box)
        if not scores[position]:
            return candidate, dict(policy='auto', mode='preset', position=position,
                                   overlap_score=0, candidate_scores=scores)
    return replace(state, legend_mode='outside', outside_side='right'), dict(
        policy='auto', mode='outside', side='right', overlap_score=0,
        candidate_scores=scores, reason='All inside positions intersect visible curves or annotations.')
