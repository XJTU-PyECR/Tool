"""Deterministic physical starts for staged dual relaxation fitting."""
from __future__ import annotations

import numpy as np

VERSION = "dual-informed-starts/2"


def start_schedule(parameterization, generator, count, model, warm_parameters=None):
    """Keep the PCG stream allocation fixed; replace selected random starts."""
    random = [parameterization.random_start(generator) for _ in range(count)]
    config = parameterization.config
    if config.start_policy == "legacy":
        return random, ["random_log_box"]*count
    chosen, sources = [], []

    def add(values, source):
        q = parameterization.encode(values)
        if not any(np.array_equal(q, previous) for previous in chosen):
            chosen.append(q)
            sources.append(source)

    if config.initial_values:
        add(config.initial_values, "user_initial")
    # Retain random exploration, even when deterministic seeds fill most of
    # the ensemble. Small explicitly requested ensembles still use M2 evidence.
    target = count-(max(1, count//5) if count >= 4 else 0)
    if parameterization.is_m3 and warm_parameters is not None:
        kernel = model.kernel
        sizes = getattr(kernel, "full_dims_cm", ())
        active = getattr(kernel, "active_dimensions", ())
        half = min((size/2 for size, use in zip(sizes, active) if use), default=None)
        pairs = [(1., .3), (10., 3.), (100., .3), (1., 3.), (10., .3), (100., 3.)]
        pairs += [(h, o) for h in (1., 10., 100.) for o in (.1, 1., 10.)]
        kb = tuple(10.**v for v in config.log10_k_bounds)
        for h, o in pairs:
            if len(chosen) >= target:
                break
            values = dict(warm_parameters)
            # For custom non-geometric kernels use the bounded log-k grid;
            # rectangular kernels use the physical half-thickness Bi scale.
            for species, bi in (("H", h), ("O", o)):
                value = (bi*values[f"D_{species}_cm2_s"]/half if half is not None
                         else 10.**(config.log10_k_bounds[0] +
                                    (np.log10(bi)+1)/3*np.ptp(config.log10_k_bounds)))
                values[f"k_{species}_cm_s"] = float(np.clip(value, *kb))
            add(values, f"m2_Bi_H={h:g}_O={o:g}")
    for q in random:
        if len(chosen) >= count:
            break
        chosen.append(q)
        sources.append("random_log_box")
    return chosen[:count], sources[:count]
