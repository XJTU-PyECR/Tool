"""Short labels for fixed application series; user/source names are preserved."""

_LABELS = {
    'Experimental Y': 'Data',
    'Experiment Y': 'Data',
    'Experimental observations': 'Data',
    'Model-derived component shape': 'Component shape',
    'δ + weighted component DCT': 'Components + δ',
    'Parent Y_fit': 'Parent Fit',
    'Physical parent Fit Y': 'Parent Fit',
    'Explicit δ': 'δ offset',
    'Explicit delta': 'δ offset',
    'Whole signed reconstruction': 'Whole DCT',
    'Whole DCT reconstruction': 'Whole DCT',
    'DCT reconstruction': 'DCT',
    'component_1 · B1/S': 'C1 · B1/S',
    'component_2 · B2/S': 'C2 · B2/S',
    'Σ weighted χ_j': 'Σ components',
    'Y_fit − component reconstruction': 'Fit − components',
    'Candidate time-scale distribution': 'Spectrum',
    'χ > 0: above terminal': 'χ > 0',
    'χ < 0: below terminal': 'χ < 0',
    'Positive signed density (1/decade)': 'χ > 0',
    'Negative signed density (1/decade)': 'χ < 0',
    'Positive whole signed density': 'Whole χ > 0',
    'Negative whole signed density': 'Whole χ < 0',
    'Simulated observations Y': 'Simulated data',
    'Simulated clean Y': 'Simulated clean',
    'component_1 independent shape': 'C1 shape',
    'component_2 independent shape': 'C2 shape',
    'component_1: (B/S)*chi': 'C1 · B1/S',
    'component_2: (B/S)*chi': 'C2 · B2/S',
    'delta + sum((B/S)*shape DCT)': 'Components + δ',
    'Parent Fit - component DCT': 'Fit − components',
    'Experiment - component DCT': 'Data − components',
    'Whole signed experiment': 'Whole DCT',
}


def compact_legend_label(label):
    return _LABELS.get(label, label)
