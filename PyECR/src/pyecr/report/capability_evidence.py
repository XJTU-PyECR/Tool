"""Export metadata from frozen component eligibility (no scientific work)."""
def component_records(archive):
    return [dict(job_id=j.job_id,eligibility=j.request.capture.eligibility.to_dict())
            for j in archive.component_jobs if j.request.capture.eligibility is not None]


def source_records(source):
    if source.result_type!='DctArchive':return []
    from pyecr.report.figure_catalog import thaw
    result=thaw(source.payload)['result']
    return [dict(job_id=j['job_id'],eligibility=j['request']['capture']['eligibility'])
            for j in result['component_jobs'] if j['request']['capture'].get('eligibility') is not None]


def caption(e):
    if e['original_reliable']:return 'Model-derived components; original parent gate passed. No independent-noise CI.'
    return 'Residual correlation warning; original parent unreliable. Component DCT permitted; correlated-error CI unsupported.'
