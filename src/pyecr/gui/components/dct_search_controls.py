"""Explicit draft edits for data-based tau ranges and boundary GCV searches."""
from pyecr.dct.advice import suggest_tau_range, gcv_boundary_side, extend_gcv_grid


class DctSearchControlsMixin:
    def _dct_search_signature(self):
        d = self._require_dpg()
        tags = ('pyecr_dct_analysis_mode','pyecr_dct_tau_min','pyecr_dct_tau_max','pyecr_dct_n_tau',
                'pyecr_dct_lambda_mode','pyecr_dct_lambda','pyecr_dct_lambdas','pyecr_signed_dct_lambdas')
        return (id(self.controller.analysis_session.dataset), *(str(d.get_value(t)) for t in tags))

    def _tau_advice(self):
        dataset = self.controller.analysis_session.dataset
        if dataset is None:
            raise ValueError('Import a dataset to suggest a range.')
        if dataset.metadata.get('time_unit', 's') != 's':
            raise ValueError('Convert processed time to seconds first.')
        cached = getattr(self, '_dct_tau_advice_cache', None)
        if cached is None or cached[0] is not dataset:
            cached = (dataset, suggest_tau_range(dataset.processed_time))
            self._dct_tau_advice_cache = cached
        return cached[1]

    def _apply_dct_tau_advice(self, *args):
        d = self._require_dpg()
        try:
            if self.controller.dct_activity:
                return
            lower, upper = self._tau_advice()
            d.set_value('pyecr_dct_tau_min', lower)
            d.set_value('pyecr_dct_tau_max', upper)
            self._dct_search_message = 'Suggested τ applied to draft. Run DCT to calculate.'
        except (ValueError, TypeError) as exc:
            self._dct_search_message = str(exc)
        self._dct_search_message_key = self._dct_search_signature()
        self._refresh_widgets()

    def _gcv_evidence(self):
        """Only offer expansion for the same current draft and captured source."""
        d = self._require_dpg()
        if str(d.get_value('pyecr_dct_lambda_mode')).lower() != 'gcv':
            return None
        if self._dct_signed_mode():
            component = self._dct_component_mode()
            job = self._selected_component_job() if component else self._selected_signed_job()
            if job is None or job.stale_reasons:
                return None
            cfg, spec, dataset, recipe = self._signed_dct_draft()
            request = job.request
            if (cfg != request.config or spec != request.normalization_spec
                    or dataset is not request.dataset or recipe != request.recipe):
                return None
            result = job.bundle.overall_result if component and job.bundle else (
                job.analysis.selected_result if not component and job.analysis else None)
            if result is None:
                return None
            return cfg.gcv_lambdas, result.selected_lambda
        presentation = self.controller.session.dct_result
        if presentation is None or self._dct_config_from_widgets() != presentation.config:
            return None
        result = presentation.result
        if result.gcv_lambdas is None:
            return None
        return tuple(result.gcv_lambdas), result.regularization_lambda

    def _expand_dct_gcv(self, *args):
        d = self._require_dpg()
        try:
            if self.controller.dct_activity:
                return
            evidence = self._gcv_evidence()
            if evidence is None:
                raise ValueError('Run GCV for the current draft before expanding.')
            expanded = extend_gcv_grid(*evidence)
            tag = 'pyecr_signed_dct_lambdas' if self._dct_signed_mode() else 'pyecr_dct_lambdas'
            d.set_value(tag, ','.join(format(x, '.17g') for x in expanded))
            self._dct_search_message = f'GCV draft: {expanded[0]:.3g}–{expanded[-1]:.3g}, {len(expanded)} candidates. Run DCT to calculate.'
        except (ValueError, TypeError) as exc:
            self._dct_search_message = str(exc)
        self._dct_search_message_key = self._dct_search_signature()
        self._refresh_widgets()

    def _refresh_dct_search_controls(self):
        d = self._require_dpg()
        if not d.does_item_exist('pyecr_dct_tau_advice'):
            return
        busy = bool(self.controller.dct_activity)
        try:
            lower, upper = self._tau_advice()
            narrow = (float(d.get_value('pyecr_dct_tau_min')) > lower * (1 + 1e-10)
                      or float(d.get_value('pyecr_dct_tau_max')) < upper * (1 - 1e-10))
            text = f'Data suggestion: {lower:.3g}–{upper:.3g} s'
            if narrow:
                text += '\nDraft is narrower; check for truncated peaks.'
            enabled = not busy
        except (ValueError, TypeError) as exc:
            text, enabled = str(exc), False
        d.set_value('pyecr_dct_tau_advice', text)
        d.configure_item('pyecr_dct_apply_tau_advice', enabled=enabled)
        manual = str(d.get_value('pyecr_dct_lambda_mode')).lower() == 'manual'
        d.configure_item('pyecr_dct_legacy_gcv', show=not self._dct_signed_mode())
        d.configure_item('pyecr_dct_lambdas', enabled=not manual)
        side, can_expand, boundary = None, False, ''
        try:
            evidence = self._gcv_evidence() if not manual else None
            if evidence is not None:
                side = gcv_boundary_side(*evidence)
                if side:
                    boundary = f'GCV λ at {side} search limit.'
                    try:
                        extend_gcv_grid(*evidence)
                        can_expand = not busy
                    except ValueError as exc:
                        boundary += ' ' + str(exc)
        except (ValueError, TypeError, AttributeError):
            pass  # Invalid drafts are reported by the normal Run gate.
        d.set_value('pyecr_dct_gcv_boundary', boundary)
        d.configure_item('pyecr_dct_gcv_boundary', show=bool(boundary))
        d.configure_item('pyecr_dct_expand_gcv', show=bool(side), enabled=can_expand)
        message = getattr(self, '_dct_search_message', '')
        if getattr(self, '_dct_search_message_key', None) != self._dct_search_signature():
            message = self._dct_search_message = ''
        d.set_value('pyecr_dct_search_message', message)
        d.configure_item('pyecr_dct_search_message', show=bool(message))
