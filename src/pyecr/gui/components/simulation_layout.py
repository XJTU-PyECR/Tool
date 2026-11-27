"""Simulation four-step display; requests and services retain their original contracts."""
from pyecr.gui.theme import TOKENS
from pyecr.gui.components.step_panel import step_panel
from pyecr.gui.components.widgets import table_style_kwargs
from pyecr.gui.simulation import MATRIX
from pyecr.units import get_physical_units, PRESSURE_UNIT_CHOICES, get_units

P='pyecr_sim_'
MODEL_LABELS=dict(zip(MATRIX,('01 Diffusion','02 Surface exchange','03 Diffusion & exchange',
    '04 Two diffusion processes','05 Two exchange processes','06 Two coupled processes')))
BOUNDARY_LABELS={'ideal_step':'Ideal step','exponential_flush':'Exponential flush','measured_history':'Measured history'}
SAMPLING_LABELS={'linear':'Linear','log_with_zero':'Log + zero','explicit':'Explicit times'}
PARAMETER_LABELS={'D_cm2_s':'D','k_cm_s':'k','D_H_cm2_s':'D (H)','D_O_cm2_s':'D (O)',
    'k_H_cm_s':'k (H)','k_O_cm_s':'k (O)','k_1_cm_s':'k (1)','k_2_cm_s':'k (2)',
    'A_H':'Amplitude\n(H)','A_O':'Amplitude\n(O)','A_1':'Amplitude\n(1)','A_2':'Amplitude\n(2)',
    'amplitude':'Amplitude','baseline':'Initial signal','y0':'Initial signal\n(y0)'}


class SimulationLayoutMixin:
    def _sim_text_input(self,key,label,**kwargs):
        d=self._require_dpg();d.add_text(label,tag=P+key+'_label')
        d.add_input_text(tag=P+key,width=-1,callback=self._simulation_edited,**kwargs)

    def _sim_unit_input(self,key,label,dimension):
        d=self._require_dpg();d.add_text(label)
        d.add_combo(tuple(u.label for u in get_physical_units(dimension)),tag=P+key,width=-1,callback=self._simulation_edited)

    def _build_simulation_layout(self):
        d=self._require_dpg();self._sim_render_key=object();self._sim_review_revision=None;self._sim_draft_request=None
        self._sim_noise_choice='None';self._sim_gaussian_sd='0'
        self._sim_fitted_axes={};self._sim_pending_axes={};self._sim_release_axes={}
        with d.group(tag=P+'root'):
            d.add_text('Simulated data · given parameters',tag=P+'title',wrap=TOKENS.wraps.simulation_form)
            with d.group(horizontal=True,tag=P+'columns'):
                with d.child_window(tag=P+'left',width=TOKENS.simulation.left_compact,height=TOKENS.heights.simulation_panel,border=False,no_scrollbar=True):
                    with d.child_window(tag=P+'inputs',width=-1,height=TOKENS.heights.simulation_panel,border=False):
                        with step_panel(d,'sim_model','01 Model & parameters',default_open=True):
                            with d.group(parent='pyecr_stage_sim_model_content'):
                                d.add_text('Model')
                                d.add_combo(tuple(MODEL_LABELS.values()),tag=P+'model',width=-1,callback=self._simulation_model_changed)
                                with d.collapsing_header(label='Model Details / copy Analysis',default_open=False):
                                    d.add_text('',tag=P+'model_notice',wrap=TOKENS.wraps.simulation_notice)
                                    d.add_button(label='Copy Analysis config to draft',tag=P+'copy_analysis',callback=self._copy_analysis_simulation)
                                with d.group(tag=P+'parameters'):pass
                                d.add_text('Full dimensions / active axes')
                                self._sim_unit_input('length_unit','Length unit','length')
                                with d.table(width=-1,policy=d.mvTable_SizingStretchProp,**table_style_kwargs()):
                                    d.add_table_column(label='Axis',width_fixed=True,init_width_or_weight=TOKENS.simulation.axis_width)
                                    d.add_table_column(label='Full L',tag=P+'dimensions_column')
                                    d.add_table_column(label='Active',width_fixed=True,init_width_or_weight=TOKENS.simulation.active_width)
                                    for axis in 'xyz':
                                        with d.table_row():
                                            d.add_text(axis.upper());d.add_input_text(tag=P+'dim_'+axis,width=-1,callback=self._simulation_edited)
                                            d.add_checkbox(tag=P+'active_'+axis,callback=self._simulation_edited)
                                d.add_text('Both opposite faces of each active axis.',wrap=TOKENS.wraps.simulation_notice)
                                d.add_text('Signal type / unit')
                                from pyecr.units import SIGNAL_TYPES
                                d.add_combo(SIGNAL_TYPES,tag=P+'signal_type',width=-1,callback=self._simulation_unit_changed)
                                d.add_combo(tuple(u.label for u in get_units('normalized')),tag=P+'signal_unit',width=-1,callback=self._simulation_signal_unit_selected)
                                d.add_input_text(tag=P+'signal_custom_unit',hint='Custom signal unit',width=-1,show=False,callback=self._simulation_edited)
                                d.add_text('',tag=P+'unit_notice',show=False,wrap=TOKENS.wraps.simulation_notice)
                                with d.collapsing_header(label='Signal unit Details',default_open=False):
                                    d.add_text('Signal units declare values; amplitude, baseline and SD are unchanged. Physical length, time, D and k inputs convert once when Generate freezes the request.',wrap=TOKENS.wraps.simulation_notice)
                                d.add_text('Boundary')
                                d.add_combo((),tag=P+'boundary',width=-1,callback=self._simulation_boundary_changed)
                                with d.collapsing_header(label='Boundary Details',default_open=False):
                                    d.add_text('',tag=P+'boundary_notice',wrap=TOKENS.wraps.simulation_notice)
                                d.add_text('Step direction')
                                d.add_combo((),tag=P+'step_type',width=-1,callback=self._simulation_boundary_changed)
                                with d.group(tag=P+'custom_signs'):
                                    self._sim_text_input('sign_H','Process H / input 1 sign')
                                    self._sim_text_input('sign_O','Process O / input 2 sign')
                                with d.group(tag=P+'flush'):
                                    self._sim_unit_input('flush_time_unit','Flush time unit','time')
                                    self._sim_text_input('flush_time_s','Flush residence time (s)')
                                with d.group(tag=P+'measured'):
                                    self._sim_unit_input('history_time_unit','Gas history time unit','time')
                                    self._sim_text_input('history_time','Measured gas times (s)',multiline=True,height=TOKENS.simulation.history_input_height)
                                    self._sim_text_input('history_value','Measured gas values',multiline=True,height=TOKENS.simulation.history_input_height)
                                    d.add_combo(('pO2','pH2O'),tag=P+'gas_type',width=-1,callback=self._simulation_edited)
                                    for k,label in (('initial_gas','Initial gas value'),('final_gas','Final gas value')):self._sim_text_input(k,label)
                                    d.add_text('Gas unit (values and endpoints)')
                                    d.add_combo(PRESSURE_UNIT_CHOICES,tag=P+'gas_unit',width=-1,callback=self._simulation_edited)
                                    with d.collapsing_header(label='Advanced convolution settings',default_open=False):
                                        self._sim_unit_input('internal_step_unit','Internal step unit','time')
                                        self._sim_text_input('internal_step','Time step (s), blank = Auto')
                                        self._sim_text_input('internal_budget','Internal point budget')
                                    d.add_text('Gas history must cover all requested times.',wrap=TOKENS.wraps.simulation_notice)
                        with step_panel(d,'sim_sampling','02 Sampling & noise',default_open=True):
                            with d.group(parent='pyecr_stage_sim_sampling_content'):
                                d.add_text('Time sampling')
                                self._sim_unit_input('time_unit','Sampling time unit','time')
                                d.add_radio_button(tuple(SAMPLING_LABELS.values()),tag=P+'sampling_choice',callback=self._simulation_sampling_changed)
                                d.add_combo(tuple(SAMPLING_LABELS),tag=P+'sampling',show=False)
                                with d.group(tag=P+'sample_regular'):
                                    for k,label in (('start','Start (s)'),('first_positive','First positive time (s)'),('stop','Stop (s)'),('count','Point count')):
                                        with d.group(tag=P+'sample_'+k):self._sim_text_input(k,label)
                                with d.group(tag=P+'sample_explicit'):
                                    self._sim_text_input('explicit_times','Times (s)',multiline=True,height=TOKENS.heights.simulation_noise_review)
                                with d.collapsing_header(label='Sampling Details',tag=P+'sampling_details',default_open=False):
                                    d.add_text('Log + zero uses [0] + geomspace(first positive, stop, count-1). Explicit times must increase strictly; all observations are retained.',wrap=TOKENS.wraps.simulation_notice)
                                d.add_text('Observation noise')
                                d.add_radio_button(('None','Gaussian'),tag=P+'noise_model',default_value='None',horizontal=True,callback=self._simulation_noise_changed)
                                with d.group(tag=P+'noise_fields'):
                                    self._sim_text_input('noise_sd','Gaussian SD')
                                    d.add_text('',tag=P+'noise_notice',wrap=TOKENS.wraps.simulation_notice)
                                with d.collapsing_header(label='Advanced noise settings',tag=P+'noise_advanced',default_open=False):
                                    self._sim_text_input('seed','Random seed')
                                    self._sim_text_input('sequence_number','Sequence number')
                                with d.collapsing_header(label='Noise Details',tag=P+'noise_details',default_open=False):
                                    d.add_text('SD uses the signal unit. Only the total receives noise; contributions stay clean. SD=0 executes no RNG; seed is ignored and not recorded as used.',wrap=TOKENS.wraps.simulation_notice)
                                    d.add_text('Generator(PCG64(SeedSequence(seed, spawn_key=(sequence,)))); sequence is uint32. The same seed, sequence and request reproduce noise.',wrap=TOKENS.wraps.simulation_notice)
                                with d.collapsing_header(label='Advanced numerical settings',default_open=False):
                                    for k,label in (('relative_tolerance','Series term tolerance'),('absolute_tolerance','Recorded absolute tolerance'),('max_terms','Maximum series terms')):self._sim_text_input(k,label)
                    with step_panel(d,'sim_generate','03 Generate',default_open=True):
                        with d.group(parent='pyecr_stage_sim_generate_content',tag=P+'run_toolbar'):
                            self._sim_text_input('name','Result name')
                            with d.group(horizontal=True,tag=P+'actions'):
                                d.add_button(label='Generate',tag=P+'generate',callback=self._generate_simulation)
                                d.add_button(label='Cancel',tag=P+'cancel',callback=self._cancel_simulation,enabled=False)
                                d.add_button(label='Details',tag=P+'details',callback=self._show_simulation_details)
                            d.add_text('',tag=P+'status',wrap=TOKENS.wraps.simulation_form)
                            d.add_text('',tag=P+'reason',wrap=TOKENS.wraps.simulation_form,color=TOKENS.colors.warning)
                with d.child_window(tag=P+'results',width=TOKENS.widths.simulation_results,height=TOKENS.heights.simulation_panel,border=False):
                    with step_panel(d,'sim_results','04 Results & export',default_open=True):
                        with d.group(parent='pyecr_stage_sim_results_content'):
                            d.add_combo((),tag=P+'history',width=-1,show=False,callback=self._simulation_history_selected)
                            d.add_text('',tag=P+'stored_summary',wrap=TOKENS.wraps.simulation_help,show=False)
                            d.add_text('',tag=P+'result_warning',wrap=TOKENS.wraps.simulation_help,show=False,color=TOKENS.colors.warning)
                            with d.group(horizontal=True):
                                d.add_button(label='Export / Send',tag=P+'exchange',callback=self._show_simulation_exchange)
                                d.add_button(label='Open saved…',tag=P+'open_history',callback=self._show_simulation_history)
                                d.add_checkbox(label='More plots',tag=P+'more_plots',callback=self._toggle_simulation_plots)
                            with d.tab_bar(tag=P+'result_tabs'):
                                with d.tab(label='Clean & observed',tag=P+'curves_tab'):self._build_simulation_plot('total','Simulated observations and clean signal')
                                with d.tab(label='Contributions & noise',tag=P+'more_tab',show=False):
                                    self._build_simulation_plot('components','Clean signed contributions')
                                    self._build_simulation_plot('noise','Actual observation noise')
                                with d.tab(label='Details',tag=P+'source_tab'):
                                    d.add_input_text(tag=P+'result_parameters',readonly=True,multiline=True,width=-1,height=TOKENS.heights.simulation_results)
        with d.window(label='Simulation run details',tag=P+'details_window',show=False,width=TOKENS.widths.simulation_details,height=TOKENS.heights.simulation_details):
            with d.group(horizontal=True):
                d.add_button(label='Close',callback=lambda:d.configure_item(P+'details_window',show=False))
                d.add_button(label='Copy run record',tag=P+'copy_details',callback=self._copy_simulation_details)
            d.add_input_text(tag=P+'details_text',readonly=True,multiline=True,width=-1,height=-1)
        self._register_owned_root(P+'root');self._register_owned_root(P+'details_window')
        for step in ('sim_model','sim_sampling','sim_generate','sim_results'):
            for tag in (f'pyecr_step_{step}_status',f'pyecr_step_{step}_warning',f'pyecr_stage_{step}_reason'):d.hide_item(tag)
        self._build_simulation_exchange();self._load_simulation_widgets()

    def _build_simulation_plot(self,kind,title):
        d=self._require_dpg()
        with d.plot(label=title,tag=P+kind+'_plot',width=-1,height=TOKENS.simulation.plot_min):
            d.add_plot_legend();d.add_plot_axis(d.mvXAxis,label='Time (s)',tag=P+kind+'_x')
            with d.plot_axis(d.mvYAxis,label='Signal',tag=P+kind+'_y'):
                if kind=='total':
                    d.add_line_series([],[],label='Clean signal',tag=P+'total_series',show=False)
                    d.add_scatter_series([],[],label='Simulated observations',tag=P+'observed_series',show=False)
                elif kind=='noise':d.add_line_series([],[],label='Observation noise',tag=P+'noise_series',show=False)
                else:
                    for slot in ('1','2'):d.add_line_series([],[],label='Component '+slot,tag=P+'component_'+slot,show=False)
                    d.add_line_series([],[],label='Zero contribution',tag=P+'zero',show=False)

    def _show_simulation_history(self,*args):
        self._show_simulation_exchange();self._require_dpg().set_value('pyecr_sim_io_tabs','pyecr_sim_io_open_tab')

    def _toggle_simulation_plots(self,*args):
        d=self._require_dpg();more=bool(d.get_value(P+'more_plots'));d.configure_item(P+'more_tab',show=more)
        selected=d.get_value(P+'result_tabs')
        if isinstance(selected,int):selected=d.get_item_alias(selected)
        if not more and selected==P+'more_tab':d.set_value(P+'result_tabs',P+'curves_tab')

    def _fit_simulation_axes(self,result):
        d=self._require_dpg();frame=d.get_frame_count()
        for kind in ('total','components','noise'):
            if result is not None and result.state.ready and d.get_item_state(P+kind+'_plot').get('visible') and self._sim_fitted_axes.get(kind) is not result:
                pending=self._sim_pending_axes.get(kind)
                if pending is not None and pending[0] is result and frame>pending[1]:
                    x=result.time_s
                    y=(*result.signal_clean,*result.signal_observed) if kind=='total' else (*result.noise,0.) if kind=='noise' else (0.,*(v for a in result.contributions.values() for v in a))
                    lo,hi=min(y),max(y);pad=(hi-lo)*TOKENS.plot.fit_padding[1] or max(abs(lo),1.)*TOKENS.plot.fit_padding[1]
                    d.set_axis_limits(P+kind+'_x',min(x),max(x));d.set_axis_limits(P+kind+'_y',lo-pad,hi+pad)
                    self._sim_fitted_axes[kind]=result;self._sim_release_axes[kind]=frame+2
                else:self._sim_pending_axes[kind]=(result,frame)
            if kind in self._sim_release_axes and frame>=self._sim_release_axes[kind]:
                for axis in ('x','y'):d.set_axis_limits_auto(P+kind+'_'+axis)
                del self._sim_release_axes[kind]
