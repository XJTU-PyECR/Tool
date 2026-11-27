"""Export stored simulation samples, never regenerate or densify scientific data."""
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def write_simulation_figures(result,root,formats,options,check,*,emit=None):
    if root is not None:root.mkdir(parents=True,exist_ok=True)
    r=result.request
    for name in ('simulation_truth','simulation_observations'):
        fig=Figure(figsize=(options.width_in,options.height_in*1.6));FigureCanvasAgg(fig)
        total,lower=fig.subplots(2,1)
        if name=='simulation_truth':
            total.plot(r.time_s,result.signal_clean,label='simulation_clean: given-parameter truth')
            baseline=r.given_parameters.get('baseline',r.given_parameters.get('y0'))
            total.axhline(baseline,color='gray',linestyle=':',label='Common given baseline')
            for key,values in result.contributions.items():lower.plot(r.time_s,values,label='Clean model '+key)
            lower.axhline(0.,color='gray',linewidth=.7)
            lower.set_ylabel('Signed contribution ('+r.signal_unit+')')
        else:
            total.plot(r.time_s,result.signal_clean,label='simulation_clean: given truth')
            total.plot(r.time_s,result.signal_observed,'.',label='simulation_observed: simulated observations')
            lower.plot(r.time_s,result.noise,'.-',label='Actual total observation noise')
            lower.axhline(0.,color='gray',linewidth=.7);lower.set_ylabel('Noise ('+r.signal_unit+')')
        from pyecr.plotting.labels import signal_unit_axis_label
        total.set_ylabel(signal_unit_axis_label(r.signal_unit));total.set_title('Simulation; no Fit / CI; full scientific sampling')
        for ax in (total,lower):ax.set_xlabel('Time (s)');ax.legend(fontsize=8)
        if emit is not None:
            try:emit(name,fig)
            finally:fig.clear()
            continue
        from matplotlib.text import Text
        for text in fig.findobj(Text):text.set_fontfamily(options.font_family)
        fig.tight_layout()
        from pyecr.plotting.view_state import apply_axis_view
        apply_axis_view(total,'pyecr_sim_total_plot',options.plot_views,
                        ('pyecr_sim_total_series','pyecr_sim_observed_series') if name=='simulation_observations' else ('pyecr_sim_total_series',),
                        legend_policy=options.legend_policy)
        apply_axis_view(lower,'pyecr_sim_components_plot' if name=='simulation_truth' else 'pyecr_sim_noise_plot', options.plot_views,
                        tuple('pyecr_sim_component_'+str(i) for i in range(1,len(result.contributions)+1)) if name=='simulation_truth' else ('pyecr_sim_noise_series',),
                        legend_policy=options.legend_policy)
        try:
            for fmt in formats:check();fig.savefig(root/(name+'.'+fmt),format=fmt,dpi=options.image_dpi)
        finally:fig.clear()
