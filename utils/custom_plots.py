import numpy
import scipy
import scipy.stats
import matplotlib
from matplotlib import pyplot
import pandas

import logomaker

import seq_utils


##############
# Constants
##############

nt_color_dict = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'U': (228/255, 38/255, 56/255),
}

greysBig = matplotlib.cm.get_cmap('Greys', 512)
greys_trunc_cm = matplotlib.colors.ListedColormap(greysBig(numpy.linspace(0.6, 1, 256)))

tpm_fraction_list = ['total', '80S', 'LMW', 'HMW']
pol_fraction_list = ['80S', 'LMW', 'HMW']
timepoint_list = [2, 4, 6, 10]

tpm_cols = ['TPM_library'] + [f'TPM_{f}_{t}hpf' for f in tpm_fraction_list for t in timepoint_list]

lib_tpm_col = 'TPM_library'
log2_lib_tpm_col = 'log2_TPM_library'

total_tpm_cols = [f'TPM_total_{t}hpf' for t in timepoint_list]
total_log2_tpm_cols = [f'log2_TPM_total_{t}hpf' for t in timepoint_list]

log2_mrl_cols = [f'log2_MRL_{t}hpf' for t in timepoint_list]
delta_log2_x_cols = [f'Δlog2_X_{t}hpf' for t in timepoint_list]

res_log2_mrl_cols = [f'res_log2_MRL_{t}hpf' for t in timepoint_list]
res_delta_log2_x_cols = [f'res_Δlog2_X_{t}hpf' for t in timepoint_list]

log2_x_cols = [f'log2_X_{t}hpf' for t in timepoint_list]

# Code to make an extended set of model outputs, containing the
# differences in MRL and FC
def make_extended_shap(val):
    val_extended = numpy.array([
            val[0],
            val[1],
            val[2],
            val[3],
            # val[1] - val[0],
            # val[2] - val[0],
            # val[2] - val[1],
            val[3] - val[0],
            # val[3] - val[1],
            # val[3] - val[2],
            val[4],
            val[5],
            val[6],
            val[7],
        ])
    return val_extended

model_outputs_ext_labels = [
    '$log_2MRL^{2hpf}$',
    '$log_2MRL^{4hpf}$',
    '$log_2MRL^{6hpf}$',
    '$log_2MRL^{10hpf}$',

    # 'Δlog2_MRL_4-2hpf',
    # 'Δlog2_MRL_6-2hpf',
    # 'Δlog2_MRL_6-4hpf',
    '$Δlog_2MRL^{10-2hpf}$',
    # 'Δlog2_MRL_10-4hpf',
    # 'Δlog2_MRL_10-6hpf',
    
    '$Δlog_2X^{2-0hpf}$',
    '$Δlog_2X^{4-2hpf}$',
    '$Δlog_2X^{6-4hpf}$',
    '$Δlog_2X^{10-6hpf}$',
]

model_output_to_ext = {k: v for k, v in zip(log2_mrl_cols, model_outputs_ext_labels[:4])} | \
    {k: v for k, v in zip(delta_log2_x_cols, model_outputs_ext_labels[-4:])}

def plot_seq_logo(nt_height=None, seq_val=None, ax=None, title=None, spines=True):
    """
    Plot a sequence logo
    
    TODO: add details
    """

    # If nt_height not provided directly, calculate from seq_val
    if nt_height is None:
        if type(seq_val)==numpy.ndarray:
            # Assume pwm
            pwm = seq_val
            entropy = numpy.zeros_like(pwm)
            entropy[pwm > 0] = pwm[pwm > 0] * -numpy.log2(pwm[pwm > 0])
            entropy = numpy.sum(entropy, axis=1)
            conservation = 2 - entropy
            # Nucleotide height
            nt_height = numpy.tile(numpy.reshape(conservation, (-1, 1)), (1, 4))
            nt_height = pwm * nt_height
        elif type(seq_val)==str:
            # Assume string
            nt_to_onehot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
            nt_height = [nt_to_onehot[c] for c in seq_val]
            nt_height = numpy.array(nt_height)
        else:
            raise ValueError(f"type of seq_val {type(seq_val)} not recognized")

    nt_height_df = pandas.DataFrame(
        nt_height,
        columns=['A', 'C', 'G', 'U'],
    )
    
    logo = logomaker.Logo(
        nt_height_df,
        # color_scheme='classic',
        color_scheme=nt_color_dict,
        ax=ax,
        font_name='Consolas',
    )
    logo.style_spines(visible=False)
    if spines:
        # logo.style_spines(spines=['bottom'], visible=True, linewidth=1)
        logo.style_spines(spines=['left'], visible=True, linewidth=1)
    ax.set_xticks([])
    # ax.set_yticks([])
    if title is not None:
        ax.set_title(title)



def plot_scatter_shaded(
        x, y,
        ax,
        xlim=None,
        ylim=None,
        linreg=False,
        cmap=greys_trunc_cm,
        n_sample_cmap=1000,
        label=None,
        rasterized=True,
        s=0.5,
    ):
    xy = numpy.vstack([x, y])
    if len(x) > n_sample_cmap:
        idx = numpy.random.choice(len(x), n_sample_cmap, replace=False)
        xy_sample = xy[:, idx]
    else:
        xy_sample = xy
    z = scipy.stats.gaussian_kde(xy_sample)(xy)
    ax.scatter(
        x,
        y,
        s=s,
        c=z,
        cmap=cmap,
        rasterized=rasterized,
        label=label,
    )

    if linreg:
        lrres = scipy.stats.linregress(
            x,
            y,
        )
        ax.axline(
            (numpy.mean(x), lrres.intercept + lrres.slope*numpy.mean(x)),
            slope=lrres.slope,
            color='dodgerblue',
            linewidth=2,
        )
        if xlim is None:
            xlim = ax.get_xlim()
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ylim = ax.get_ylim()
        else:
            ax.set_ylim(ylim)
        ax.annotate(
            f'$r^2$ = {lrres.rvalue**2:.3f}',
            xy=(xlim[0], ylim[1]),
            xytext=(4,-4), textcoords='offset points',
            va='top',
        )

        return lrres
    
# Plot prediction performance scatter plots
def plot_pred_vs_obs(
        data,
        obs_col_prefix,
        pred_col_prefix,
        axis_limits,
        obs_label_prefix=None,
        pred_label_prefix=None,
        fig=None,
        plot_mrl=True,
        plot_delta_mrl=True,
        plot_delta_x=True,
        plot_x=True,
        return_r2=False,
    ):
    
    nrows = plot_mrl + plot_delta_mrl + plot_delta_x + plot_x

    if fig is None:
        fig, axes = pyplot.subplots(
            nrows,
            len(timepoint_list),
            figsize=(3*len(timepoint_list), nrows*3.5),
        )
        fig.subplots_adjust(hspace=0.5)

    if (obs_col_prefix is not None) and (not (obs_col_prefix=='')) and (not obs_col_prefix.endswith('_')):
        obs_col_prefix += '_'
    if (pred_col_prefix is not None) and (not (pred_col_prefix=='')) and (not pred_col_prefix.endswith('_')):
        pred_col_prefix += '_'

    if obs_label_prefix is None:
        obs_label_prefix = 'Observed'
    if pred_label_prefix is None:
        pred_label_prefix = 'Predicted'

    row_idx = 0

    r2_dict = {}

    # log2MRL
    if plot_mrl:
        for y_idx, ycol in enumerate(log2_mrl_cols):
            ax = axes[row_idx, y_idx]

            lrres = plot_scatter_shaded(
                data[obs_col_prefix + ycol],
                data[pred_col_prefix + ycol],
                ax,
                xlim=(axis_limits[row_idx]),
                ylim=(axis_limits[row_idx]),
                linreg=True,
            )

            ax.set_xlabel(f'{obs_label_prefix} log2_MRL')
            if y_idx==0:
                ax.set_ylabel(f'{pred_label_prefix} log2_MRL')

            ax.text(0.96, 0.96, f't = {timepoint_list[y_idx]}hpf', transform=ax.transAxes, ha='right', va='top')

            r2_dict[f"log2_MRL_{timepoint_list[y_idx]}hpf"] = lrres.rvalue**2
        
        row_idx += 1

    # log2MRL differences
    if plot_delta_mrl:
        axes[row_idx, 0].set_visible(False)
        for y_idx, ycol in enumerate(log2_mrl_cols[1:]):
            ax = axes[1, y_idx+1]

            lrres = plot_scatter_shaded(
                data[obs_col_prefix + ycol] - data[obs_col_prefix + log2_mrl_cols[0]],
                data[pred_col_prefix + ycol] - data[pred_col_prefix + log2_mrl_cols[0]],
                ax,
                xlim=(axis_limits[row_idx]),
                ylim=(axis_limits[row_idx]),
                linreg=True,
            )

            ax.set_xlabel(f'{obs_label_prefix} Δlog2_MRL')
            if y_idx==0:
                ax.set_ylabel(f'{pred_label_prefix} Δlog2_MRL')

            ax.text(0.96, 0.96, f't = {timepoint_list[y_idx +1]} - 2 hpf', transform=ax.transAxes, ha='right', va='top')

            r2_dict[f"Δlog2_MRL_{timepoint_list[y_idx+1]}-2hpf"] = lrres.rvalue**2

        row_idx += 1

    # log2X differences
    if plot_delta_x:
        for y_idx, ycol in enumerate(delta_log2_x_cols):
            ax = axes[row_idx, y_idx]

            lrres = plot_scatter_shaded(
                data[obs_col_prefix + ycol],
                data[pred_col_prefix + ycol],
                ax,
                xlim=(axis_limits[row_idx]),
                ylim=(axis_limits[row_idx]),
                linreg=True,
            )

            ax.set_xlabel(f'{obs_label_prefix} Δlog2_X')
            if y_idx==0:
                ax.set_ylabel(f'{pred_label_prefix} Δlog2_X')

            if y_idx==0:
                ax.text(0.96, 0.96, f't = {timepoint_list[y_idx]} - 0 hpf', transform=ax.transAxes, ha='right', va='top')
            else:
                ax.text(0.96, 0.96, f't = {timepoint_list[y_idx]} - {timepoint_list[y_idx-1]} hpf', transform=ax.transAxes, ha='right', va='top')
            r2_dict[f"Δlog2_X_{timepoint_list[y_idx]}hpf"] = lrres.rvalue**2

        row_idx += 1

    # log2X
    if plot_x:
        for y_idx, ycol in enumerate(delta_log2_x_cols):
            ax = axes[row_idx, y_idx]

            lrres = plot_scatter_shaded(
                data[[obs_col_prefix + z for z in delta_log2_x_cols[:y_idx+1]]].sum(axis=1),
                data[[pred_col_prefix + z for z in delta_log2_x_cols[:y_idx+1]]].sum(axis=1),
                ax,
                xlim=(axis_limits[row_idx]),
                ylim=(axis_limits[row_idx]),
                linreg=True,
            )

            ax.set_xlabel(f'{obs_label_prefix} log2_X')
            if y_idx==0:
                ax.set_ylabel(f'{pred_label_prefix} log2_X')
            
            ax.text(0.96, 0.96, f't = {timepoint_list[y_idx]}hpf', transform=ax.transAxes, ha='right', va='top')

            r2_dict[f"log2_X_{timepoint_list[y_idx]}hpf"] = lrres.rvalue**2

        row_idx += 1
        
    if return_r2:
        return fig, r2_dict
    else:
        return fig

def plot_timecourse_ax(
        ax,
        data_df,
        seq_id_to_plot,
        time,
        measured_cols,
        pred_cols,
        pred_len_cols=None,
        rand_cols=None,
        global_label=None,
        color_dict={},
        linestyle_dict={},
        marker_dict={},
    ):
    
    ax.plot(
        time,
        data_df.loc[seq_id_to_plot, measured_cols].values,
        marker=marker_dict.get('measured', 'o'),
        color=color_dict.get('measured', 'darkblue'),
        linestyle=linestyle_dict.get('measured', '-'),
        label='Measured' if global_label is None else f"{global_label}, measured",
    )

    ax.plot(
        time,
        data_df.loc[seq_id_to_plot, pred_cols].values,
        marker=marker_dict.get('predicted', '^'),
        color=color_dict.get('predicted', 'tab:blue'),
        linestyle=linestyle_dict.get('predicted', '-'),
        label='Predicted' if global_label is None else f"{global_label}, predicted"
    )

    if rand_cols is not None:
        ax.plot(
            time,
            data_df.loc[seq_id_to_plot, rand_cols].values,
            marker=marker_dict.get('random', 's'),
            color=color_dict.get('random', 'k'),
            linewidth=1,
            linestyle=linestyle_dict.get('random', '--'),
            zorder=-1,
            label='Random' if global_label is None else f"{global_label}, random",
        )
    if pred_len_cols is not None:
        if global_label is None:
            label=f"Length model ({int(data_df.loc[seq_id_to_plot, 'insert_length']):d}nt)"
        else:
            label=f"{global_label}, length model ({int(data_df.loc[seq_id_to_plot, 'insert_length']):d}nt)"
        ax.plot(
            time,
            data_df.loc[seq_id_to_plot, pred_len_cols].values,
            marker=marker_dict.get('length', 's'),
            color=color_dict.get('length', 'gray'),
            linewidth=1,
            linestyle=linestyle_dict.get('length', '--'),
            zorder=-1,
            label=label,
        )
        
    return ax

def plot_single_timecourse(
        data_df,
        seq_id_to_plot,

        mode='cnn',

        plot_length_pred=True,
        # plot_random=False,

        plot_mrl=True,
        plot_delta_x=False,
        plot_x=False,

        mrl_lim=None,
        mrl_lim_auto=False,
        mrl_timeticks=None,

        delta_x_lim=None,
        delta_x_lim_auto=False,
        delta_x_timeticks=None,

        x_lim=None,
        x_lim_auto=False,
        x_timeticks=None,

        figsize=None,
        wspace=.25,
        savefig=None,
    ):

    if (not plot_mrl) and (not plot_delta_x) and not (plot_x):
        raise ValueError('one of plot_mrl, plot_delta_x, or plot_x should be True')
    if plot_delta_x and plot_x:
        raise ValueError('only one of plot_delta_x or plot_x should be True')
    n_plots = plot_mrl + plot_delta_x + plot_x
    
    if figsize is None:
        figsize = (4*n_plots, 4)
    if savefig is not None:
        fig, axes = pyplot.subplots(1, n_plots, squeeze=False, num=1, clear=True)
        fig.set_size_inches(figsize[0], figsize[1])
    else:
        fig, axes = pyplot.subplots(1, n_plots, squeeze=False, figsize=figsize)
    fig.subplots_adjust(wspace=wspace)

    data_seq_df = data_df.loc[[seq_id_to_plot]].copy()
    if plot_x:
        timepoint_list_with_zero = [0] + timepoint_list
        log2_x_cols_with_zero = ['log2_X_0hpf'] + log2_x_cols
        data_seq_df['log2_X_0hpf'] = 0
        if mode=='cnn':
            data_seq_df[[f'res_{c}' for c in log2_x_cols]] = \
                data_seq_df[[f'res_{c}' for c in delta_log2_x_cols]].cumsum(axis=1)
            data_seq_df['res_log2_X_0hpf'] = 0
            data_seq_df[[f'pred_cnn_ens_{c}' for c in log2_x_cols]] = \
                data_seq_df[[f'pred_cnn_ens_{c}' for c in delta_log2_x_cols]].cumsum(axis=1)
            data_seq_df['pred_cnn_ens_log2_X_0hpf'] = 0
        elif mode=='full':
            data_seq_df[[f'pred_full_{c}' for c in log2_x_cols]] = \
                data_seq_df[[f'pred_full_{c}' for c in delta_log2_x_cols]].cumsum(axis=1)
            data_seq_df['pred_full_log2_X_0hpf'] = 0
            data_seq_df[[f'pred_len_{c}' for c in log2_x_cols]] = \
                data_seq_df[[f'pred_len_{c}' for c in delta_log2_x_cols]].cumsum(axis=1)
            data_seq_df['pred_len_log2_X_0hpf'] = 0

    # MRL plot
    if plot_mrl:
        ax = axes[0, 0]

        if mode=='cnn':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in log2_mrl_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in log2_mrl_cols],
            )

        elif mode=='full':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list,
                measured_cols=log2_mrl_cols,
                pred_cols=[f'pred_full_{c}' for c in log2_mrl_cols],
                pred_len_cols=[f'pred_len_{c}' for c in log2_mrl_cols],
            )

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('log$_2$MRL')
        if not mrl_lim_auto:
            if mrl_lim is not None:
                ax.set_ylim(mrl_lim)
            else:
                if mode=='cnn':
                    ax.set_ylim(-4, 4)
                elif mode=='full':
                    ax.set_ylim(-1, 9)
        if mrl_timeticks is not None:
            ax.set_xticks(mrl_timeticks)
        if not (plot_delta_x or plot_x):
            ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.025))

    if plot_delta_x:
        if plot_mrl:
            ax = axes[0, 1]
        else:
            ax = axes[0, 0]

        if mode=='cnn':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in delta_log2_x_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in delta_log2_x_cols],
            )

        elif mode=='full':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list,
                measured_cols=delta_log2_x_cols,
                pred_cols=[f'pred_full_{c}' for c in delta_log2_x_cols],
                pred_len_cols=[f'pred_len_{c}' for c in delta_log2_x_cols],
            )

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Δlog$_2$X')
        if not delta_x_lim_auto:
            if delta_x_lim is not None:
                ax.set_ylim(delta_x_lim)
            else:
                if mode=='cnn':
                    ax.set_ylim(-2, 2)
                elif mode=='full':
                    ax.set_ylim(-5.5, 1.2)
        if delta_x_timeticks is not None:
            ax.set_xticks(delta_x_timeticks)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.025))
    
    elif plot_x:
        if plot_mrl:
            ax = axes[0, 1]
        else:
            ax = axes[0, 0]

        if mode=='cnn':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list_with_zero,
                measured_cols=[f'res_{c}' for c in log2_x_cols_with_zero],
                pred_cols=[f'pred_cnn_ens_{c}' for c in log2_x_cols_with_zero],
            )

        elif mode=='full':
            plot_timecourse_ax(
                ax,
                data_seq_df,
                seq_id_to_plot,
                timepoint_list_with_zero,
                measured_cols=log2_x_cols_with_zero,
                pred_cols=[f'pred_full_{c}' for c in log2_x_cols_with_zero],
                pred_len_cols=[f'pred_len_{c}' for c in log2_x_cols_with_zero],
            )

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('log$_2$X')
        if not x_lim_auto:
            if x_lim is not None:
                ax.set_ylim(x_lim)
            else:
                if mode=='cnn':
                    ax.set_ylim(-9, 0.5)
                elif mode=='full':
                    ax.set_ylim(-9, 0.5)
        if x_timeticks is not None:
            ax.set_xticks(x_timeticks)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.025))

    fig.suptitle(f"{seq_id_to_plot}")

    if savefig is not None:
        fig.savefig(savefig, dpi=200, bbox_inches='tight')
        # pyplot.close(fig)
        return fig
    else:
        return fig

def plot_switching_timecourse(
        data_df,
        pre_seq_id,
        post_seq_id,
        pre_label='Maternal',
        post_label='Zygotic',

        title=None,

        mode='cnn',

        plot_length_pred=True,
        # plot_random=False,

        plot_mrl=True,
        plot_delta_x=True,

        mrl_lim=None,
        mrl_timeticks=None,

        delta_x_lim=None,
        delta_x_timeticks=None,

        figsize=None,
        wspace=.25,
        savefig=None,
):
    if (not plot_mrl) and (not plot_delta_x):
        raise ValueError('one of plot_mrl or plot_delta_x should be True')
    n_plots = plot_mrl + plot_delta_x
    
    if figsize is None:
        figsize = (4*n_plots, 4)
    if savefig is not None:
        fig, axes = pyplot.subplots(1, n_plots, squeeze=False, num=1, clear=True)
        fig.set_size_inches(figsize[0], figsize[1])
    else:
        fig, axes = pyplot.subplots(1, n_plots, squeeze=False, figsize=figsize)
    fig.subplots_adjust(wspace=wspace)

    # MRL plot
    if plot_mrl:
        ax = axes[0, 0]

        # measured, predicted, random, length
        # color_dict={},
        # linestyle_dict={},
        # marker_dict={},

        if mode=='cnn':
            plot_timecourse_ax(
                ax,
                data_df,
                pre_seq_id,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in log2_mrl_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in log2_mrl_cols],
                global_label=pre_label,
                color_dict={'measured': 'darkred', 'predicted': 'tab:red'},
                marker_dict={'measured': 'o', 'predicted': '^'},
            )
            plot_timecourse_ax(
                ax,
                data_df,
                post_seq_id,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in log2_mrl_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in log2_mrl_cols],
                global_label=post_label,
                color_dict={'measured': 'darkblue', 'predicted': 'tab:blue'},
                marker_dict={'measured': 'o', 'predicted': '^'},
            )

        elif mode=='full':
            plot_timecourse_ax(
                ax,
                data_df,
                pre_seq_id,
                timepoint_list,
                measured_cols=log2_mrl_cols,
                pred_cols=[f'pred_full_{c}' for c in log2_mrl_cols],
                pred_len_cols=[f'pred_len_{c}' for c in log2_mrl_cols] if plot_length_pred else None,
                global_label=pre_label,
                color_dict={'measured': 'darkred', 'predicted': 'tab:red', 'length': 'lightsalmon'},
                marker_dict={'measured': 'o', 'predicted': '^', 'length': 's'},
                linestyle_dict={'length': '--'},
            )
            plot_timecourse_ax(
                ax,
                data_df,
                post_seq_id,
                timepoint_list,
                measured_cols=log2_mrl_cols,
                pred_cols=[f'pred_full_{c}' for c in log2_mrl_cols],
                pred_len_cols=[f'pred_len_{c}' for c in log2_mrl_cols] if plot_length_pred else None,
                global_label=post_label,
                color_dict={'measured': 'darkblue', 'predicted': 'tab:blue', 'length': 'lightblue'},
                marker_dict={'measured': 'o', 'predicted': '^', 'length': 's'},
                linestyle_dict={'length': '--'},
            )

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('log2(MRL)')
        if mrl_lim is not None:
            if mrl_lim != 'auto':
                ax.set_ylim(mrl_lim)
        else:
            if mode=='cnn':
                ax.set_ylim(-4, 4)
            elif mode=='full':
                ax.set_ylim(-1, 9)
        if mrl_timeticks is not None:
            ax.set_xticks(mrl_timeticks)
        if not plot_delta_x:
            ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.025))

    if plot_delta_x:
        if plot_mrl:
            ax = axes[0, 1]
        else:
            ax = axes[0, 0]

        if mode=='cnn':
            plot_timecourse_ax(
                ax,
                data_df,
                pre_seq_id,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in delta_log2_x_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in delta_log2_x_cols],
                global_label=pre_label,
                color_dict={'measured': 'darkred', 'predicted': 'tab:red'},
                marker_dict={'measured': 'o', 'predicted': '^'},
            )
            plot_timecourse_ax(
                ax,
                data_df,
                post_seq_id,
                timepoint_list,
                measured_cols=[f'res_{c}' for c in delta_log2_x_cols],
                pred_cols=[f'pred_cnn_ens_{c}' for c in delta_log2_x_cols],
                global_label=post_label,
                color_dict={'measured': 'darkblue', 'predicted': 'tab:blue'},
                marker_dict={'measured': 'o', 'predicted': '^'},
            )

        elif mode=='full':
            plot_timecourse_ax(
                ax,
                data_df,
                pre_seq_id,
                timepoint_list,
                measured_cols=delta_log2_x_cols,
                pred_cols=[f'pred_full_{c}' for c in delta_log2_x_cols],
                pred_len_cols=[f'pred_len_{c}' for c in delta_log2_x_cols] if plot_length_pred else None,
                global_label=pre_label,
                color_dict={'measured': 'darkred', 'predicted': 'tab:red', 'length': 'lightsalmon'},
                marker_dict={'measured': 'o', 'predicted': '^', 'length': 's'},
                linestyle_dict={'length': '--'},
            )
            plot_timecourse_ax(
                ax,
                data_df,
                post_seq_id,
                timepoint_list,
                measured_cols=delta_log2_x_cols,
                pred_cols=[f'pred_full_{c}' for c in delta_log2_x_cols],
                pred_len_cols=[f'pred_len_{c}' for c in delta_log2_x_cols] if plot_length_pred else None,
                global_label=post_label,
                color_dict={'measured': 'darkblue', 'predicted': 'tab:blue', 'length': 'lightblue'},
                marker_dict={'measured': 'o', 'predicted': '^', 'length': 's'},
                linestyle_dict={'length': '--'},
            )

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Δlog2(X)')
        if delta_x_lim is not None:
            if delta_x_lim != 'auto':
                ax.set_ylim(delta_x_lim)
        else:
            ax.set_ylim(-7, 1)
        if delta_x_timeticks is not None:
            ax.set_xticks(delta_x_timeticks)
        # # Add legend, but manually add one additional line for the length model
        # if mode=='full' and plot_length_pred:
        #     label=f"Length model ({int(data_df.loc[seq_id_to_plot, 'insert_length']):d}nt)"
        #     ax.plot([], [], color='gray', linestyle='--', label=label)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.025))

    if title is not None:
        fig.suptitle(title)

    if savefig is not None:
        fig.savefig(savefig, dpi=200, bbox_inches='tight')
        # pyplot.close(fig)
        return fig
    else:
        return fig

def plot_contribution_scores(
        data_df,
        contributions_dict,
        seq_id_to_plot,
        outputs_to_plot=None,
        pos=None,
        xtick_res=5,
        ylim=None,
        ytickpos='left',
        make_extended_shap_fxn=make_extended_shap,
        model_outputs_ext_labels=model_outputs_ext_labels,
        savefig=None,
    ):

    if outputs_to_plot is None:
        outputs_to_plot = model_outputs_ext_labels
    # if any(o not in model_outputs_ext_labels for o in outputs_to_plot):
    #     outputs_to_plot = [model_output_to_ext[o] for o in outputs_to_plot]

    contribution_scores = contributions_dict[seq_id_to_plot]
    contribution_scores_ext = make_extended_shap_fxn(contribution_scores)

    seq = data_df.loc[seq_id_to_plot, 'insert_seq']
    seq_onehot = seq_utils.one_hot_encode(
        [seq],
        max_seq_len=contribution_scores.shape[1], padding='right', mask_val=0,
    )

    seq_length = int(data_df.loc[seq_id_to_plot, 'insert_length'])
    if pos is None:
        pos = [-seq_length, -1]
        seq_to_plot_len = seq_length
    else:
        if (pos[0] is None) or (pos[0] < -seq_length):
            pos[0] = -seq_length
        if (pos[1] is None) or (pos[1] > -1):
            pos[1] = -1
        seq_to_plot_len = pos[1] - pos[0] + 1
    
    if savefig is not None:
        fig = pyplot.figure(num=1, clear=True)
    else:
        fig = pyplot.figure()
    figsize = (seq_to_plot_len*0.2, 0.8*(len(outputs_to_plot) + 0.4))
    fig.set_size_inches(figsize)
    gs = fig.add_gridspec(
        1 + len(outputs_to_plot), 1,
        height_ratios=[0.32] + [0.8]*len(outputs_to_plot),
    )
    seq_ax = fig.add_subplot(gs[0])
    contrib_axes = [fig.add_subplot(gs[i+1]) for i in range(len(outputs_to_plot))]

    # Plot sequence
    if pos[1] >= -1:
            seq_onehot_to_plot = seq_onehot[0, pos[0]:, :]
    else:
        seq_onehot_to_plot = seq_onehot[0, pos[0]:pos[1] + 1, :]
    plot_seq_logo(seq_val=seq_onehot_to_plot, ax=seq_ax)
    seq_ax.set_yticks([])
    seq_ax.set_ylim(0, 2)
    seq_ax.spines['top'].set_visible(False)
    seq_ax.spines['right'].set_visible(False)
    seq_ax.spines['bottom'].set_visible(False)
    seq_ax.spines['left'].set_visible(False)

    # Plot contributions
    for output_idx, output in enumerate(outputs_to_plot):
        ax = contrib_axes[output_idx]
        model_output = outputs_to_plot[output_idx]
        model_output_idx = model_outputs_ext_labels.index(model_output)

        # Scores to plot
        # Project by only considering the actual base
        scores_projected = contribution_scores_ext[model_output_idx][None,:,:]*seq_onehot

        if pos[1] >= -1:
            scores_to_plot = scores_projected[0, pos[0]:, :]
        else:
            scores_to_plot = scores_projected[0, pos[0]:pos[1] + 1, :]
        plot_seq_logo(nt_height=scores_to_plot, ax=ax)

        ax.set_ylabel(model_output, rotation=0, ha='right', va='center')
        if ytickpos=='right':
            ax.yaxis.tick_right()
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(False)
    
    # Common settings for all axes
    for ax_idx, ax in enumerate([seq_ax] + contrib_axes):
        ax.set_xlim(-0.5, seq_to_plot_len - 0.5)
        xticklabels = [-i for i in range(0, seq_length, xtick_res) if -i >= pos[0] and -i <= pos[1]]
        if len(xticklabels) <= 0:
            xticklabels = [pos[1]]
        xticks = numpy.array(xticklabels) - pos[0]
        ax.set_xticks(xticks)
        if ax_idx < len(outputs_to_plot):
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(xticklabels)

    # Y axis limits shared for all except the first axis
    if ylim is None:
        ylim = [numpy.inf, -numpy.inf]

        for ax_idx, ax in enumerate(contrib_axes):
            ax_ylim = contrib_axes[ax_idx].get_ylim()
            ylim[0] = min(ylim[0], ax_ylim[0])
            ylim[1] = max(ylim[1], ax_ylim[1])
            
    for ax_idx, ax in enumerate(contrib_axes):
        contrib_axes[ax_idx].set_ylim(ylim)

    if savefig is not None:
        fig.savefig(savefig, dpi=200, bbox_inches='tight')
#         pyplot.close(fig)
        return fig
    else:
        return fig

