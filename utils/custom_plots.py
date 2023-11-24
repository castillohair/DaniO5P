import numpy
import scipy
import scipy.stats
import matplotlib
from matplotlib import pyplot
import pandas

import logomaker

greysBig = matplotlib.cm.get_cmap('Greys', 512)
greys_trunc_cm = matplotlib.colors.ListedColormap(greysBig(numpy.linspace(0.6, 1, 256)))

def plot_scatter_shaded(x, y, ax, xlim=None, ylim=None, linreg=False, cmap=greys_trunc_cm, label=None):
    """
    Scatter plot where the markers are shaded by density

    """
    xy = numpy.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy)(xy)
    ax.scatter(
        x,
        y,
        # s=5,
        # alpha=0.5,
        s=0.5,
        c=z,
        cmap=cmap,
        rasterized=True,
        label=label,
    )

    if linreg:
        lrres = scipy.stats.linregress(
            x,
            y,
        )
        ax.axline((0, lrres.intercept), slope=lrres.slope, color='dodgerblue', linewidth=2)
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

nt_color_dict = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'U': (228/255, 38/255, 56/255),
}

def plot_seq_logo(nt_height=None, seq_val=None, ax=None, title=None, spines=True):
    """
    Plot a sequence logo based on nucleotide height or sequence string

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