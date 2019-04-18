import matplotlib
import matplotlib.pyplot as plt


def setup_figure_1ax(x_label='', y_label='', size=(13, 9), shrink_ax=True):
    """Returns a (figure, ax) tuple with legend on the right hand side, no spines."""

    matplotlib.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Shrink current axis by 20%
    if shrink_ax:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.grid()
    return fig, ax