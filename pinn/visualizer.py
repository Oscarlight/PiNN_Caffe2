import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pylab import rcParams
from matplotlib.ticker import AutoMinorLocator

FIGURE_SIZE = (7, 8)
FONT_SIZE = 16
LINE_WIDTH = 2
MAJOR_LABEL_SIZE = 14
MINOR_LABEL_SIZE = 0

def write_csv_file(file_name, data, description):
    ''' Write data to .csv file.'''
    np.savetxt(file_name, data, delimiter=',', newline='\n', header=description)

def VgVd_counter (vg, vd):

    vd_count = 0

    vg_tmp = list(vg)
    vg_tmp.sort()
    while (vg_tmp[vd_count + 1] == vg_tmp[vd_count]):
        vd_count = vd_count + 1

    vd_count = vd_count + 1
    vg_count = len(vg) / vd_count

    return vg_count, vd_count

def plot_data (tag, x, y, tag_count, x_count, line_style, y_scale):
    i = 0
    if (y_scale == 'log'):
        y_in = np.abs(y)
    elif (y_scale == 'linear'):
        y_in = y
    else:
        raise Exception('Please choose linear or log for y-scale!')

    while (i < tag_count):
        x_tmp = x[i * x_count: (i + 1) * x_count]
        y_tmp = y_in[i * x_count: (i + 1) * x_count]
        # vg_label = 'vg='+str(vg[i*vd_count])
        plt.plot(x_tmp, y_tmp, ls=line_style, color=colors.cnames.keys()[i])
        i = i + 1

def plot_linear_fig (fig, ax, x_name, y_name, save_name):
    xminorLocator = AutoMinorLocator()
    yminorLocator = AutoMinorLocator()

    plt.xlabel(x_name, fontsize=FONT_SIZE)
    plt.ylabel(y_name, fontsize=FONT_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=MAJOR_LABEL_SIZE)
    plt.tick_params(axis='both', which='minor', labelsize=MINOR_LABEL_SIZE)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # plt.legend(loc = 'upper left')
    if save_name is not None:
        plt.savefig(save_name)

def plot_log_fig (fig, ax, x_name, y_name, save_name):
    xminorLocator = AutoMinorLocator()
    yminorLocator = AutoMinorLocator()

    plt.xlabel(x_name, fontsize=FONT_SIZE)
    plt.ylabel(y_name, fontsize=FONT_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=MAJOR_LABEL_SIZE)
    plt.tick_params(axis='both', which='minor', labelsize=MINOR_LABEL_SIZE)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.yscale('log')
    # plt.legend(loc = 'upper left')
    if save_name is not None:
        plt.savefig(save_name)

def sort_vg_func (vg, vd, ids):
    # Re-arrange the data so that data is sorted by vg
    idx = sorted(range(len(vg)), key=lambda x : (vg[x], vd[x]))
    return vg[idx], vd[idx], ids[idx]

def sort_vd_func (vg, vd, ids):
    # Re-arrange the data so that data is sorted by vd
    idx = sorted(range(len(vg)), key=lambda x : (vd[x], vg[x]))
    return vg[idx], vd[idx], ids[idx]

def plot_linear_Id_vs_Vd_at_Vg(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$'):

    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vg_func (vg, vd, ids)
    plot_data(sort_vg, sort_vd, sort_ids, vg_count, vd_count, 'solid', 'linear')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_linear_fig(fig, ax,'V$_d$', yLabel, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vg_func (vg_comp, vd_comp, ids_comp)
        vg_count_comp, vd_count_comp = VgVd_counter(vg_comp, vd_comp)
        plot_data(sort_vg_comp, sort_vd_comp, sort_ids_comp, vg_count_comp, vd_count_comp, '--', 'linear')
        plot_linear_fig(fig, ax,'V$_d$', yLabel, save_name)
        plt.show()

def plot_linear_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$' ):

    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vd_func (vg, vd, ids)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid', 'linear')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_linear_fig(fig, ax, 'V$_g$', yLabel, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vd_func (vg_comp, vd_comp, ids_comp)
        vg_count_comp, vd_count_comp = VgVd_counter(vg_comp, vd_comp)
        plot_data (sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count_comp, vg_count_comp, '--', 'linear')
        plot_linear_fig(fig, ax, 'V$_g$', yLabel, save_name)
        plt.show()

def plot_log_Id_vs_Vd_at_Vg(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$'):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vg_func(vg, vd, ids)
    plot_data(sort_vg, sort_vd, sort_ids, vg_count, vd_count, 'solid', 'log')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, 'V$_d$', yLabel, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vg_func(vg_comp, vd_comp, ids_comp)
        vg_count_comp, vd_count_comp = VgVd_counter(vg_comp, vd_comp)
        plot_data(sort_vg_comp, sort_vd_comp, sort_ids_comp, vg_count_comp, vd_count_comp, '--', 'log')
        plot_log_fig(fig, ax, 'V$_d$', yLabel, save_name)
        plt.show()

def plot_log_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$'):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vd_func(vg, vd, ids)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid', 'log')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, 'V$_g$', yLabel, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vd_func(vg_comp, vd_comp, ids_comp)
        vg_count_comp, vd_count_comp = VgVd_counter(vg_comp, vd_comp)
        plot_data(sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count_comp, vg_count_comp, '--', 'log')
        plot_log_fig(fig, ax, 'V$_g$', yLabel, save_name)
        plt.show()
