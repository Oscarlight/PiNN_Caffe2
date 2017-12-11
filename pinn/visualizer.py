import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pylab import rcParams
from matplotlib.ticker import AutoMinorLocator

FIGURE_SIZE = (8, 12)
FONT_SIZE = 22
LINE_WIDTH = 2
MAJOR_LABEL_SIZE = 22
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

##TODO: tag_count should be depreciated
def plot_data (tag, x, y, tag_count, x_count, line_style, y_scale):
    i = 0
    if (y_scale == 'log'):
        y_in = np.abs(y)
    elif (y_scale == 'linear'):
        y_in = y
    else:
        raise Exception('Please choose linear or log for y-scale!')

    while (i * x_count < len(x)):
        if (i + 1) * x_count < len(x):
            x_tmp = x[i * x_count: (i + 1) * x_count]
            y_tmp = y_in[i * x_count: (i + 1) * x_count]
        else:
            x_tmp = x[i * x_count: ]
            y_tmp = y_in[i * x_count: ]           

        # plt.plot(x_tmp, y_tmp, ls=line_style, color=colors.cnames.keys()[i])
        if line_style == 'solid':
            plt.plot(x_tmp, y_tmp, linewidth=1, ls='solid', color='#3498db')
            # plt.plot(x_tmp, y_tmp, linewidth=1, ls='solid', color='#143c57')
            # color gradient
            # cmap=plt.get_cmap('Blues')
            # plt.plot(x_tmp, y_tmp, linewidth=1, ls='solid', color=cmap(100 + 8*i) )
        elif line_style == '0':
            plt.scatter(x_tmp, y_tmp, s=50, alpha = 1, color='#e74c3c')
            # plt.scatter(x_tmp, y_tmp, s=50, alpha = 0.6, color='#3498db')
        elif line_style == '1':
            plt.scatter(x_tmp, y_tmp, s=50, alpha = 0.6, color='#3498db')
        # plt.xlim([-0.8, 2.7])
        # plt.ylim([0, 3])
        i = i + 1

def plot_linear_fig (fig, ax, x_name, y_name, save_name):
    xminorLocator = AutoMinorLocator()
    yminorLocator = AutoMinorLocator()

    plt.xlabel(x_name, fontsize=FONT_SIZE)
    plt.ylabel(y_name, fontsize=FONT_SIZE)
    plt.tick_params(axis='both', which='major', length=10, labelsize=MAJOR_LABEL_SIZE)
    plt.tick_params(axis='both', which='minor', length=5, labelsize=MINOR_LABEL_SIZE)
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
    plt.tick_params(axis='both', which='major', length=10, labelsize=MAJOR_LABEL_SIZE)
    plt.tick_params(axis='both', which='minor', length=5, labelsize=MINOR_LABEL_SIZE)
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
        plot_linear_fig(fig, ax,'', yLabel, save_name)
        # plt.show()

    else:
        i = 0
        for _vg_comp, _vd_comp, _ids_comp in zip(vg_comp, vd_comp, ids_comp):
            sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vg_func (_vg_comp, _vd_comp, _ids_comp)
            vg_count_comp, vd_count_comp = VgVd_counter(_vg_comp, _vd_comp)
            plot_data(sort_vg_comp, sort_vd_comp, sort_ids_comp, vg_count_comp, vd_count_comp, str(i), 'linear')
            plot_linear_fig(fig, ax,'', yLabel, save_name)
            i += 1
            # plt.show()

def plot_linear_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$' ):

    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vd_func (vg, vd, ids)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid', 'linear')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_linear_fig(fig, ax, '', yLabel, save_name)
        # plt.show()

    else:
        i = 0
        for _vg_comp, _vd_comp, _ids_comp in zip(vg_comp, vd_comp, ids_comp):
            sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vd_func (_vg_comp, _vd_comp, _ids_comp)
            vg_count_comp, vd_count_comp = VgVd_counter(_vg_comp, _vd_comp)
            plot_data (sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count_comp, vg_count_comp, str(i), 'linear')
            plot_linear_fig(fig, ax, '', yLabel, save_name)
            i += 1
            # plt.show()

def plot_log_Id_vs_Vd_at_Vg(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$'):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vg_func(vg, vd, ids)
    plot_data(sort_vg, sort_vd, sort_ids, vg_count, vd_count, 'solid', 'log')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, '', yLabel, save_name)
        # plt.show()

    else:
        i = 0
        for _vg_comp, _vd_comp, _ids_comp in zip(vg_comp, vd_comp, ids_comp):
            sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vg_func(_vg_comp, _vd_comp, _ids_comp)
            vg_count_comp, vd_count_comp = VgVd_counter(_vg_comp, _vd_comp)
            plot_data(sort_vg_comp, sort_vd_comp, sort_ids_comp, vg_count_comp, vd_count_comp, str(i), 'log')
            plot_log_fig(fig, ax, '', yLabel, save_name)
            i += 1
            # plt.show()

def plot_log_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None, yLabel = 'I$_d$'):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = FIGURE_SIZE
    rcParams['axes.linewidth'] = LINE_WIDTH
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort_vd_func(vg, vd, ids)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid', 'log')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, '', yLabel, save_name)
        # plt.show()

    else:
        i = 0
        for _vg_comp, _vd_comp, _ids_comp in zip(vg_comp, vd_comp, ids_comp):
            sort_vg_comp, sort_vd_comp, sort_ids_comp = sort_vd_func(_vg_comp, _vd_comp, _ids_comp)
            vg_count_comp, vd_count_comp = VgVd_counter(_vg_comp, _vd_comp)
            plot_data(sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count_comp, vg_count_comp, str(i), 'log')
            plot_log_fig(fig, ax, '', yLabel, save_name)
            i += 1
            # plt.show()
