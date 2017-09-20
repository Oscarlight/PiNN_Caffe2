import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pylab import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

def write_csv_file(file_name, data, description):
    ''' Write data to .csv file.'''
    np.savetxt(file_name, data, delimiter=',', newline='\n', header=description)

def VgVd_counter (vg, vd):

    vd_count = 0

    vg_tmp = vg
    vg_tmp.sort()
    while (vg_tmp[vd_count + 1] == vg_tmp[vd_count]):
        vd_count = vd_count + 1

    vd_count = vd_count + 1
    vg_count = len(vg) / vd_count

    return vg_count, vd_count

def plot_data (tag, x, y, tag_count, x_count, line_style):
    i = 0
    while (i < tag_count):
        x_tmp = x[i*x_count : (i+1)*x_count]
        y_tmp = np.abs(y[i*x_count : (i+1)*x_count])
        # vg_label = 'vg='+str(vg[i*vd_count])
        plt.plot(x_tmp, y_tmp, ls = line_style, color = colors.cnames.keys()[i])
        i = i+1

def plot_linear_fig (fig, ax, save_name):
    xminorLocator = AutoMinorLocator()
    yminorLocator = AutoMinorLocator()

    plt.xlabel('vd')
    plt.ylabel('id')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=0)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # plt.legend(loc = 'upper left')
    if save_name is not None:
        plt.savefig(save_name)

def plot_log_fig (fig, ax, save_name):
    xminorLocator = AutoMinorLocator()
    yminorLocator = AutoMinorLocator()

    plt.xlabel('vd')
    plt.ylabel('id')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=0)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.yscale('log')
    # plt.legend(loc = 'upper left')
    if save_name is not None:
        plt.savefig(save_name)

def sort (vg, vd, ids, vg_count, vd_count):
    # Re-arrange the data so that data is sorted by vd instead of vg
    sort_vg = []
    sort_vd = []
    sort_ids = []
    i = 0
    while (i < vd_count):
        j = 0
        while (j < vg_count):
            sort_vg.append(vg[j * vd_count + i])
            sort_vd.append(vd[j * vd_count + i])
            sort_ids.append(ids[j * vd_count + i])
            j = j + 1
        i = i + 1

    return sort_vg, sort_vd, sort_ids

def plot_linear_Id_vs_Vd_at_Vg(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None):

    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = 6.5, 10
    rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots()
    plot_data (vg, vd, ids, vg_count, vd_count, 'solid')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_linear_fig(fig, ax, save_name)
        plt.show()

    else:
        plot_data (vg_comp, vd_comp, ids_comp, vg_count, vd_count, '--')
        plot_linear_fig(fig, ax, save_name)
        plt.show()

def plot_linear_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None):

    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = 6.5, 10
    rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots()
    # Re-arrange the data so that data is sorted by vd instead of vg
    sort_vg, sort_vd, sort_ids = sort(vg, vd, ids, vg_count, vd_count)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_linear_fig(fig, ax, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort(vg_comp, vd_comp, ids_comp, vg_count, vd_count)
        plot_data (sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count, vg_count, '--')
        plot_linear_fig(fig, ax, save_name)
        plt.show()

def plot_log_Id_vs_Vd_at_Vg(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = 6.5, 10
    rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots()
    plot_data(vg, vd, ids, vg_count, vd_count, 'solid')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, save_name)
        plt.show()

    else:
        plot_data(vg_comp, vd_comp, ids_comp, vg_count, vd_count, '--')
        plot_log_fig(fig, ax, save_name)
        plt.show()

def plot_log_Id_vs_Vg_at_Vd(vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None, save_name = None):
    vg_count, vd_count = VgVd_counter(vg, vd)
    rcParams['figure.figsize'] = 6.5, 10
    rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots()
    sort_vg, sort_vd, sort_ids = sort(vg, vd, ids, vg_count, vd_count)
    plot_data(sort_vd, sort_vg, sort_ids, vd_count, vg_count, 'solid')

    if (vg_comp is None or vd_comp is None or ids_comp is None):
        plot_log_fig(fig, ax, save_name)
        plt.show()

    else:
        sort_vg_comp, sort_vd_comp, sort_ids_comp = sort(vg_comp, vd_comp, ids_comp, vg_count, vd_count)
        plot_data(sort_vd_comp, sort_vg_comp, sort_ids_comp, vd_count, vg_count, '--')
        plot_log_fig(fig, ax, save_name)
        plt.show()