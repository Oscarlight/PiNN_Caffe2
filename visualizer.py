import numpy as np
import matplotlib.pyplot as plt

def write_csv_file (file_name, data, description):
    ''' Write data to .csv file.'''
    np.savetxt(file_name, data, delimiter=',', newline='\n', header=description)


def plot_linear_Id_vs_Vd_at_Vg (header, vg, vd, id):

    vg_count = 0
    vd_count = 0
    for item in header['Inputs']:
        if (item[0] == 'vd'):
            vd_count = int(item[10])
        elif (item[0] == 'vg'):
            vg_count = int(item[10])
        else:
            pass

    i = 0
    while (i < vg_count):
        vd_tmp = vd[i*vd_count : i*vd_count+vd_count]
        id_tmp = id[i*vd_count : i*vd_count+vd_count]
        vg_label = 'vg='+str(vg[i*vd_count])
        plt.plot(vd_tmp, id_tmp, label = vg_label)
        i = i+1

    plt.xlabel('vd')
    plt.ylabel('id')
    plt.legend(loc = 'upper left')
    plt.show()

def plot_linear_Id_vs_Vg_at_Vd (header, vg, vd, id):

    vg_count = 0
    vd_count = 0
    for item in header['Inputs']:
        if (item[0] == 'vd'):
            vd_count = int(item[10])
        elif (item[0] == 'vg'):
            vg_count = int(item[10])
        else:
            pass

    # Re-arrange the data so that data is sorted by vd instead of vg
    sort_vg = []
    sort_vd = []
    sort_id = []
    i = 0
    while (i < vd_count):
        j = 0
        while (j < vg_count):
            sort_vg.append(vg[j*vd_count+i])
            sort_vd.append(vd[j*vd_count+i])
            sort_id.append(id[j*vd_count+i])
            j = j+1
        i = i+1

    k = 0
    while (k < vd_count):
        vg_tmp = sort_vg[k*vg_count : k*vg_count+vg_count]
        id_tmp = sort_id[k*vg_count : k*vg_count+vg_count]
        vd_label = 'vd='+str(sort_vd[k*vg_count])
        plt.plot(vg_tmp, id_tmp, label = vd_label)
        k = k+1

    plt.xlabel('vg')
    plt.ylabel('id')
    plt.legend(loc = 'upper left', ncol = 3)
    plt.show()

def plot_log_Id_vs_Vd_at_Vg (header, vg, vd, id):

    vg_count = 0
    vd_count = 0
    for item in header['Inputs']:
        if (item[0] == 'vd'):
            vd_count = int(item[10])
        elif (item[0] == 'vg'):
            vg_count = int(item[10])
        else:
            pass

    i = 0
    while (i < vg_count):
        vd_tmp = vd[i*vd_count : i*vd_count+vd_count]
        id_tmp = np.abs(id[i*vd_count : i*vd_count+vd_count])
        vg_label = 'vg='+str(vg[i*vd_count])
        print(vd_tmp)
        print(len(vd_tmp))
        print(id_tmp)
        print(len(id_tmp))
        plt.plot(vd_tmp, id_tmp, label = vg_label)
        i = i+1

    plt.xlabel('vd')
    plt.ylabel('id')
    plt.yscale('log')
    plt.legend(loc = 'lower right')
    plt.show()

def plot_log_Id_vs_Vg_at_Vd (header, vg, vd, id):

    vg_count = 0
    vd_count = 0
    for item in header['Inputs']:
        if (item[0] == 'vd'):
            vd_count = int(item[10])
        elif (item[0] == 'vg'):
            vg_count = int(item[10])
        else:
            pass

    # Re-arrange the data so that data is sorted by vd instead of vg
    sort_vg = []
    sort_vd = []
    sort_id = []
    i = 0
    while (i < vd_count):
        j = 0
        while (j < vg_count):
            sort_vg.append(vg[j*vd_count+i])
            sort_vd.append(vd[j*vd_count+i])
            sort_id.append(id[j*vd_count+i])
            j = j+1
        i = i+1

    k = 0
    while (k < vd_count):
        vg_tmp = sort_vg[k*vg_count : k*vg_count+vg_count]
        id_tmp = np.abs(sort_id[k*vg_count : k*vg_count+vg_count])
        vd_label = 'vd='+str(sort_vd[k*vg_count])
        plt.plot(vg_tmp, id_tmp, label = vd_label)
        k = k+1

    plt.xlabel('vg')
    plt.ylabel('id')
    plt.yscale('log')
    plt.legend(loc = 'lower right', ncol = 3)
    plt.show()