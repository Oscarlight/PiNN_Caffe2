import numpy as np
import matplotlib.pyplot as plt

def write_csv_file (file_name, data, description):
    ''' Write data to .csv file.'''
    np.savetxt(file_name, data, delimiter=',', newline='\n', header=description)

def plot_linear_Id_vs_Vd_at_Vg (vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None):

    vd_count = 0

    vg_tmp = vg
    vg_tmp.sort()
    while (vg_tmp[vd_count+1] == vg_tmp[vd_count]):
        vd_count = vd_count +1

    vd_count = vd_count + 1
    vg_count = len(vg)/vd_count

    i = 0
    while (i < vg_count):
        vd_tmp = vd[i*vd_count : i*vd_count+vd_count]
        ids_tmp = ids[i*vd_count : i*vd_count+vd_count]
        vg_label = 'vg='+str(vg[i*vd_count])
        plt.plot(vd_tmp, ids_tmp, label = vg_label)
        i = i+1


    if (vg_comp == None):
        plt.xlabel('vd')
        plt.ylabel('id')
        plt.legend(loc = 'upper left')
        plt.show()

    else:
        j = 0
        while (j < vg_count):
            vd_comp_tmp = vd_comp[j * vd_count: j * vd_count + vd_count]
            ids_comp_tmp = ids_comp[j * vd_count: j * vd_count + vd_count]
            vg_comp_label = 'vg=' + str(vg[j * vd_count])
            plt.plot(vd_comp_tmp, ids_comp_tmp, label=vg_comp_label, ls='--')
            j = j + 1

        plt.xlabel('vd')
        plt.ylabel('id')
        plt.legend(loc='best', ncol = 2)
        plt.show()

def plot_linear_Id_vs_Vg_at_Vd (vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None):

    vd_count = 0

    vg_tmp = vg
    vg_tmp.sort()
    while (vg_tmp[vd_count+1] == vg_tmp[vd_count]):
        vd_count = vd_count +1

    vd_count = vd_count + 1
    vg_count = len(vg)/vd_count

    # Re-arrange the data so that data is sorted by vd instead of vg
    sort_vg = []
    sort_vd = []
    sort_ids = []
    i = 0
    while (i < vd_count):
        j = 0
        while (j < vg_count):
            sort_vg.append(vg[j*vd_count+i])
            sort_vd.append(vd[j*vd_count+i])
            sort_ids.append(ids[j*vd_count+i])
            j = j+1
        i = i+1

    k = 0
    while (k < vd_count):
        vg_tmp = sort_vg[k*vg_count : k*vg_count+vg_count]
        ids_tmp = sort_ids[k*vg_count : k*vg_count+vg_count]
        vd_label = 'vd='+str(sort_vd[k*vg_count])
        plt.plot(vg_tmp, ids_tmp, label = vd_label)
        k = k+1

    if (vg_comp == None):
        plt.xlabel('vg')
        plt.ylabel('id')
        plt.legend(loc = 'upper left', ncol = 3)
        plt.show()

    else:
        sort_vg_comp = []
        sort_vd_comp = []
        sort_ids_comp = []
        i = 0
        while (i < vd_count):
            j = 0
            while (j < vg_count):
                sort_vg_comp.append(vg_comp[j * vd_count + i])
                sort_vd_comp.append(vd_comp[j * vd_count + i])
                sort_ids_comp.append(ids_comp[j * vd_count + i])
                j = j + 1
            i = i + 1

        k = 0
        while (k < vd_count):
            vg_tmp_comp = sort_vg_comp[k * vg_count: k * vg_count + vg_count]
            ids_tmp_comp = sort_ids_comp[k * vg_count: k * vg_count + vg_count]
            vd_label_comp = 'vd=' + str(sort_vd_comp[k * vg_count])
            plt.plot(vg_tmp_comp, ids_tmp_comp, label = vd_label_comp, ls = '--')
            k = k + 1

        plt.xlabel('vg')
        plt.ylabel('id')
        plt.legend(loc='best', ncol = 6)
        plt.show()

def plot_log_Id_vs_Vd_at_Vg (vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None):

    vd_count = 0

    vg_tmp = vg
    vg_tmp.sort()
    while (vg_tmp[vd_count+1] == vg_tmp[vd_count]):
        vd_count = vd_count +1

    vd_count = vd_count + 1
    vg_count = len(vg)/vd_count

    i = 0
    while (i < vg_count):
        vd_tmp = vd[i * vd_count: i * vd_count + vd_count]
        ids_tmp = np.abs(ids[i * vd_count: i * vd_count + vd_count])
        vg_label = 'vg=' + str(vg[i * vd_count])
        plt.plot(vd_tmp, ids_tmp, label=vg_label)
        i = i + 1

    if (vg_comp == None):
        plt.xlabel('vd')
        plt.ylabel('id')
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.show()

    else:
        j = 0
        while (j < vg_count):
            vd_comp_tmp = vd_comp[j * vd_count: j * vd_count + vd_count]
            ids_comp_tmp = np.abs(ids_comp[j * vd_count: j * vd_count + vd_count])
            vg_comp_label = 'vg=' + str(vg[j * vd_count])
            plt.plot(vd_comp_tmp, ids_comp_tmp, label=vg_comp_label, ls='--')
            j = j + 1

        plt.xlabel('vd')
        plt.ylabel('id')
        plt.yscale('log')
        plt.legend(loc='best', ncol=2)
        plt.show()

def plot_log_Id_vs_Vg_at_Vd (vg, vd, ids, vg_comp = None, vd_comp = None, ids_comp = None):

    vd_count = 0

    vg_tmp = vg
    vg_tmp.sort()
    while (vg_tmp[vd_count+1] == vg_tmp[vd_count]):
        vd_count = vd_count +1

    vd_count = vd_count + 1
    vg_count = len(vg)/vd_count

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

    k = 0
    while (k < vd_count):
        vg_tmp = sort_vg[k * vg_count: k * vg_count + vg_count]
        ids_tmp = np.abs(sort_ids[k * vg_count: k * vg_count + vg_count])
        vd_label = 'vd=' + str(sort_vd[k * vg_count])
        plt.plot(vg_tmp, ids_tmp, label=vd_label)
        k = k + 1

    if (vg_comp == None):
        plt.xlabel('vg')
        plt.ylabel('id')
        plt.yscale('log')
        plt.legend(loc='upper left', ncol=3)
        plt.show()

    else:
        sort_vg_comp = []
        sort_vd_comp = []
        sort_ids_comp = []
        i = 0
        while (i < vd_count):
            j = 0
            while (j < vg_count):
                sort_vg_comp.append(vg_comp[j * vd_count + i])
                sort_vd_comp.append(vd_comp[j * vd_count + i])
                sort_ids_comp.append(ids_comp[j * vd_count + i])
                j = j + 1
            i = i + 1

        k = 0
        while (k < vd_count):
            vg_tmp_comp = sort_vg_comp[k * vg_count: k * vg_count + vg_count]
            ids_tmp_comp = np.abs(sort_ids_comp[k * vg_count: k * vg_count + vg_count])
            vd_label_comp = 'vd=' + str(sort_vd_comp[k * vg_count])
            plt.plot(vg_tmp_comp, ids_tmp_comp, label=vd_label_comp, ls='--')
            k = k + 1

        plt.xlabel('vg')
        plt.ylabel('id')
        plt.yscale('log')
        plt.legend(loc='best', ncol=6)
        plt.show()