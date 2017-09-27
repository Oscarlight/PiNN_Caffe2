import numpy as np
import scipy.linalg as linalg

def deembed(read_data_fun,file_name,lg,ld,rg,rd):
    s11arr,s12arr,s21arr,s22arr,freq,vg,vd,id = read_data_fun(file_name)
    yfinal = []
    y11re = []
    y12re = []
    y21re = []
    y22re = []
    y11im = []
    y12im = []
    y21im = []
    y22im = []

    for i in range (0, np.size(s11arr)):
        s11 = s11arr[i]
        s12 = s12arr[i]
        s21 = s21arr[i]
        s22 = s22arr[i]
        omega = 2*3.14*freq[i]
        s = np.array([[s11,s12],[s21,s22]])
        imat = np.array([[1,0],[0,1]])
        z = (imat+s).dot(linalg.inv(imat-s))
        znew = np.array([
            [z[0,0]-1j*omega*lg-rg, z[0,1]],
            [z[1,0],z[1,1]-1j*omega*ld-rd]])

        yfinal.append(linalg.inv(znew))
        y = linalg.inv(znew)

        y11re.append(y[0][0].real)
        y12re.append(y[0][1].real)
        y21re.append(y[1][0].real)
        y22re.append(y[1][1].real)

        y11im.append((y[0][0].imag)/omega)
        y12im.append((y[0][1].imag)/omega)
        y21im.append((y[1][0].imag)/omega)
        y22im.append((y[1][1].imag)/omega)

    idcinput = listcombine(y11re,y12re)
    iacinput = listcombine(y21re,y22re)
    qdinput = listcombine(y21im,y22im)
    qginput = listcombine(y11im,y12im)

    return vg,vd,id,idcinput,iacinput,qdinput,qginput

def listcombine(list1,list2):
    result = []
    x = list1
    y = list2
    for i in range(0, np.size(x)):
        result.append([x[i],y[i]])
    return result


if __name__ == '__main__':
    pass


