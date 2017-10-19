import numpy as np
import scipy.linalg as linalg

def deembed(
    read_data_fun,file_name,
    lg=0,ld=0,rg=0,rd=0):

    s11arr,s12arr,s21arr,s22arr,freq,vg,vd,ids = read_data_fun(file_name)

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
    qdinput = listcombine(y21im,y22im) # [dQd/dVg, dQd/dVd]
    qginput = listcombine(y11im,y12im) # [dQg/dVg, dQg/dVd]

    return vg,vd,ids,idcinput,iacinput,qdinput,qginput

def listcombine(list1,list2):
    return np.concatenate(
        (np.expand_dims(list1, axis=1),
         np.expand_dims(list2, axis=1)),
         axis=1
        )


if __name__ == '__main__':
    pass

