#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:16:48 2017

@author: jashansinghal
"""

from numpy import *
import scipy.linalg
from parser import ac_s_input


def deembed(file_name,lg,ld,rg,rd):
    s11arr,s12arr,s21arr,s22arr,freq,vg,vd,id = ac_s_input(file_name)
    yfinal = []
    y11re = []
    y12re = []
    y21re = []
    y22re = []
    y11im = []
    y12im = []
    y21im = []
    y22im = []
    qdinput = []
    qdqginput = []

    for i in range (0, size(s11arr)):
        s11 = s11arr[i]
        s12 = s12arr[i]
        s21 = s21arr[i]
        s22 = s22arr[i]
        omega = 2*3.14*freq[i]
        s = array([[s11,s12],[s21,s22]])
        imat = array([[1,0],[0,1]])
        z = (imat+s).dot(linalg.inv(imat-s))
        znew = array([
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


    # for i in range (0,size(y11im)):
    #     qdqginput.append([y21im[i],y22im[i]])
    #     qdqginput.append([y11im[i],y12im[i]])


    return idcinput,iacinput,qdinput,qginput

def listcombine(list1,list2):
    result = []
    x = list1
    y = list2
    for i in range(0,size(x)):
        result.append([x[i],y[i]])
    return result


if __name__ == '__main__':
    lg=1
    ld=1
    rg=1
    rd=1
    
    idcinput,iacinput,qdinput,qginput = (deembed('./HEMT_bo/s_at_f_vs_Vd.mdm',lg,ld,rg,rd))
    print (qdinput)
#     list1 = [1,3,4]
#     list2 = [4,6,7]

# print(listcombine(list1,list2))


