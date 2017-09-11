#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:16:48 2017

@author: jashansinghal
"""

from numpy import *
import scipy.linalg
from parser import ac_s_input


def deembed(file_name,Lg,Ld,Rg,Rd):
    s11arr,s12arr,s21arr,s22arr,freq = ac_s_input(file_name)
    omega = 2*3.14*float(freq)
    yfinal = []

    for i in range (0, size(s11arr)):
        s11 = s11arr[i]
        s12 = s12arr[i]
        s21 = s21arr[i]
        s22 = s22arr[i]
        s = array([[s11,s12],[s21,s22]])
        id = array([[1,0],[0,1]])
        z = (id+s).dot(linalg.inv(id-s))
        znew = array([
            [z[0,0]-1j*omega*Lg-Rg, z[0,1]],
            [z[1,0],z[1,1]-1j*omega*Ld-Rd]])

        yfinal.append(linalg.inv(znew))


    
    return yfinal

if __name__ == '__main__':
    Lg=1
    Ld=1
    Rg=1
    Rd=1
    
    print(deembed('./HEMT_bo/s_at_f_vs_Vd.mdm',Lg,Ld,Rg,Rd))
    





