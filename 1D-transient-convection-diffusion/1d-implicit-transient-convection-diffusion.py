# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:59:42 2022

@author: CFD_Tyro
"""


import matplotlib.pyplot as plt
import numpy as np


'''global parameters'''
discretization="outer"
# space
N=1001
L=9
dx=L/(N-1)
# time
dt=0.001
dt_array=np.arange(0,2.5,dt)
# constant
U=1
D=0.05

'''functions'''
def set_x_range(discre_way):
    if discre_way == "inner":
        dx_array=np.arange(0+dx/2,L,dx) # size=N-1
    elif discre_way == "outer":
        dx_array=np.linspace(0,L,N) # size=N
    else:
        print("wrong discretization")
        exit(-1)
    print("dx_array[1]-dx_array[0] =",dx_array[1]-dx_array[0],", dx =",dx)
    return dx_array

def init_array(dx_array):
    f_p0=np.exp(-(dx_array-1)*(dx_array-1)/D)
    return f_p0

def update(discre_way,dx_array,f_p0,t):
    left_border=1/np.sqrt(4*t+1)*np.exp(-pow(1+U*t,2)/D/(4*t+1))
    right_border=1/np.sqrt(4*t+1)*np.exp(-pow(8-U*t,2)/D/(4*t+1))
    size=dx_array.shape[0]
    a_P=1+U*dt/dx+2*D*dt/dx/dx
    a_W=U*dt/dx+D*dt/dx/dx
    a_E=dt*D/dx/dx
    # update P & Q
    P=np.zeros(size)
    Q=np.zeros(size)
    P[0]=a_E/a_P
    Q[0]=f_p0[0]/a_P
    for i in range(1,size):
        P[i]=a_E/(a_P-a_W*P[i-1])
        Q[i]=(f_p0[i]+a_W*Q[i-1])/(a_P-a_W*P[i-1])
    # update f_p
    f_p=np.zeros(size)
    # two border
    f_p[0]=left_border
    f_p[size-1]=right_border
    #print(Q[size-1],f_p[size-1])
    for i in range(size-1,0,-1):
        f_p[i-1]=P[i-1]*f_p[i]+Q[i-1]
    return f_p


"""main function"""
if __name__ == '__main__':
    dx_array=set_x_range(discretization)
    f_p0=init_array(dx_array)
    #plt.figure(figsize=(8,6))
    #plt.plot(dx_array,f_p0)
    #plt.show()
    for i in dt_array:
        f_p=update(discretization,dx_array,f_p0,i)
        f_p0=f_p
    # analytical solution
    t=2.5
    f_p_ana=1/np.sqrt(4*t+1)*np.exp(-pow((dx_array-1-U*t),2)/D/(4*t+1))
    plt.figure(figsize=(8,6))
    plt.plot(dx_array,f_p0,label="simulation")
    plt.plot(dx_array,f_p_ana,label="analytical")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f")
    plt.xlim(0,9)
    plt.show()