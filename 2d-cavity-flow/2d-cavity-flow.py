# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:21:04 2022

@author: CFD_Tyro
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''global parameters'''
nu=0.01
# space
N=20
left=0
right=0.1
L=right-left
h=L/N
d=h
Sf=h
Vp=Sf*h
dx_array=np.linspace(left,right,N)
dy_array=np.linspace(left,right,N)
# time
dt=0.005
end_time=0.101
dt_array=np.arange(0,end_time,dt)
# coefficient matrix
velo_u=np.zeros((N,N))
velo_v=np.zeros((N,N))
p=np.zeros((N,N))
velo_u_prev=np.zeros((N,N))
velo_v_prev=np.zeros((N,N))
p_prev=np.zeros((N,N))
HbyA_x=np.zeros((N,N))
HbyA_y=np.zeros((N,N))
A_P_vec=np.zeros((N,N))
# iterative residual
velo_iter_err_init=1e-5
velo_iter_err=velo_iter_err_init
p_velo_iter_err_init=1e-6
p_iter_err=p_velo_iter_err_init
max_iter_count_per_timestep=1000
nCorr=3 # PISO loop

'''functions'''
def update_coefficient(location,velo_u_0,velo_v_0,i,j):
    A_P=1 # A_P is denominator
    A_W=0
    A_E=0
    A_S=0
    A_N=0
    if location == "left": # left boundary
        A_P=Vp/dt+h/4*(velo_u_0[i,j]+velo_u_0[i+1,j]+velo_v_0[i,j+1]-velo_v_0[i,j-1])+5*nu*Sf/d
        A_W=0
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    elif location == "right": # right boundary
        A_P=Vp/dt+h/4*(-velo_u_0[i,j]-velo_u_0[i-1,j]+velo_v_0[i,j+1]-velo_v_0[i,j-1])+5*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=0
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    elif location == "bottom": # bottom boundary
        A_P=Vp/dt+h/4*(velo_u_0[i+1,j]-velo_u_0[i-1,j]+velo_v_0[i,j]+velo_v_0[i,j+1])+5*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=0
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    elif location == "top": # top boundary
        A_P=Vp/dt+h/4*(velo_u_0[i+1,j]-velo_u_0[i-1,j]-velo_v_0[i,j]-velo_v_0[i,j-1])+5*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=0
    elif location == "leftTop": # left top corner
        A_P=Vp/dt+h/4*(velo_u_0[i,j]+velo_u_0[i+1,j]-velo_v_0[i,j]-velo_v_0[i,j-1])+6*nu*Sf/d
        A_W=0
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=0
    elif location == "leftBottom": # left bottom corner
        A_P=Vp/dt+h/4*(velo_u_0[i,j]+velo_u_0[i+1,j]+velo_v_0[i,j]+velo_v_0[i,j+1])+6*nu*Sf/d
        A_W=0
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=0
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    elif location == "rightTop": # right top corner
        A_P=Vp/dt+h/4*(-velo_u_0[i,j]-velo_u_0[i-1,j]-velo_v_0[i,j]-velo_v_0[i,j-1])+6*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=0
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=0
    elif location == "rightBottom": # right bottom corner
        A_P=Vp/dt+h/4*(-velo_u_0[i,j]-velo_u_0[i-1,j]+velo_v_0[i,j]+velo_v_0[i,j+1])+6*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=0
        A_S=0
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    elif location == "internal": # internal field
        A_P=Vp/dt+h/4*(velo_u_0[i+1,j]-velo_u_0[i-1,j]+velo_v_0[i,j+1]-velo_v_0[i,j-1])+4*nu*Sf/d
        A_W=h/4*(velo_u_0[i,j]+velo_u_0[i-1,j])+nu*Sf/d
        A_E=-h/4*(velo_u_0[i,j]+velo_u_0[i+1,j])+nu*Sf/d
        A_S=h/4*(velo_v_0[i,j]+velo_v_0[i,j-1])+nu*Sf/d
        A_N=-h/4*(velo_v_0[i,j]+velo_v_0[i,j+1])+nu*Sf/d
    else:
        print("Error Location !")
    return A_P, A_W, A_E, A_S, A_N

def update_velo_GS():
    '''
    a=b --> change 'b' will make 'a' change too
    a=np.copy(b) --> the change of 'b' will have NO influence on 'a'
    '''
    velo_u_0=np.copy(velo_u)
    velo_v_0=np.copy(velo_v)
    # boundary first
    # left boundary
    i=0
    for j in range(1,N-1):
        res=update_coefficient("left",velo_u_prev,velo_v_prev,i,j)
        velo_u[i,j]=(res[2]*velo_u_0[i+1,j]+res[3]*velo_u_0[i,j-1]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i,j]-p[i+1,j]))/res[0]
        velo_v[i,j]=(res[2]*velo_v_0[i+1,j]+res[3]*velo_v_0[i,j-1]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j+1]))/res[0]
    # right boundary
    i=N-1
    for j in range(1,N-1):
        res=update_coefficient("right",velo_u_prev,velo_v_prev,i,j)
        velo_u[i,j]=(res[1]*velo_u_0[i-1,j]+res[3]*velo_u_0[i,j-1]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i-1,j]-p[i,j]))/res[0]
        velo_v[i,j]=(res[1]*velo_v_0[i-1,j]+res[3]*velo_v_0[i,j-1]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j+1]))/res[0]
    # bottom boundary
    j=0
    for i in range(1,N-1):
        res=update_coefficient("bottom",velo_u_prev,velo_v_prev,i,j)
        velo_u[i,j]=(res[1]*velo_u_0[i-1,j]+res[2]*velo_u_0[i+1,j]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i-1,j]-p[i+1,j]))/res[0]
        velo_v[i,j]=(res[1]*velo_v_0[i-1,j]+res[2]*velo_v_0[i+1,j]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j]-p[i,j+1]))/res[0]
    # top boundary
    j=N-1
    for i in range(1,N-1):
        res=update_coefficient("top",velo_u_prev,velo_v_prev,i,j)
        velo_u[i,j]=(res[1]*velo_u_0[i-1,j]+res[2]*velo_u_0[i+1,j]+res[3]*velo_u_0[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d+h/2*(p[i-1,j]-p[i+1,j]))/res[0]
        velo_v[i,j]=(res[1]*velo_v_0[i-1,j]+res[2]*velo_v_0[i+1,j]+res[3]*velo_v_0[i,j-1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j]))/res[0]
    # left top corner
    i=0
    j=N-1
    res=update_coefficient("leftTop",velo_u_prev,velo_v_prev,i,j)
    velo_u[i,j]=(res[2]*velo_u_0[i+1,j]+res[3]*velo_u_0[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d+h/2*(p[i,j]-p[i+1,j]))/res[0]
    velo_v[i,j]=(res[2]*velo_v_0[i+1,j]+res[3]*velo_v_0[i,j-1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j]))/res[0]
    # left bottom corner
    i=0
    j=0
    res=update_coefficient("leftBottom",velo_u_prev,velo_v_prev,i,j)
    velo_u[i,j]=(res[2]*velo_u_0[i+1,j]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i,j]-p[i+1,j]))/res[0]
    velo_v[i,j]=(res[2]*velo_v_0[i+1,j]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j]-p[i,j+1]))/res[0]
    # right top corner
    i=N-1
    j=N-1
    res=update_coefficient("rightTop",velo_u_prev,velo_v_prev,i,j)
    velo_u[i,j]=(res[1]*velo_u_0[i-1,j]+res[3]*velo_u_0[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d+h/2*(p[i-1,j]-p[i,j]))/res[0]
    velo_v[i,j]=(res[1]*velo_v_0[i-1,j]+res[3]*velo_v_0[i,j-1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j]))/res[0]
    # right bottom corner
    i=N-1
    j=0
    res=update_coefficient("rightBottom",velo_u_prev,velo_v_prev,i,j)
    velo_u[i,j]=(res[1]*velo_u_0[i-1,j]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i-1,j]-p[i,j]))/res[0]
    velo_v[i,j]=(res[1]*velo_v_0[i-1,j]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j]-p[i,j+1]))/res[0]
    # internal mesh
    for i in range(1,N-1):
        for j in range(1,N-1):
            res=update_coefficient("internal",velo_u_prev,velo_v_prev,i,j)
            velo_u[i,j]=(res[1]*velo_u[i-1,j]+res[2]*velo_u_0[i+1,j]+res[3]*velo_u[i,j-1]+res[4]*velo_u_0[i,j+1]+Vp/dt*velo_u_prev[i,j]+h/2*(p[i-1,j]-p[i+1,j]))/res[0]
            velo_v[i,j]=(res[1]*velo_v[i-1,j]+res[2]*velo_v_0[i+1,j]+res[3]*velo_v[i,j-1]+res[4]*velo_v_0[i,j+1]+Vp/dt*velo_v_prev[i,j]+h/2*(p[i,j-1]-p[i,j+1]))/res[0]
    # calc residual
    resi=0
    for i in range(N):
        for j in range(N):
            resi=resi+np.abs(velo_u[i,j]-velo_u_0[i,j])/(np.abs(velo_u[i,j])+1e-8)+np.abs(velo_v[i,j]-velo_v_0[i,j])/(np.abs(velo_v[i,j])+1e-8)
    resi=resi/N/N
    return resi

def update_HbyA_AP():
    '''
    compute coefficient matrix: HbyA and A_P
    update HbyA using the latest velo_u and velo_v
    update A_P using last step velo_u_prev and velo_v_prev
    '''
    # boundary first
    # left boundary
    i=0
    for j in range(1,N-1):
        res=update_coefficient("left",velo_u_prev,velo_v_prev,i,j)
        HbyA_x[i,j]=(res[2]*velo_u[i+1,j]+res[3]*velo_u[i,j-1]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
        HbyA_y[i,j]=(res[2]*velo_v[i+1,j]+res[3]*velo_v[i,j-1]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
        A_P_vec[i,j]=res[0]
    # right boundary
    i=N-1
    for j in range(1,N-1):
        res=update_coefficient("right",velo_u_prev,velo_v_prev,i,j)
        HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[3]*velo_u[i,j-1]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
        HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[3]*velo_v[i,j-1]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
        A_P_vec[i,j]=res[0]
    # bottom boundary
    j=0
    for i in range(1,N-1):
        res=update_coefficient("bottom",velo_u_prev,velo_v_prev,i,j)
        HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[2]*velo_u[i+1,j]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
        HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[2]*velo_v[i+1,j]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
        A_P_vec[i,j]=res[0]
    # top boundary
    j=N-1
    for i in range(1,N-1):
        res=update_coefficient("top",velo_u_prev,velo_v_prev,i,j)
        HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[2]*velo_u[i+1,j]+res[3]*velo_u[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d)/res[0]
        HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[2]*velo_v[i+1,j]+res[3]*velo_v[i,j-1]+Vp/dt*velo_v_prev[i,j])/res[0]
        A_P_vec[i,j]=res[0]
    # left top corner
    i=0
    j=N-1
    res=update_coefficient("leftTop",velo_u_prev,velo_v_prev,i,j)
    HbyA_x[i,j]=(res[2]*velo_u[i+1,j]+res[3]*velo_u[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d)/res[0]
    HbyA_y[i,j]=(res[2]*velo_v[i+1,j]+res[3]*velo_v[i,j-1]+Vp/dt*velo_v_prev[i,j])/res[0]
    A_P_vec[i,j]=res[0]
    # left bottom corner
    i=0
    j=0
    res=update_coefficient("leftBottom",velo_u_prev,velo_v_prev,i,j)
    HbyA_x[i,j]=(res[2]*velo_u[i+1,j]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
    HbyA_y[i,j]=(res[2]*velo_v[i+1,j]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
    A_P_vec[i,j]=res[0]
    # right top corner
    i=N-1
    j=N-1
    res=update_coefficient("rightTop",velo_u_prev,velo_v_prev,i,j)
    HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[3]*velo_u[i,j-1]+Vp/dt*velo_u_prev[i,j]+2*nu*Sf/d)/res[0]
    HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[3]*velo_v[i,j-1]+Vp/dt*velo_v_prev[i,j])/res[0]
    A_P_vec[i,j]=res[0]
    # right bottom corner
    i=N-1
    j=0
    res=update_coefficient("rightBottom",velo_u_prev,velo_v_prev,i,j)
    HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
    HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
    A_P_vec[i,j]=res[0]
    # internal mesh
    for i in range(1,N-1):
        for j in range(1,N-1):
            res=update_coefficient("internal",velo_u_prev,velo_v_prev,i,j)
            HbyA_x[i,j]=(res[1]*velo_u[i-1,j]+res[2]*velo_u[i+1,j]+res[3]*velo_u[i,j-1]+res[4]*velo_u[i,j+1]+Vp/dt*velo_u_prev[i,j])/res[0]
            HbyA_y[i,j]=(res[1]*velo_v[i-1,j]+res[2]*velo_v[i+1,j]+res[3]*velo_v[i,j-1]+res[4]*velo_v[i,j+1]+Vp/dt*velo_v_prev[i,j])/res[0]
            A_P_vec[i,j]=res[0]
    return

def update_pressure_GS():
    '''
    construct pressure poisson (sumPhiHbyA) equation using HbyA and A_P
    '''
    #for i in range(N):
    #    for j in range(N):
    #        p[i,j]=p[i,j]-p[0,0]
    p_0=np.copy(p)
    # boundary first
    # left boundary
    i=0
    for j in range(1,N-1):
        sumPhiHbyA=update_sumPhiHbyA("left",i,j)
        C=update_rAUf("left",A_P_vec,i,j)
        p[i,j]=((C[1]*p_0[i+1,j]+C[2]*p_0[i,j-1]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # right boundary
    i=N-1
    for j in range(1,N-1):
        sumPhiHbyA=update_sumPhiHbyA("right",i,j)
        C=update_rAUf("right",A_P_vec,i,j)
        p[i,j]=((C[0]*p_0[i-1,j]+C[2]*p_0[i,j-1]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # bottom boundary
    j=0
    for i in range(1,N-1):
        sumPhiHbyA=update_sumPhiHbyA("bottom",i,j)
        C=update_rAUf("bottom",A_P_vec,i,j)
        p[i,j]=((C[0]*p_0[i-1,j]+C[1]*p_0[i+1,j]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # top boundary
    j=N-1
    for i in range(1,N-1):
        sumPhiHbyA=update_sumPhiHbyA("top",i,j)
        C=update_rAUf("top",A_P_vec,i,j)
        p[i,j]=((C[0]*p_0[i-1,j]+C[1]*p_0[i+1,j]+C[2]*p_0[i,j-1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # left top corner
    i=0
    j=N-1
    sumPhiHbyA=update_sumPhiHbyA("leftTop",i,j)
    C=update_rAUf("leftTop",A_P_vec,i,j)
    p[i,j]=((C[1]*p_0[i+1,j]+C[2]*p_0[i,j-1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # left bottom corner
    i=0
    j=0
    sumPhiHbyA=update_sumPhiHbyA("leftBottom",i,j)
    C=update_rAUf("leftBottom",A_P_vec,i,j)
    p[i,j]=((C[1]*p_0[i+1,j]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # right top corner
    i=N-1
    j=N-1
    sumPhiHbyA=update_sumPhiHbyA("rightTop",i,j)
    C=update_rAUf("rightTop",A_P_vec,i,j)
    p[i,j]=((C[0]*p_0[i-1,j]+C[2]*p_0[i,j-1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # right bottom corner
    i=N-1
    j=0
    sumPhiHbyA=update_sumPhiHbyA("rightBottom",i,j)
    C=update_rAUf("rightBottom",A_P_vec,i,j)
    p[i,j]=((C[0]*p_0[i-1,j]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # internal mesh
    for i in range(1,N-1):
        for j in range(1,N-1):    
            sumPhiHbyA=update_sumPhiHbyA("internal",i,j)
            C=update_rAUf("internal",A_P_vec,i,j)
            p[i,j]=((C[0]*p[i-1,j]+C[1]*p_0[i+1,j]+C[2]*p[i,j-1]+C[3]*p_0[i,j+1])-d/h*sumPhiHbyA)/(C[0]+C[1]+C[2]+C[3])
    # calc residual
    resi=0
    for i in range(N):
        for j in range(N):
            resi=resi+np.abs(p[i,j]-p_0[i,j])/np.abs(p[i,j]+1e-8)
    resi=resi/N/N
    return resi

def update_rAUf(location,A_P,i,j):
    Cw=0
    Ce=0
    Cs=0
    Cn=0
    if location == "left": # left boundary
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=0
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    elif location == "right": # right boundary
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=0
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    elif location == "bottom": # bottom boundary
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=0
    elif location == "top": # top boundary
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=0
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    elif location == "leftTop": # left top corner
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=0
        Cn=0
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    elif location == "leftBottom": # left bottom corner
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=0
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=0
    elif location == "rightTop": # right top corner
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=0
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=0
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    elif location == "rightBottom": # right bottom corner
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=0
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=0
    elif location == "internal": # internal field
        if np.abs(A_P[i,j])<1e-12 or np.abs(A_P[i+1,j])<1e-12 or np.abs(A_P[i-1,j])<1e-12 or np.abs(A_P[i,j+1])<1e-12 or np.abs(A_P[i,j-1])<1e-12:
            print("A_P cannot be ZERO !",A_P[i,j],A_P[i-1,j],A_P[i+1,j],A_P[i,j-1],A_P[i,j+1])
        Ce=(1/A_P[i,j]+1/A_P[i+1,j])/2
        Cw=(1/A_P[i,j]+1/A_P[i-1,j])/2
        Cn=(1/A_P[i,j]+1/A_P[i,j+1])/2
        Cs=(1/A_P[i,j]+1/A_P[i,j-1])/2
    else:
        print("Error Location !")
    return Cw,Ce,Cs,Cn

def update_sumPhiHbyA(location,i,j):
    if location == "left": # left boundary
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=0
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    elif location == "right": # right boundary
        HbyA_x_e=0
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    elif location == "bottom": # bottom boundary
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=0
    elif location == "top": # top boundary
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=0
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    elif location == "leftTop": # left top corner
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=0
        HbyA_y_n=0
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    elif location == "leftBottom": # left bottom corner
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=0
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=0
    elif location == "rightTop": # right top corner
        HbyA_x_e=0
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=0
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    elif location == "rightBottom": # right bottom corner
        HbyA_x_e=0
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=0
    elif location == "internal": # internal field
        HbyA_x_e=(HbyA_x[i,j]+HbyA_x[i+1,j])/2
        HbyA_x_w=(HbyA_x[i,j]+HbyA_x[i-1,j])/2
        HbyA_y_n=(HbyA_y[i,j]+HbyA_y[i,j+1])/2
        HbyA_y_s=(HbyA_y[i,j]+HbyA_y[i,j-1])/2
    else:
        print("Error Location !")
    return h*(HbyA_x_e-HbyA_x_w+HbyA_y_n-HbyA_y_s)/Vp

def update_continuity_error():
    res=0
    # boundary first
    # left boundary
    i=0
    for j in range(1,N-1):
        res+=h/2*(velo_u[i,j]+velo_u[i+1,j]+velo_v[i,j+1]-velo_v[i,j-1])
    # right boundary
    i=N-1
    for j in range(1,N-1):
        res+=h/2*(-velo_u[i,j]-velo_u[i-1,j]+velo_v[i,j+1]-velo_v[i,j-1])
    # bottom boundary
    j=0
    for i in range(1,N-1):
        res+=h/2*(velo_u[i+1,j]-velo_u[i-1,j]+velo_v[i,j]+velo_v[i,j+1])
    # top boundary
    j=N-1
    for i in range(1,N-1):
        res+=h/2*(velo_u[i+1,j]-velo_u[i-1,j]-velo_v[i,j]-velo_v[i,j-1])
    # left top corner
    i=0
    j=N-1
    res+=h/2*(velo_u[i,j]+velo_u[i+1,j]-velo_v[i,j]-velo_v[i,j-1])
    # left bottom corner
    i=0
    j=0
    res+=h/2*(velo_u[i,j]+velo_u[i+1,j]+velo_v[i,j]+velo_v[i,j+1])
    # right top corner
    i=N-1
    j=N-1
    res+=h/2*(-velo_u[i,j]-velo_u[i-1,j]-velo_v[i,j]-velo_v[i,j-1])
    # right bottom corner
    i=N-1
    j=0
    res+=h/2*(-velo_u[i,j]-velo_u[i-1,j]+velo_v[i,j]+velo_v[i,j+1])
    # internal mesh
    for i in range(1,N-1):
        for j in range(1,N-1):    
            res+=h/2*((velo_u[i,j]+velo_u[i+1,j])-(velo_u[i,j]+velo_u[i-1,j])+(velo_v[i,j]+velo_v[i,j+1])-(velo_v[i,j]+velo_v[i,j-1]))
    return res

def one_step(step):
    # 1. update velocity
    iter_count=0
    resi=1
    while 1:
        iter_count+=1
        resi=update_velo_GS()
        #print(iter_count,resi)
        if resi<velo_iter_err or iter_count>max_iter_count_per_timestep:
            break
    print("Update velocity done. Iteration number %d. Residual %lf" % (iter_count,resi))
    con_err=update_continuity_error()
    print("Continuity error %g" % con_err)
    # 2. update pressure
    for i in range(nCorr):
        iter_count=0
        resi=1
        update_HbyA_AP()
        while 1:
            iter_count+=1
            resi=update_pressure_GS()
            #print(iter_count,resi)
            if resi<p_iter_err or iter_count>max_iter_count_per_timestep:
                break
        print("PISO Loop %d. Update pressure done. Iteration number %d. Residual %lf" % (i,iter_count,resi))
        # 3. update velocity with new pressure
        update_velo_final()
        con_err=update_continuity_error()
        print("PISO Loop %d. Continuity error %g" % (i,con_err))
    return

def update_velo_final():
    # boundary
    # left boundary
    i=0
    for j in range(1,N-1):
        velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i,j])
        velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j-1])
    # right boundary
    i=N-1
    for j in range(1,N-1):
        velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i-1,j])
        velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j-1])
    # bottom boundary
    j=0
    for i in range(1,N-1):
        velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i-1,j])
        velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j])
    # top boundary
    j=N-1
    for i in range(1,N-1):
        velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i-1,j])
        velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i,j-1])
    # left top corner
    i=0
    j=N-1
    velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i,j])
    velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i,j-1])
    # left bottom corner
    i=0
    j=0
    velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i,j])
    velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j])
    # right top corner
    i=N-1
    j=N-1
    velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i-1,j])
    velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i,j-1])
    # right bottom corner
    i=N-1
    j=0
    velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j]-p[i-1,j])
    velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j])
    # internal mesh
    for i in range(1,N-1):
        for j in range(1,N-1):
            velo_u[i,j]=HbyA_x[i,j]-1/A_P_vec[i,j]*h/2*(p[i+1,j]-p[i-1,j])
            velo_v[i,j]=HbyA_y[i,j]-1/A_P_vec[i,j]*h/2*(p[i,j+1]-p[i,j-1])
    return

def plot_result():
    plt.figure()
    sns.heatmap(velo_u.transpose(),cmap="RdBu_r").invert_yaxis()
    plt.title("Velocity-u")
    plt.show()
    plt.figure()
    sns.heatmap(velo_v.transpose(),cmap="RdBu_r").invert_yaxis()
    plt.title("Velocity-v")
    plt.show()
    plt.figure()
    sns.heatmap(p.transpose(),cmap="RdBu_r").invert_yaxis()
    plt.title("Pressure")
    plt.show()
    return

"""main function"""
if __name__ == '__main__':
    for i in range(1,len(dt_array)):
    #for i in range(1,20):
        print("Step %d. Time %lf s" % (i,dt_array[i]))
        velo_u_prev=np.copy(velo_u)
        velo_v_prev=np.copy(velo_v)
        p_prev=np.copy(p)
        one_step(i)
        print(" ")
    plot_result()
    # compare result with OpenFOAM
    # 1. upper boundary
    ofdata=np.loadtxt("0.1-u-ico-upper.txt",dtype=float)
    plt.figure()
    plt.scatter(dx_array,velo_u[:,N-1],label="This work")
    plt.scatter(dx_array,ofdata,label="icoFoam")
    plt.legend()
    plt.show()
    # 2. middle cells
    ofdata=np.loadtxt("0.1-u-ico-middle.txt",dtype=float)
    plt.figure()
    plt.scatter(dy_array,velo_u[10,:],label="This work")
    plt.scatter(dy_array,ofdata,label="icoFoam")
    plt.legend()
    plt.show()
    
        