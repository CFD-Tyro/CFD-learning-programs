# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:57:02 2022

@author: CFD_Tyro
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


'''global parameters'''
# space
N=101
left=-1
right=1
L=right-left
h=L/(N-1)
dx_array=np.linspace(left,right,N)
dy_array=np.linspace(left,right,N)
#print(dx_array)
# iterative method: Jacobi, Gauss-Seidel, SOR
iter_method="SOR"
err=1e-5

'''functions'''
def set_init_boundary():
    f_p0=np.zeros((N,N))
    for i in range(N):
        f_p0[0,i]=np.cos(-1-dy_array[i])*np.exp(-1-dy_array[i]) # left border
        f_p0[N-1,i]=np.cos(1-dy_array[i])*np.exp(1-dy_array[i]) # right border
        f_p0[i,0]=np.cos(dx_array[i]+1)*np.exp(dx_array[i]+1) # bottom
        f_p0[i,N-1]=np.cos(dx_array[i]-1)*np.exp(dx_array[i]-1) # top
    #print(f_p0)
    return f_p0

def set_source():
    S_p=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            S_p[i,j]=-4*np.sin(dx_array[i]-dy_array[j])*np.exp(dx_array[i]-dy_array[j])
    #print(S_p)
    return S_p

def update_Jacobi(f_p0,S_p):
    Niter=0
    while 1:
        f_p=np.copy(f_p0)
        for i in range(1,N-1):
            for j in range(1,N-1):
                f_p[i,j]=0.25*(f_p0[i-1,j]+f_p0[i,j-1]+f_p0[i+1,j]+f_p0[i,j+1]-h*h*S_p[i,j])
        Niter=Niter+1
        # compare f_p and f_p0
        resi=0
        for i in range(1,N-1):
            for j in range(1,N-1):
                resi=resi+np.abs(f_p[i,j]-f_p0[i,j])/np.abs(f_p[i,j]+1e-8)
        resi=resi/(N-1)/(N-1)
        f_p0=f_p
        #print("Niter =",Niter,"resi =",resi)
        if resi<err:
            break
    # plot heatmap
    plt.figure()
    sns.heatmap(f_p,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter

def update_GS(f_p0,S_p):
    Niter=0
    while 1:
        f_p=np.copy(f_p0)
        for i in range(1,N-1):
            for j in range(1,N-1):
                f_p[i,j]=0.25*(f_p[i-1,j]+f_p[i,j-1]+f_p0[i+1,j]+f_p0[i,j+1]-h*h*S_p[i,j])
        Niter=Niter+1
        # compare f_p and f_p0
        resi=0
        for i in range(1,N-1):
            for j in range(1,N-1):
                resi=resi+np.abs(f_p[i,j]-f_p0[i,j])/np.abs(f_p[i,j]+1e-8)
        resi=resi/(N-1)/(N-1)
        f_p0=f_p
        #print("Niter =",Niter,"resi =",resi)
        if resi<err:
            break
    # plot heatmap
    plt.figure()
    sns.heatmap(f_p,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter

def update_SOR(f_p0,S_p,beta):
    Niter=0
    while 1:
        f_p=np.copy(f_p0)
        for i in range(1,N-1):
            for j in range(1,N-1):
                f_p[i,j]=beta/4*(f_p[i-1,j]+f_p[i,j-1]+f_p0[i+1,j]+f_p0[i,j+1]-h*h*S_p[i,j])+(1-beta)*f_p0[i,j]
        Niter=Niter+1
        # compare f_p and f_p0
        resi=0
        for i in range(1,N-1):
            for j in range(1,N-1):
                resi=resi+np.abs(f_p[i,j]-f_p0[i,j])/np.abs(f_p[i,j]+1e-8)
        resi=resi/(N-1)/(N-1)
        f_p0=f_p
        #print("Niter =",Niter,"resi =",resi)
        if resi<err:
            break
    # plot heatmap
    plt.figure()
    sns.heatmap(f_p,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter


"""main function"""
if __name__ == '__main__':
    f_p0=set_init_boundary()
    S_p=set_source()
    if iter_method == "Jacobi":
        Niter=update_Jacobi(f_p0,S_p)
        print("Jacobi iteration number",Niter," N =",N," residual =",err)
    elif iter_method == "GS":
        Niter=update_GS(f_p0,S_p)
        print("Gauss-Seidel iteration number",Niter," N =",N," residual =",err)
    elif iter_method == "SOR":
        sor_beta=1.9
        Niter=update_SOR(f_p0,S_p,sor_beta)
        print("SOR iteration number",Niter," N =",N," residual =",err," beta =",sor_beta)
    else:
        print("Choose right iterative method")
        exit(-1)
    # analytical solution
    solu=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            solu[i,j]=np.cos(dx_array[i]-dy_array[j])*np.exp(dx_array[i]-dy_array[j])
    plt.figure()
    sns.heatmap(solu,cmap="RdBu_r").invert_yaxis()
    plt.show()
    
