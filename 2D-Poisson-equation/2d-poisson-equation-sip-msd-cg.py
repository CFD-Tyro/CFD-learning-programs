# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:16:28 2022

@author: CFD_Tyro
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


'''global parameters'''
# space
N=51
left=-1
right=1
L=right-left
h=L/(N-1)
dx_array=np.linspace(left,right,N)
dy_array=np.linspace(left,right,N)
#print(dx_array)
# iterative method: Jacobi, Gauss-Seidel, SOR, SIP, MSD, CG
iter_method="CG"
err=1e-2

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

def matrix_product(A,f_p0):
    Q=np.zeros(f_p0.shape)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            Q[i]=Q[i]+A[i,j]*f_p0[j]
    return Q

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

def update_sip(alpha):
    # 1. set boundary
    top_border=np.cos(dx_array-1)*np.exp(dx_array-1)
    bottom_border=np.cos(dx_array+1)*np.exp(dx_array+1)
    left_border=np.cos(-1-dy_array)*np.exp(-1-dy_array)
    right_border=np.cos(1-dy_array)*np.exp(1-dy_array)
    # 2. construct coefficient matrix A: AX=Q
    # 4x_p = x_w + x_e + x_n + x_s - h*h*f_p
    # 4x_{i} = x_{i-1} + x_{i+1} + x_{i+N} + x_{i-N} - h*h*f_{i}
    A=np.zeros((N*N,N*N))
    Q=np.zeros(N*N)
    for k in range(N*N):
        # boundary first, internal second
        if k<N: # bottom border
            A[k,k]=1
            i=k%N
            Q[k]=bottom_border[i]
        elif k>=N*N-N: # top border
            A[k,k]=1
            i=k%N
            Q[k]=top_border[i]
        elif k%N==0: # left border
            A[k,k]=1
            j=int(k/N)
            Q[k]=left_border[j]
        elif (k+1)%N==0: # right border
            A[k,k]=1
            j=int(k/N)
            Q[k]=right_border[j]
        else: # internal
            A[k,k]=4 # P
            A[k,k-N]=-1 # S
            A[k,k+N]=-1 # N
            A[k,k-1]=-1 # W
            A[k,k+1]=-1 # E
            i=k%N
            j=int(k/N)
            Q[k]=-h*h*(-4*np.sin(dx_array[i]-dy_array[j])*np.exp(dx_array[i]-dy_array[j]))
    # 3. construct B,D,E,F,H from A
    B=np.zeros(N*N)
    D=np.zeros(N*N)
    E=np.zeros(N*N)
    F=np.zeros(N*N)
    H=np.zeros(N*N)
    for k in range(N*N):
        E[k]=A[k,k]
        if k-1>=0:
            D[k]=A[k,k-1]
        if k-N>=0:
            B[k]=A[k,k-N]
        if k+1<N*N:
            F[k]=A[k,k+1]
        if k+N<N*N:
            H[k]=A[k,k+N]
    # 4. construct L(b,c,d),U(e,f)
    b=np.zeros(N*N)
    c=np.zeros(N*N)
    d=np.zeros(N*N)
    e=np.zeros(N*N)
    f=np.zeros(N*N)
    d[0]=E[0]
    e[0]=F[0]/d[0]
    f[0]=H[0]/d[0]
    for k in range(1,N*N):
        if k>N:
            b[k]=B[k]/(1+alpha*e[k-N])
        c[k]=D[k]/(1+alpha*f[k-1])
        if k>N:
            d[k]=E[k]+alpha*(b[k]*e[k-N]+c[k]*f[k-1])-b[k]*f[k-N]-c[k]*e[k-1]
        else:
            d[k]=E[k]+alpha*(c[k]*f[k-1])-c[k]*e[k-1]
        f[k]=(H[k]-alpha*c[k]*f[k-1])/d[k]
        if k>N:
            e[k]=(F[k]-alpha*b[k]*e[k-N])/d[k]
        else:
            e[k]=F[k]/d[k]
    # 5. compute R^n iteratively
    X=Q # set initial values
    Niter=0
    while 1:
        # compute residual: sqrt(sum[(Q-AX^n)*(Q-AX^n)])
        prod=matrix_product(A,X)
        R=Q-prod
        resi2=np.sqrt(sum(R*R))
        print("Niter=",Niter,"resi2=",resi2)
        if resi2<err:
            break
        # compute Y: LY=R
        Y=np.zeros(R.shape)
        Y[0]=R[0]/d[0]
        for k in range(1,Y.shape[0]):
            if k>N:
                Y[k]=(R[k]-c[k]*Y[k-1]-b[k]*Y[k-N])/d[k]
            else:
                Y[k]=(R[k]-c[k]*Y[k-1])/d[k]
        # compute delta: U delta = Y
        delta=np.zeros(R.shape)
        delta[-1]=Y[-1]
        for k in range(Y.shape[0]-2,-1,-1):
            #print(k)
            if k<=N:
                delta[k]=Y[k]-e[k]*delta[k+1]-f[k]*delta[k+N]
            else:
                delta[k]=Y[k]-e[k]*delta[k+1]
        X=X+delta
        Niter+=1
    # 6. output result, turn 1d array to 2d
    solu=np.zeros((N,N))
    for k in range(N*N):
        i=k%N
        j=int(k/N)
        solu[i,j]=X[k]
    plt.figure()
    sns.heatmap(solu,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter

def update_msd():
    # 1. set boundary
    top_border=np.cos(dx_array-1)*np.exp(dx_array-1)
    bottom_border=np.cos(dx_array+1)*np.exp(dx_array+1)
    left_border=np.cos(-1-dy_array)*np.exp(-1-dy_array)
    right_border=np.cos(1-dy_array)*np.exp(1-dy_array)
    # 2. construct coefficient matrix A: AX=Q
    # 4x_p = x_w + x_e + x_n + x_s - h*h*f_p
    # 4x_{i} = x_{i-1} + x_{i+1} + x_{i+N} + x_{i-N} - h*h*f_{i}
    A=np.zeros((N*N,N*N))
    Q=np.zeros(N*N)
    for k in range(N*N):
        # boundary first, internal second
        if k<N: # bottom border
            A[k,k]=1
            i=k%N
            Q[k]=bottom_border[i]
        elif k>=N*N-N: # top border
            A[k,k]=1
            i=k%N
            Q[k]=top_border[i]
        elif k%N==0: # left border
            A[k,k]=1
            j=int(k/N)
            Q[k]=left_border[j]
        elif (k+1)%N==0: # right border
            A[k,k]=1
            j=int(k/N)
            Q[k]=right_border[j]
        else: # internal
            A[k,k]=4 # P
            A[k,k-N]=-1 # S
            A[k,k+N]=-1 # N
            A[k,k-1]=-1 # W
            A[k,k+1]=-1 # E
            i=k%N
            j=int(k/N)
            Q[k]=-h*h*(-4*np.sin(dx_array[i]-dy_array[j])*np.exp(dx_array[i]-dy_array[j]))
    # 3. compute R^n iteratively
    X=Q # set initial values
    Niter=0
    while 1:
        # compute residual: sqrt(sum[(Q-AX^n)*(Q-AX^n)])
        prod=matrix_product(A,X)
        R=Q-prod
        resi2=np.sqrt(sum(R*R))
        print("Niter=",Niter,"resi2=",resi2)
        if resi2<err:
            break
        # compute alpha
        numerator=sum(R*R)
        denominator=sum(R*matrix_product(A, R))
        alpha=numerator/denominator
        X=X+alpha*R
        Niter+=1
    # 4. output result, turn 1d array to 2d
    solu=np.zeros((N,N))
    for k in range(N*N):
        i=k%N
        j=int(k/N)
        solu[i,j]=X[k]
    plt.figure()
    sns.heatmap(solu,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter

def update_cg():
    # 1. set boundary
    top_border=np.cos(dx_array-1)*np.exp(dx_array-1)
    bottom_border=np.cos(dx_array+1)*np.exp(dx_array+1)
    left_border=np.cos(-1-dy_array)*np.exp(-1-dy_array)
    right_border=np.cos(1-dy_array)*np.exp(1-dy_array)
    # 2. construct coefficient matrix A: AX=Q
    # 4x_p = x_w + x_e + x_n + x_s - h*h*f_p
    # 4x_{i} = x_{i-1} + x_{i+1} + x_{i+N} + x_{i-N} - h*h*f_{i}
    A=np.zeros((N*N,N*N))
    Q=np.zeros(N*N)
    for k in range(N*N):
        # boundary first, internal second
        if k<N: # bottom border
            A[k,k]=1
            i=k%N
            Q[k]=bottom_border[i]
        elif k>=N*N-N: # top border
            A[k,k]=1
            i=k%N
            Q[k]=top_border[i]
        elif k%N==0: # left border
            A[k,k]=1
            j=int(k/N)
            Q[k]=left_border[j]
        elif (k+1)%N==0: # right border
            A[k,k]=1
            j=int(k/N)
            Q[k]=right_border[j]
        else: # internal
            A[k,k]=4 # P
            A[k,k-N]=-1 # S
            A[k,k+N]=-1 # N
            A[k,k-1]=-1 # W
            A[k,k+1]=-1 # E
            i=k%N
            j=int(k/N)
            Q[k]=-h*h*(-4*np.sin(dx_array[i]-dy_array[j])*np.exp(dx_array[i]-dy_array[j]))
    # 3. compute R^n iteratively
    X=Q # set initial values
    D=np.zeros(Q.shape)
    # compute initial residual
    prod=matrix_product(A,X)
    R0=Q-prod
    D=R0
    Niter=0
    while 1:
        # compute alpha using R0 and D
        numerator=sum(R0*R0)
        denominator=sum(D*matrix_product(A, D))
        alpha=numerator/denominator
        X=X+alpha*D
        # compute new residual
        R1=Q-matrix_product(A,X)
        resi2=np.sqrt(sum(R1*R1))
        print("Niter=",Niter,"resi2=",resi2)
        if resi2<err:
            break
        # compute beta
        beta=sum(R1*R1)/sum(R0*R0)
        # update D
        D=R1+beta*D
        # update R0
        R0=R1
        Niter+=1
    # 4. output result, turn 1d array to 2d
    solu=np.zeros((N,N))
    for k in range(N*N):
        i=k%N
        j=int(k/N)
        solu[i,j]=X[k]
    plt.figure()
    sns.heatmap(solu,cmap="RdBu_r").invert_yaxis()
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
    elif iter_method == "SIP":
        alpha=0.9
        Niter=update_sip(alpha)
        print("SIP iteration number",Niter," N =",N," residual =",err)
    elif iter_method == "MSD":
        Niter=update_msd()
        print("MSD iteration number",Niter," N =",N," residual =",err)
    elif iter_method == "CG":
        Niter=update_cg()
        print("CG iteration number",Niter," N =",N," residual =",err)
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