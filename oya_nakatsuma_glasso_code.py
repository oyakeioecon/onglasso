#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
"""
Sakae Oya and Teruo Nakatsuma, 2020  
Please run this file and input simulation conditions.
"""

#%% import packages 
from tqdm import trange
import numpy as np
import scipy.linalg as sl
import numpy.random as rnd
import scipy.stats as st
from sklearn.metrics import confusion_matrix
import time
#import seaborn as sns
#import matplotlib.pyplot as plt

#%% define functions
# compute partitions for the columewise gibbs sampling 
def exclude_rowcol(A, ind1, ind2):
    if ind1[0] == 'False':
        return A[1:,1:]
    elif ind1[-1] == 'False':
        return A[:-1,:-1]
    else:
        return A[ind1, :][:, ind2]

# update off-diagonal elements 
def update_offdiagonals(A, x, ind, i):
    B = A.copy()
    if ind[0] == 'False':
        B[0, i] = x
        B[i, 0] = x
    elif ind[-1]=='False':
        B[-1,i] = x
        B[i,-1] = x
    else:    
        B[ind, i] = x
        B[i, ind] = x
    return B

# compute the quadratic form
def quad_form(A, x):
    return x @ A @ x

# generate random variables from invgauss distribution
# We copied Wang(2012)'s rand_ig.m code and rewrite it in python.
def rand_ig(theta, chi, sampleSize):
     chisq1 = rnd.randn(sampleSize)**2
     y = theta + 0.5*theta/chi*(theta*chisq1 - np.sqrt(4*theta*chi*chisq1 + theta**2*chisq1**2))
     l = rnd.rand(sampleSize)>= theta/(theta+y)   
     y[l] = theta[l]**2/y[l]
     return y  

# assessment for covariance matrix estimation |omega_ij|<10**-3
def assess_cve(post_Omega, True_Omega):
    hat_Sigma = sl.inv(post_Omega)
    sigome = np.dot(hat_Sigma,True_Omega)
    Stein_Loss = np.trace(sigome)- np.log(sl.det(sigome)) - p
    F_norm = sl.norm((post_Omega - True_Omega),ord=None)
    result = ['F_norm',F_norm, 'Stein',Stein_Loss]
    print(result)
    
# assessment for graphical structure learning based on |omega_ij|<10**-3
def assess_gsl(post_Omega, True_Omega):
    aaa = np.zeros((p,p),dtype=float)
    bbb = np.zeros((p,p),dtype=float)
    for k in range(p):
        for v in range(p):
            if np.abs(True_Omega[k,v])<0.001:
                aaa[k,v] = 0.0
            else:
                aaa[k,v] = 1.0               
    for k in range(p):
        for v in range(p):
            if np.abs(post_Omega[k,v])<0.001:
                bbb[k,v] = 0.0 
            else:
                bbb[k,v] = 1.0
    omega_true = np.ravel(aaa)
    omega_estimate = np.ravel(bbb)
    TN, FP, FN, TP = confusion_matrix(omega_true,omega_estimate).ravel()
    Specificity = TN / (TN+FP)
    Sensitivity = TP / (TP+FN)
    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    result = ['Specificity',Specificity*100,'Sensitivity', Sensitivity*100,'MCC', MCC*100]
    print('performance of graphical structure learning based on |omega_ij|<10**-3')
    print(result)

# assessment for graphical structure learning based on omega_ij<10**-3
def assess_gsl_no_absolute_value(post_Omega, True_Omega):
    aaa = np.zeros((p,p),dtype=float)
    bbb = np.zeros((p,p),dtype=float)
    for k in range(p):
        for v in range(p):
            if True_Omega[k,v]<0.001:
                aaa[k,v] = 0.0
            else:
                aaa[k,v] = 1.0               
    for k in range(p):
        for v in range(p):
            if post_Omega[k,v]<0.001:
                bbb[k,v] = 0.0 
            else:
                bbb[k,v] = 1.0
    omega_true = np.ravel(aaa)
    omega_estimate = np.ravel(bbb)
    TN, FP, FN, TP = confusion_matrix(omega_true,omega_estimate).ravel()
    Specificity = TN / (TN+FP)
    Sensitivity = TP / (TP+FN)
    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    result = ['Specificity',Specificity*100,'Sensitivity', Sensitivity*100,'MCC', MCC*100]
    print('performance of graphical structure learning based on omega_ij<10**-3')
    print(result)
    
#  count number of times that positive definite of Omega breaks after update of off-diagonal elements
def detcount(runsdet):
    count = 0
    for ccc in range(runsdet.size):
        if runsdet[ccc] == 0.0:
            count = count + 1
    print('Positive Definiteness of Omega breaks',count, 'times out of',
          runsdet.size, '. This is',(count/runsdet.size)*100, '% of sample size.') 
    
# data generation for simutlation section
def data_generation(n,p,model):
    if model == 'AR(1)':
        Sigma = np.zeros((p,p),dtype=float) 
        for kk in range(p):
            for vv in range(p):
                if kk>vv:
                    Sigma[kk,vv]=0.7**(kk-vv)
                    Sigma[vv,kk]=0.7**(kk-vv)
                elif kk<vv:
                    Sigma[kk,vv]=0.7**(vv-kk)
                    Sigma[vv,kk]=0.7**(vv-kk)
                else:
                    Sigma[kk,vv]=0.7**0   
                    Sigma[vv,kk]=0.7**0
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)    
        True_Omega = sl.inv(Sigma)  
        return(Y, True_Omega)
       
    elif model == 'AR(2)':
        temp_Omega = np.zeros((p,p),dtype=float)
        for kk in range(p):
            if 1<=kk<=p:
                temp_Omega[kk,kk-1]=0.5
                temp_Omega[kk-1,kk]=0.5
                if 2<=kk<=p:
                    temp_Omega[kk-2,kk]=0.25
                    temp_Omega[kk,kk-2]=0.25
        True_Omega = temp_Omega + np.eye(p,dtype=float)
        Sigma = sl.inv(True_Omega)
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)  
        return(Y, True_Omega)
    elif model == 'Block':
        temp_Sigma = np.zeros((p,p),dtype=float)
        for kk in range(p):
            for vv in range(p):
                if 0<=kk<=1 and  0<=vv<=p/2:
                    if not kk==vv:
                        temp_Sigma[kk,vv]=0.5
                        temp_Sigma[vv,kk]=0.5
                elif p/2+1<=kk<=10 and  p/2<=vv<=10:
                    if not kk==vv:
                        temp_Sigma[kk,vv]=0.5
                        temp_Sigma[vv,kk]=0.5
                else:
                    temp_Sigma[kk,vv] = 0
                    temp_Sigma[vv,kk] = 0
        Sigma = temp_Sigma + np.eye(p,dtype=float)
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)
        True_Omega = sl.inv(Sigma)  
        return(Y, True_Omega)
    elif model == 'Star':
        temp_Omega = np.zeros((p,p),dtype=float)
        for kk in range(p):
            if kk == 0:
                temp_Omega[kk,0] = 0
                temp_Omega[0,kk] = 0
            else:
                temp_Omega[kk,0] = 0.1
                temp_Omega[0,kk] =0.1
        True_Omega = temp_Omega + np.eye(p,dtype=float)
        Sigma = sl.inv(True_Omega)
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)
        return(Y, True_Omega)        
    elif model == 'Circle':
        temp_Omega = np.zeros((p,p),dtype=float)
        for kk in range(p):
            temp_Omega[kk,kk-1]=1.0
            temp_Omega[kk-1,kk]=1.0
        temp_Omega[0,p-1] = 0.9
        temp_Omega[p-1,0] = 0.9
        True_Omega = temp_Omega + 2*np.eye(p,dtype=float)
        Sigma = sl.inv(True_Omega)
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)
        return(Y, True_Omega)
    elif model == 'Full':
        True_Omega = np.zeros((p,p),dtype=float)
        for kk in range(p):
            for vv in range(p):
                if kk == vv:
                    True_Omega[kk, vv] = 2.0
                    True_Omega[vv, kk] = 2.0
                else:
                    True_Omega[kk, vv] = 1.0
                    True_Omega[vv, kk] = 1.0
        Sigma = sl.inv(True_Omega)
        Y = st.multivariate_normal.rvs(cov=Sigma, size=n)
        return(Y, True_Omega)
        
#%%         
#Input of simulation conditions

input('Please input data size. Input sample size n first and push enter. Then, input dimension p and push enter like:\n50\n30')
n = input()
print('n =',n )
p = input()
print('p =',p )
while True:
    if str.isnumeric(n) == False or str.isnumeric(p)== False:
        input('Error. Please input correct data size. Input sample size n first and push enter. Then, input dimension p and push enter like:\n50\n30')
        n = input()
        print('n =',n )
        p = input()
        print('p =',p )
    else:
        n = int(n)
        p = int(p)
        break    
input('Thank you. Then, please choose a number of model you want to simulate from following candidates and push enter:\n AR(1): enter 1\n AR(2): enter 2\n Block: enter 3\n Star: enter 4\n Circle: enter 5 \n Full:enter 6')
num = input()
while True:
    if num == str(1) or num == str(2) or num == str(3) or num == str(4) or num == str(5) or num == str(6):
        if num == str(1):
            model = 'AR(1)'
        elif num == str(2):
            model = 'AR(2)'
        elif num == str(3):
            model = 'Block'
        elif num == str(4):
            model = 'Star'
        elif num == str(5):
            model = 'Circle'
        else: 
            model = 'Full'
        print('model=',model)
        break
    else:
        input('Error. Please choose a correct model number and push enter. \n AR(1): enter 1\n AR(2): enter 2\n Block: enter 3\n Star: enter 4\n Circle: enter 5 \n Full:enter 6')    
        num = input()
print('Thank you. We confirmed all your inputs:n=',n,'p=',p, model,'model. Then we will compute Hit-and-Run sampler first, then compute block Gibbs Sampler and output results. Please wait for the computation to finish.')
[Y, True_Omega] = data_generation(n,p,model)

#%% HRS (Hit-and-Run sampler) proposed in this paper.
hrs_start = time.time()
#Set initial variables and save matrix for results.
n, p = Y.shape
pp = p * (p - 1) // 2
shape_gam = n / 2.0 + 1.0
ind_lower = np.tril(np.ones((p, p), dtype=bool), -1)
ind_upper = np.triu(np.ones((p, p), dtype=bool), 1)
ind_exclude = ind_lower | ind_upper
ind_diag_lower = np.tril(np.ones((p, p), dtype=bool))
S = Y.T @ Y
Omega = True_Omega
Tau = np.ones((p, p)) - np.eye(p)
r=10**(-2)
s=10**(-6)
lambda_12 = rnd.gamma(r,1/s)
lambda_22 = 1.0
burnin = 5000
sample_size = 10000
iterations = burnin + sample_size
runs_hrs = {'vech_Omega': np.empty((sample_size, p + pp))}
hrs_det = np.empty((p,iterations),dtype=float)
rpost = r + 1
jt = 0
for it in trange(iterations):        
    for i in range(p):            
        # Step 1: compute partitions
        ind_e = ind_exclude[:, i]
        s_22 = S[i, i]
        beta = Omega[ind_e, i]
        inv_Omega_11 = sl.inv(exclude_rowcol(Omega, ind_e, ind_e))
        
        # Step 2: Hit-and-Run algorithm for sampling beta
        # We used some while roops to avoid errors 
        c = quad_form(inv_Omega_11, beta) - Omega[i, i]    
        while True:    
            z = rnd.randn(p - 1)
            alp = z / sl.norm(z)
            s_12 = -S[ind_e, i]
            b = beta @ inv_Omega_11 @ alp
            a = quad_form(inv_Omega_11, alp)
            inv_C = (s_22 + 2*lambda_22) * inv_Omega_11 + np.diag(1.0 / (Tau[ind_e, i]))
            var_kappa = 1.0 / quad_form(inv_C, alp)
            if var_kappa > 0:
                break  
        mu_kappa = var_kappa * (alp @ (s_12 - inv_C @ beta))
        sd_kappa = np.sqrt(var_kappa)
        lb_kappa = ((-b - np.sqrt(b**2 - a * c)) / a - mu_kappa) / sd_kappa
        ub_kappa = ((-b + np.sqrt(b**2 - a * c)) / a - mu_kappa) / sd_kappa
        lb_prob = st.norm.cdf(lb_kappa)
        ub_prob = st.norm.cdf(ub_kappa)
        if lb_prob < ub_prob:
            while True:
                while True:
                    uni = rnd.uniform(lb_prob, ub_prob)
                    if np.isinf(uni) == False and np.isnan(uni) == False:
                        break
                stnppf = st.norm.ppf(uni)
                if np.isinf(stnppf) == False and np.isnan(stnppf) == False:
                    break
            kappa = mu_kappa + sd_kappa * stnppf                
        else:
            while True:
                uni = rnd.uniform(lb_kappa, ub_kappa) 
                if np.isinf(uni) == False and np.isnan(uni) == False:
                    break
            kappa = mu_kappa + sd_kappa * uni           
        beta += kappa * alp
        Omega = update_offdiagonals(Omega, beta, ind_e, i)             
        if sl.det(Omega) > 0:
            hrs_det[i,it] = 1.0
        else: 
            hrs_det[i,it] = 0.0
            
        # Step 3: sample gamma    
        gamma = rnd.gamma(shape_gam, 1.0 / (s_22/2 + lambda_22))
        Omega[i, i] = gamma + quad_form(inv_Omega_11, beta)
        
        # Step 4: sample lambda_12
        spost = s*np.ones((p-1,)) + np.abs(beta)
        lambda_12 = rnd.gamma(rpost, 1/spost) 
        # Step 5: sample tau_12
        tau_12 = 1 / rand_ig(np.sqrt(lambda_12**2/beta**2),lambda_12**2,p-1)
        Tau = update_offdiagonals(Tau, tau_12, ind_e, i) 
    if it >= burnin:
        # save mcmc results
        runs_hrs['vech_Omega'][jt, :] = Omega[ind_diag_lower]
        jt += 1
# compute posterior mean 
hrs_temp_Omega = np.zeros((p, p))
hrs_temp_Omega[np.tril_indices(p)] = np.mean(runs_hrs['vech_Omega'], axis=0)
hrs_post_mean_Omega = hrs_temp_Omega.T + hrs_temp_Omega - np.eye(p)*np.diag(hrs_temp_Omega)
hrs_time = time.time() - hrs_start
# output results
print('results for HRS')
print('HRS takes',hrs_time/15,'seconds per 1,000 iter.')
assess_cve(hrs_post_mean_Omega, True_Omega)
if not model == 'Full':
    assess_gsl(hrs_post_mean_Omega, True_Omega)
    assess_gsl_no_absolute_value(hrs_post_mean_Omega, True_Omega)
detcount(np.reshape(hrs_det[:, burnin:], (p*sample_size,)))
    

#BGS (block Gibbs Sampler) proposed in Wang(2012)
# We copied Wang(2012)'s BayesGlassoGDP and rewrite it in python.
bgs_start = time.time()
n, p = Y.shape
pp = p * (p - 1) // 2
shape_gam = n / 2.0 + 1.0
ind_lower = np.tril(np.ones((p, p), dtype=bool), -1)
ind_upper = np.triu(np.ones((p, p), dtype=bool), 1)
ind_exclude = ind_lower | ind_upper
ind_diag_lower = np.tril(np.ones((p, p), dtype=bool))
S = Y.T @ Y
Omega = True_Omega
Tau = np.ones((p, p)) - np.eye(p)
r=10**(-2)
s=10**(-6)
lambda_22 = 1.0
burnin = 5000
sample_size = 10000
iterations = burnin + sample_size
runs_bgs = {'vech_Omega': np.empty((sample_size, p + pp))}
bgs_det = np.empty((p,iterations),dtype=float)
rpost = r + 1
jt = 0
for it in trange(iterations):
    for i in range(p):
        s_22 = S[i, i]
        ind_e = ind_exclude[:, i]      
        beta = Omega[ind_e, i]
        Cadjust = np.zeros((p-1),dtype=float)   
        for m1 in range(p-1):
            Cadjust[m1] = max(np.abs(beta[m1]),10**-12)
        spost = s*np.ones((p-1,)) + Cadjust
        lambda_12 = rnd.gamma(rpost, 1/spost,size=beta.size)
        temp1 = lambda_12/Cadjust
        mu_prime = np.zeros((p-1,),dtype=float)
        for m2 in range(p-1):
            mu_prime[m2] = min(temp1[m2],10**12)
        lambda_prime = lambda_12**2
        temp2 = 1 / rand_ig(mu_prime,lambda_prime, beta.size)
        tau_12 = np.zeros((p-1,),dtype=float)
        for m3 in range(p-1):
            tau_12[m3] = max(temp2[m3],10**-12)        
        Tau = update_offdiagonals(Tau, tau_12, ind_e, i)        
        inv_Omega_11 = sl.inv(exclude_rowcol(Omega, ind_e, ind_e))
        s_12 = -S[ind_e, i]
        inv_C0 = (s_22 + 2*lambda_22) * inv_Omega_11 + np.diag(1.0 / (Tau[ind_e, i]))
        inv_C = (inv_C0+inv_C0.T)/2        
        chol_inv_C = sl.cholesky(inv_C)
        mu_i = sl.solve(inv_C,s_12)
        beta = mu_i + sl.solve(chol_inv_C,rnd.randn(p-1)) 
        Omega = update_offdiagonals(Omega, beta, ind_e, i)
        if sl.det(Omega) > 0:
            bgs_det[i,it] = 1.0
        else: 
            bgs_det[i,it] = 0.0
        gamma = rnd.gamma(shape_gam, 1.0 / (s_22/2+lambda_22)) 
        Omega[i,i] = gamma + quad_form(inv_Omega_11, beta)     

    if it >= burnin:
        # save mcmc results
        runs_bgs['vech_Omega'][jt, :] = Omega[ind_diag_lower]
        jt += 1
# compute posterior mean 
bgs_temp_Omega = np.zeros((p, p))
bgs_temp_Omega[np.tril_indices(p)] = np.mean(runs_bgs['vech_Omega'], axis=0)
bgs_post_mean_Omega = bgs_temp_Omega.T + bgs_temp_Omega - np.eye(p)*np.diag(bgs_temp_Omega)
bgs_time = time.time() - bgs_start
# output results
print('results for BGS')
print('BGS takes',bgs_time/15,'seconds per 1,000 iter.')
assess_cve(bgs_post_mean_Omega, True_Omega)
if not model == 'Full':
    assess_gsl(bgs_post_mean_Omega, True_Omega)
    assess_gsl_no_absolute_value(bgs_post_mean_Omega, True_Omega)
detcount(np.reshape(bgs_det[:, burnin:], (p*sample_size,)) )

