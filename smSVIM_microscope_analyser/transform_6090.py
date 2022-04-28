#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:04:40 2022

@author: marcovitali
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})
    
    
class dct_6090:
    
    
    def __init__(self, disp_f):
        
        self.N = len(disp_f)
        self.disp_f = disp_f


    def create_space(self):
        
        N = self.N
        
        self.k = self.disp_f
        self.x = np.linspace(1/(2*N), (N - 0.5)/N, N)
        
        self.X , self.K = np.meshgrid(self.x,self.k)
        
    def create_matrix_cos(self):
        
        self.matrix =  np.cos(2*np.pi * np.multiply(self.K,self.X) )
        
        # fig1, ax1 = plt.subplots(1,1)
        # ax1.set_ylim([-1.1,1.1])
        # ax1.set_xlabel('x', fontsize = 12)
        # for i in range(4):
        #     ax1.plot(self.x, self.matrix[i,:], '-o')
        
    def create_matrix_sq(self):
        
        temp = self.K
        temp[0,:] = 0.1* np.ones([1,self.N])
        Periods = np.reciprocal(temp)
        self.matrix = -1 + 2*( (self.X + Periods/4)%(Periods) < (Periods/2))
        
        # fig1, ax1 = plt.subplots(1,1)
        # ax1.set_ylim([-1.1,1.3])
        # ax1.set_xlabel('x', fontsize = 12)
        # for i in range(4):
        #     ax1.plot(self.x, self.matrix[i,:] + 0.08*i, '-o')
        
    def compute_inverse(self):
        
        #condition number
        
        # det  = np.linalg.det(self.matrix)
        cond = np.linalg.cond(self.matrix)
        # rank = np.linalg.matrix_rank(self.matrix)
        
        print(f'\nCondition number:  {cond:.5e}\n')
        
        self.inv_matrix = np.linalg.inv(self.matrix)
    
# =====================================================================================
# =====================================================================================

if __name__ == '__main__':
    
    import scipy.fftpack as sp_fft
    
    disp_f  = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942])
    
    tras = dct_6090(disp_f)
    tras.create_space()
    
    # base = 'cos'
    base = 'sq'
    
    
    if base == 'cos':
        tras.create_matrix_cos()
    elif base == 'sq':
        tras.create_matrix_sq()
    
    tras.compute_inverse()
    
    # x = np.array([0,0,0,0,0,0,0,0,0,0,1,3,5,9,10,9,5,3,1,0,0,0,0,0])
    x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,9,10,9,1,0,0,0,0,0,0,0])
    
    
    y = np.matmul(tras.matrix,x)
    err_on_y = False
    
    sigma = 0
    offset = 0.4
    # offset = 0 # with PosNeg I remove darkcounts
    error_1D = np.random.normal(loc = 1, scale = sigma, size = y.shape)
    error_2D = np.random.normal(loc = 1, scale = sigma, size = tras.matrix.shape)
    
    # ----------------------------------
    #  I add error to the direct matrix
    # ----------------------------------

    # m_dir_with_e = np.multiply(tras.matrix + np.max(np.max(tras.matrix, 0))*offset, error_2D)
    
    # y_with_e = np.matmul(m_dir_with_e, x)
    # err_on_y = True
    
    # fig1, ax1 = plt.subplots(1,1)
    # # ax1.set_ylim([-1.1,1.1])
    # ax1.set_xlabel('x', fontsize = 12)
    # for i in range(4):
    #     ax1.plot( m_dir_with_e[i,:], '-o')
    # --------------------------------------
    #  I add error to the measured spectrum
    # --------------------------------------
    
    
    y_with_e = np.multiply(y + max(y)*offset, error_1D)
    # y_with_e =y + max(y)*offset
    err_on_y = True
    
    
    # ----------------------------------
    
    if err_on_y:
        
        y_dct = sp_fft.dct(x, norm = 'ortho')
        y_dct_e = np.multiply(y_dct + max(y_dct)*offset, error_1D)
        x_prime_dct = sp_fft.idct(y_dct_e, norm = 'ortho')
    
    
    # ----------------------------------
    
    
    if not err_on_y:
        x_prime_inv = np.matmul(tras.inv_matrix,y)
        x_prime_solve = np.linalg.solve(tras.matrix, y_with_e)
    else:
        
        x_prime_inv = np.matmul(tras.inv_matrix, y_with_e)
        x_prime_solve = np.linalg.solve(tras.matrix, y_with_e)
        x_prime_lstsq = np.linalg.lstsq(tras.matrix, y_with_e, rcond = None)[0]
    
    
    rec_e_inv = np.linalg.norm(x-x_prime_inv, 2)
    rec_e_solve = np.linalg.norm(x - x_prime_solve, 2)
    rec_e_dct = np.linalg.norm(x-x_prime_dct, 2)
    rec_e_lstsq = np.linalg.norm(x-x_prime_lstsq, 2)
    
    
    print(f'Error with custom base {base}, inverse matrix: {rec_e_inv:.5e}')
    print(f'Error with custom base {base}, np solve: {rec_e_solve:.5e}')
    print(f'Error with custom base {base}, np lstsq: {rec_e_lstsq:.5e}')
    print(f'Error with scipy DCT: {rec_e_dct:.5e}')
    
    fig2, (ax1, ax2) =plt.subplots(2, 1, gridspec_kw={'height_ratios': [ 3, 5]}, constrained_layout=True)
    # fig2.clf()
    ax1.plot( y,'o', color = 'C0', label = 'Noiseless Spectrum')
    if err_on_y: ax1.plot( y_with_e,'D', color = 'C1', label = f'With noise (s = {sigma})')
    ax1.set_ylim([-50,70])
    ax1.set_xlabel('frequency component number', fontsize = 12)
    ax1.legend(fontsize = 10)
    
    ax2.plot(tras.x, x,'-o', color = 'C0', label = 'Original dist.')
    ax2.plot(tras.x, x_prime_inv,'--o', color = 'C3', markersize = 6, label = f'Inverted dist. (base: {base})')
    if err_on_y: ax2.plot(tras.x, x_prime_dct,'--D', color = 'C6', markersize = 4, label = 'DCT inversion')
    ax2.set_xlabel('x', fontsize = 12)
    ax2.legend(fontsize = 10)
    
    