#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:04:40 2022

@author: marcovitali
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from numpy import genfromtxt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})
    
    
class dct_6090:
    
    
    def __init__(self, disp_f):
        
        # It receives the actual frequencies used to illuminate the sample
        
        self.N = int(np.ceil(max(disp_f)*2) + 1) # sort of Shannon Th.
        
        # I throw away doubles
        mask = np.append(np.diff(disp_f)!= 0, True)
        self.disp_f = np.array(disp_f)[mask]
    

    def create_space(self):
        
        N = self.N
        
        self.k = self.disp_f.copy()
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
        
        # We perform the following steps to create a sq wave with freq = 0, period = inf
        
        temp = self.K.copy()
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
        
        try:
            self.inv_matrix = np.linalg.inv(self.matrix)
        except:
            print('Warning: singular matrix')
            
    def compute_pinv(self):
        
        rcond_pinv = 0.1
        cond = np.linalg.cond(self.matrix)
        print(f'\nDirect matrix condition number:  {cond:.5e}\nP_inv rcond = {rcond_pinv}\n')
        
        self.pinv_matrix = np.linalg.pinv(self.matrix, rcond = rcond_pinv)
        
        
        
        
# =============================================================================
#         hadamard
# =============================================================================
        
def create_hadamard_matrix(num_of_patterns, had_type = 'hadam'):
    
    if had_type == 'hadam':
        return hadamard(num_of_patterns)
    
    elif had_type == 'walsh':
        return genfromtxt(f'/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/hadamard/wh{num_of_patterns}.csv', delimiter=',')
    
    elif had_type == 'scrambled':
        np.random.seed(222)
        
        I = np.eye(num_of_patterns)
        Pr = I[np.random.permutation(num_of_patterns), :]
        Pc = I[np.random.permutation(num_of_patterns), :]
        
        return Pr @ hadamard(num_of_patterns) @ Pc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =====================================================================================
#          main
# =====================================================================================

if __name__ == '__main__':
    
    import scipy.fftpack as sp_fft
    
    
    #actual frequncies displayed on the DMD. Normally they are recalculated while performing the image inversion (reading parameters from the H5 file)
    disp_f  = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942, 12.5, 12.5, 13.333333333333334, 14.285714285714286, 14.285714285714286, 15.384615384615385, 15.384615384615385, 16.666666666666668, 16.666666666666668, 16.666666666666668, 18.181818181818183, 18.181818181818183, 18.181818181818183, 20.0, 20.0, 20.0, 20.0, 22.22222222222222, 22.22222222222222, 22.22222222222222, 22.22222222222222, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 33.333333333333336, 33.333333333333336, 33.333333333333336])
    
    # Creation of the transform object
    
    tras = dct_6090(disp_f)
    tras.create_space()
    
    
    
    base = 'cos'
    # base = 'sq'
    
    
    if base == 'cos':
        tras.create_matrix_cos() # the direct matrix is now in tras.matrix
    elif base == 'sq':
        tras.create_matrix_sq() # the direct matrix is now in tras.matrix
    
    tras.compute_inverse() # the inverse matrix is now in tras.inv_matrix
    tras.compute_pinv()
    
    
    
    # Test signal
    x = np.exp( - ((tras.x - 0.62)**2)/0.001 )
    
    
    
    
    # Noiseless spectrum
    y = np.matmul(tras.matrix,x)
    err_on_y = False
    
    sigma = 0.3
    # offset = 0.4
    offset = 0 # with PosNeg I remove darkcounts
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
    err_on_y = True
    
    
    # ----------------------------------
    # SCIPY DCT
    
    # if err_on_y:
        
        # y_dct = sp_fft.dct(x, norm = 'ortho')
        # y_dct_e = np.multiply(y_dct + max(y_dct)*offset, error_1D)
        
        # >> To perform the DCT inversion on the DCT spectrum
        # x_prime_dct = sp_fft.idct(y_dct_e, norm = 'ortho')
        
        # >> To perform the DCT inversion on the cos or sq spectrum
        # x_prime_dct = sp_fft.idct(y_with_e, norm = 'ortho')
        
        
    # ----------------------------------
    
    
    if not err_on_y:
        # x_prime_inv = np.matmul(tras.inv_matrix,y)
        x_prime_lstsq ,_,_,_= np.linalg.lstsq(tras.matrix, y, rcond = None)
        x_prime_pinv = np.matmul(tras.pinv_matrix, y)
    else:
        
        # x_prime_inv = np.matmul(tras.inv_matrix, y_with_e)
        # x_prime_solve = np.linalg.solve(tras.matrix, y_with_e)
        x_prime_lstsq ,_,_,_= np.linalg.lstsq(tras.matrix, y_with_e, rcond = 0.1)
        x_prime_pinv = np.matmul(tras.pinv_matrix, y_with_e)
    
    
    # x_prime_dct *= max(x_prime_lstsq)/max(x_prime_dct) # rescale the DCT inverted x
    
    # rec_e_inv = np.linalg.norm(x-x_prime_inv, 2)
    # rec_e_solve = np.linalg.norm(x - x_prime_solve, 2)
    # rec_e_dct = np.linalg.norm(x-x_prime_dct, 2)
    rec_e_lstsq = np.linalg.norm(x-x_prime_lstsq, 2)
    rec_e_pinv = np.linalg.norm(x-x_prime_pinv, 2)
    
    
    # print(f'>> L2 norm errors <<\n{base} base, inverse:  {rec_e_inv:.5e}')
    # print(f'{base} base, np solve: {rec_e_solve:.5e}')
    print(f'{base} base, np lstsq: {rec_e_lstsq:.5e}')
    # print(f'Scipy DCT:          {rec_e_dct:.5e}')
    print(f'{base} base, Pinv:     {rec_e_pinv:.5e}')
    
    fig2, (ax1, ax2) =plt.subplots(2, 1, gridspec_kw={'height_ratios': [ 3, 5]}, constrained_layout=True)
    # fig2.clf()
    ax1.plot( y,'o', color = 'C0', label = 'Noiseless Spectrum')
    if err_on_y: ax1.plot( y_with_e,'D', color = 'C1', label = f'With noise (s = {sigma})')
    # ax1.set_ylim([-50,70])
    ax1.set_xlabel('frequency component number', fontsize = 12)
    ax1.legend(fontsize = 10)
    
    ax2.plot(tras.x, x,'-o', color = 'C0', label = 'Original dist.')
    ax2.plot(tras.x, x_prime_lstsq,'--o', color = 'C3', markersize = 6, label = f'Inverted LSQR (base: {base})')
    ax2.plot(tras.x, x_prime_pinv,'--x', color = 'C4', markersize = 6, label = f'Pseudo Inv dist. (base: {base})')
    
    # if err_on_y: ax2.plot(tras.x, x_prime_dct,'--D', color = 'C6', markersize = 4, label = 'DCT inversion')
    ax2.set_xlabel('x', fontsize = 12)
    ax2.legend(fontsize = 10)
    
    
    #%%
    
    temp = np.linspace(0,1,1000)
    temp_y = np.cos(2*np.pi * disp_f[-1] *temp)
    y_dct = sp_fft.dct(tras.matrix[-1,:], norm = 'ortho')
    x_dct = sp_fft.idct(y_dct, norm = 'ortho')
    
    
    fig3, ax3 = plt.subplots(1,1)
    ax3.plot(tras.x, tras.matrix[1,:])
    ax3.plot(temp, temp_y)
    ax3.plot(tras.x, tras.matrix[-1,:], 'o')
    ax3.plot(tras.x, x_dct)
    
    plt.figure()
    plt.plot(y_dct)
    
    