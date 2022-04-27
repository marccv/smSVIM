#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:04:40 2022

@author: marcovitali
"""

import numpy as np
import matplotlib.pyplot as plt

    
    
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
        
        self.matrix = 0.5 + 0.5*np.cos(2*np.pi * np.multiply(self.K,self.X) )
        
        # fig1=plt.figure(num=1, figsize=(10,4))
        # fig1.clf()
        # ax1=fig1.add_subplot(121)

        # for i in range(4):
        #     ax1.plot(self.x, self.matrix[i,:], '-o')
        
    def create_matrix_sq(self):
        
        Periods = np.reciprocal(self.K)
        Periods[0,:] = 10* np.ones([1,self.N])
        self.matrix = ( (self.X + Periods/4)%(Periods) < (Periods/2))*1
        
        # fig1=plt.figure(num=1, figsize=(10,4))
        # fig1.clf()
        # ax1=fig1.add_subplot(121)

        # for i in range(4):
        #     ax1.plot(self.x, self.matrix[i,:] + 0.1*i, '-o')
        
    def compute_inverse(self):
        
        #condition number
        
        det  = np.linalg.det(self.matrix)
        cond = np.linalg.cond(self.matrix)
        rank = np.linalg.matrix_rank(self.matrix)
        
        print(f'Condition number:  {cond:.5e}\nRank:             {rank:6.5f}\nDeterminat:        {det:.5e}')
        
        self.inv_matrix = np.linalg.inv(self.matrix)
        


if __name__ == '__main__':
    
    disp_f  = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942])
    
    tras = dct_6090(disp_f)
    tras.create_space()
    tras.create_matrix_sq()
    tras.compute_inverse()
    
    # x = np.array([0,0,0,0,0,0,0,0,0,0,1,3,5,9,10,9,5,3,1,0,0,0,0,0])
    x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,9,10,9,1,0,0,0,0,0,0,0])
    y = np.matmul(tras.matrix,x)
    
    x_prime = np.matmul(tras.inv_matrix,y)
    
    fig2=plt.figure(num=2, figsize=(10,4))
    fig2.clf()
    ax2=fig2.add_subplot(121)

    ax2.plot(tras.x, x,'-o')
    ax2.plot(tras.x, y,'o')
    ax2.plot(tras.x, x_prime,'x', color = 'C3')
    
    
    
    
    
    
    