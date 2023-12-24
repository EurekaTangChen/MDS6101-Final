# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:25:11 2023

@author: Lenovo
"""

import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix,csc_matrix
import time
from scipy import sparse
from cv2 import idct, dct
from scipy.optimize import line_search
from scipy.sparse import spdiags


def ddct(x):
    #return x
    return dct(x)
def iidct(x):
    #return x
    return idct(x)


def img2array(dir,dir_):
    im = Image.open(dir+dir_).convert('L')
    img = np.array(im)
    img = img.astype(np.float64) / 255
    return img

def mask2array(dir,dir_,img):
    im = Image.open(dir+dir_).convert('L')
    mask = np.array(im)
    mask = mask.astype(np.float64) / 255    
    mask[mask > 0] = 1
    m=img.shape[0]
    n=mask.shape[0]
    if n>m:
        mask_=mask[:m,:m]
    else:
        a=np.ones((m,m))
        a[:n,:n]=mask
        mask_=a
    return mask_

def get_P(m,n,mask):
    m,n=mask.shape
    s=int(mask[mask==1].sum())
    count=0
    P=lil_matrix((s,m*n))
    for i,num in enumerate(mask.flatten()):
        if  num==1:
            P[count,i]=1
            count+=1
    return P

def get_b_u(img,P):
    u=img.flatten().reshape(-1,1)
    b=P@u
    return b,u

def get_D(m,n):
    

    """
    ----------------------------------------------------------------------------
    INPUT
    ----------------------------------------------------------------------------
      - m     dimension of the image: rows.
      - n      dimension of the image: columns.
    """
    
    mn = m*n

    ones_mnm  = -np.append(np.ones(mn-m), np.zeros(m))
    ones_mm = -np.append(np.ones(m-1), [0])
    
    data = np.vstack((ones_mnm, np.ones(mn)))
    dgs = np.array([0,m])
    Dx  = sparse.spdiags(data, dgs, mn, mn)
    
    data = np.vstack((ones_mm, np.ones(m)))
    dgs = np.array([0,1])
    Dy_Base  = sparse.spdiags(data, dgs, m, m)
    Dy = sparse.kron(sparse.eye(n), Dy_Base)
    
    D = sparse.vstack([Dx, Dy])
    
    return D


def compare_plot(img,reconstruct,mask,method):
    plt.figure(figsize=(12, 10))

    # First subplot - original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Second subplot - image with the inpainting mask overlaid
    # Using red color to indicate the damaged areas
    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(1-mask, cmap='Reds', alpha=0.3)  # Using alpha for transparency
    plt.title("Image with Inpainting Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(reconstruct)
    plt.title(method+ " reconstruct Img")
    plt.axis('off')

    # Display the plots
    plt.show()
    return

class Options:
    def __init__(self, sigma=0.5, gamma=0.1, a=1, alpha=None, maxit_gs=100, tol_gs=1e-5,delta=0.001,mu=0.01,
                 P_csr=None,P_csc=None,b=None,u=None):
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.a = a
        self.maxit_gs = maxit_gs
        self.tol_gs = tol_gs
        self.delta=delta
        self.mu=mu
        self.P_csc=P_csc
        self.P_csr=P_csr
        self.b=b
        self.u=u
option=Options()


def huber(z, delta=option.delta):

    return np.where(np.abs(z) <= delta, z**2 * (2 * delta)**(-1), np.abs(z) - 0.5 * delta)

def huber_prime(z, delta=option.delta):

    return np.where(np.abs(z) <= delta, z / delta, np.sign(z))

def obj(x,option):
    P_csr=option.P_csr
    b=option.b
    mu=option.mu  
    err=P_csr@iidct(x)-b
    return 0.5*np.dot(err.T,err)+mu*sum(huber(x))

def gradient(x,option):
    P_csc=option.P_csc
    P_csr=option.P_csr
    b=option.b
    mu=option.mu
    mid=P_csr@iidct(x)-b
    grad1=ddct(P_csc.T@mid)
    grad2=mu*huber_prime(x)
    return grad1+grad2

def PSNR(m,n,x,u):
    a=np.dot((x-u).T,(x-u))
    return 10*np.log10(m*n/a)

def lasso_huber(option,beta,  tol=1e-6, iterations=1000,lr=0.05):
    """Lasso regression using Huber smoothing with gradient descent."""
    beta_1=beta
    
    for iteration in range(iterations):
        
        k=iteration+1
        rate=(k-2)/(k+1)
        
        y_plus=beta+rate*(beta-beta_1)
        
        beta_1=beta
        grad=gradient(y_plus,option)
        # Update the coefficients
        beta =y_plus-lr*grad
        '''
        grad=gradient(P_csc,P_csr,beta,b)
        beta-=lr*grad
        '''
        bg=np.linalg.norm(grad)
        print(bg)
        if bg<=tol:
            break
    return beta


def backtracking_alpha( f, grad,x_k,option):
    alpha=0.5
    gamma=option.gamma
    sigma=option.sigma    
    norm=np.dot(grad.T, grad)
    f_x=f(x_k,option)
    while f(x_k - alpha * grad,option) > f_x - gamma * alpha * norm:
        alpha *= sigma
    return alpha

def lbfgs_two_loop_recursion(grad_diff, s_list, y_list, rho_list,option=option):
    """
    L-BFGS Two-Loop Recursion

    Parameters:
    - grad_diff: The gradients at x_k.
    - s_list: List of the most recent s vectors.
    - y_list: List of the most recent y vectors.
    - rho_list: List of the most recent rho values.

    Returns:
    - H_s_y: Matrix-vector product H_k * s_k.
    """

    q = grad_diff.copy()
    s_k, y_k = s_list[-1], y_list[-1]

    alpha_list = []

    # First loop
    for s, y, rho in zip(s_list[::-1], y_list[::-1], rho_list[::-1]):
        alpha = rho * np.dot(s.T, q)
        q -= alpha * y
        alpha_list.append(alpha)

    gamma = np.dot(s_k.T, y_k) / np.dot(y_k.T, y_k)
    nn=len(s_list[0])
    diag=spdiags(gamma *np.ones(nn), 0, nn, nn)
    H_s_y = diag@q
    # Second loop
    for s, y, rho, alpha in zip(s_list, y_list, rho_list, alpha_list[::-1]):
        beta = rho * np.dot(y.T, H_s_y)
        H_s_y += (alpha-beta)*s

    return H_s_y

def lbfgs_optimizer(objective_func, gradient_func, x0,option=option, max_iterations=100, epsilon=1e-5, m=5):
    """
    L-BFGS Optimization Algorithm with Two-Loop Recursion

    Parameters:
    - objective_func: The objective function to minimize.
    - gradient_func: The gradient (derivative) of the objective function.
    - x0: Initial guess for the optimization.
    - max_iterations: Maximum number of iterations.
    - epsilon: Convergence threshold.
    - m: Number of stored vectors.

    Returns:
    - x_optimal: Optimal solution.
    - f_optimal: Objective value at the optimal solution.
    """

    x_k = x0
    n = len(x0)

    s_list = []
    y_list = []
    rho_list = []
    p_k= -gradient_func(x0,option)
    for iteration in range(max_iterations):
        g_k = gradient_func(x_k,option)

        # Use line search to find the step size alpha
        # alpha, _, _, _, _, _ = line_search(objective_func, gradient_func, x_k, p_k)
        alpha = backtracking_alpha(obj, g_k, x_k, option)
        #alpha=0.01
        print('learning rate---------')
        print(alpha)
        if alpha is None:
            raise ValueError("Line search failed to find a suitable step size.")

        x_next = x_k + alpha * p_k
        s_k = x_next - x_k
        y_k = gradient_func(x_next,option) - g_k
        rho_k = 1 / np.dot(y_k.T, s_k)

        # Update stored vectors
        if len(s_list) == m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)

        s_list.append(s_k)
        y_list.append(y_k)
        rho_list.append(rho_k)
        bg=np.linalg.norm(gradient_func(x_next,option))
        print('gradient is ---------')
        print(bg)
        # Check for convergence
        if  bg< epsilon:
            return x_next, objective_func(x_next,option)
        
        # Check for update 
        if np.dot(s_k.T, y_k) < 1e-14:
            p_k = p_k
        else:
            # Use L-BFGS update
            p_k = -lbfgs_two_loop_recursion(g_k, s_list, y_list, rho_list)
        
        x_k = x_next

    return x_k, iteration


#########data path
dir1='C:/Users/Lenovo/Desktop/python数据分析/opt/final_project'
dir_='/test_images-2/256_256_buildings.png'
mask_dir='/test_masks/640_640_handwriting.png'

##########laod data
img=img2array(dir1, dir_)
mask=mask2array(dir1,mask_dir,img)
m,n=img.shape
P=get_P(m, n, mask)
D=get_D(m, n)
D=csc_matrix(D)
b,u=get_b_u(img,P)

mn = m*n

P_csc=csc_matrix(P)
P_csr=csr_matrix(P)


#########load params
option=Options(P_csc=P_csc,P_csr=P_csr,b=b,u=u)


############(1) AGM
x0=np.zeros((mn,1))
x=lasso_huber(option, x0, tol=1e-6, iterations=100,lr=0.05)
y=iidct(x)
print(PSNR(m,n,y,u))
PSNR_AGM=PSNR(m,n,y,u)
compare_plot(img,y.reshape(m,n),mask,'AGM')


#############(2) LBFGS
option.sigma=0.5 ###sigma越小,back tracking迭代次数越少
option.gamma=0.8 ###gamma 越小，alpha越小

x_LBFGS,_=lbfgs_optimizer(obj, gradient, x0, max_iterations=1000, epsilon=1e-6, m=10)
y_LBFGS=iidct(x_LBFGS)
print(PSNR(m,n,y_LBFGS,u))
PSNR_LBFGS=PSNR(m,n,y_LBFGS,u)
compare_plot(img,y_LBFGS.reshape(m,n),mask,'LBFGS')


