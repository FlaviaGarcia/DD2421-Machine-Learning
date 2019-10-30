# -*- coding: utf-8 -*-

import numpy as np, random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


np.random.seed(100)

# GLOBAL VARIABLES 
FLOAT_ZERO_TH = math.pow(10, -5)


def generate_data_1():
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], 
                             np.random.randn(10, 2) * 0.2 + [-1.5, 0.5])) 
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB
    

def generate_data_2():
    classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [3, 1], 
                             np.random.randn(10, 2) * 0.3 + [4, 0.5])) 
    classB = np.random.randn(20, 2) * 0.1 + [0.0, -2]
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB
    

def generate_data_3():
    classA = np.concatenate((np.random.randn(10, 2) * 0.9 + [3, 1], 
                             np.random.randn(10, 2) * 0.9 + [4, 0.5])) 
    classB = np.random.randn(20, 2) * 0.8 + [3, -2]
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB


def generate_data_4():
    classA = np.concatenate((np.random.randn(10, 2) * 0.9 + [3, 1], 
                             np.random.randn(10, 2) * 0.9 + [4, 0.5],
                             np.random.randn(10, 2) * 0.2 + [-3, -0.25])) 
    classB = np.concatenate((np.random.randn(20, 2) * 0.8 + [3, -2],
                            np.random.randn(10, 2) * 0.1 + [-1, 0.5]))
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB


def generate_data_5():
    classA = np.concatenate((np.random.randn(10, 2) * 0.9 + [3, 1], 
                             np.random.randn(10, 2) * 0.9 + [4, 0.5],
                             np.random.randn(10, 2) * 0.2 + [-3, -0.25],
                             np.random.randn(10,2) * 0.4 + [1,-3])) 
    classB = np.concatenate((np.random.randn(20, 2) * 0.8 + [3, -2],
                            np.random.randn(10, 2) * 0.1 + [-1, 0.5],
                            np.random.randn(10, 2) * 0.4 + [-4, -0.5]))
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB



def generate_data_6():
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [3, 1], 
                             np.random.randn(10, 2) * 0.2 + [4, 0.5],
                             np.random.randn(10, 2) * 0.3 + [-3, -0.25],
                             np.random.randn(10,2) * 0.4 + [1,-3],
                             np.random.randn(10,2) * 0.2 + [-0.4,-0.3])) 
    classB = np.concatenate((np.random.randn(20, 2) * 0.3 + [3, -2],
                            np.random.randn(20, 2) * 0.1 + [-1, 0.5],
                            np.random.randn(10, 2) * 0.4 + [-4, -0.5]))
    
    inputs = np.concatenate((classA, classB)) 
    targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    permute=list(range(N))
    random.shuffle(permute) 
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return inputs, targets, N, classA, classB




X, t_vector, N, classA, classB = generate_data_3()


#
#X = np.array([
#        [4,1],
#        [1, 6]
#        ])
#    
#t_vector = np.array([-1,1])
#N = len(X)    


## KERNELS 
def linear_kernel(vector_x, vector_y):
    return np.dot(vector_x, vector_y)


def polynomial_kernel(vector_x, vector_y, polynomial_power):
    return math.pow((np.dot(vector_x, vector_y) + 1), polynomial_power)


def RBF_kernel(vector_x, vector_y, sigma):
    module_result_v = np.linalg.norm(vector_x-vector_y)
    return math.exp(-math.pow(module_result_v, 2)/(2*math.pow(sigma, 2)))



## SVM auxiliary functions
def get_K_all_data_points(is_kernel_linear= False, is_kernel_polynomial = False, is_kernel_RBF = False, polynomial_power=2, sigma=0.9):
    K = np.zeros([N,N])
    for idx_row in range(N):
        for idx_col in range(N):
            if is_kernel_linear:
                K[idx_row, idx_col] = linear_kernel(X[idx_row], X[idx_col])
            elif is_kernel_polynomial:
                K[idx_row, idx_col] = polynomial_kernel(X[idx_row], X[idx_col], polynomial_power)
            elif is_kernel_RBF:
                K[idx_row, idx_col] = RBF_kernel(X[idx_row], X[idx_col], sigma)
    return K




def get_Ti_Tj():
    ti_tj = np.zeros([N,N])
    for idx_row in range(N):
        for idx_col in range(N):
            ti_tj[idx_row, idx_col] = t_vector[idx_row]*t_vector[idx_col]
    return ti_tj



## MORE GLOBAL VARIABLES 
K = get_K_all_data_points(is_kernel_linear=True)
Ti_Tj = get_Ti_Tj()  
P = K * Ti_Tj




## SVM main functions

def get_alpha_vector_after_optimization(slack=False, C=0):
    start = np.zeros(N)
    if slack:
        B = [(0, C) for b in range(N)]
    else:    
        B = [(0, None) for b in range(N)]
    cons = {'type':'eq', 'fun':zerofun}
    
    ret = minimize(objective_function, start, bounds = B, constraints = cons)
    learned_alpha_vector = ret['x']
    return learned_alpha_vector


def zerofun(alpha_vector):
    return np.dot(alpha_vector, t_vector) # the value that should be constrained to zero 


def objective_function(alpha_vector):
    alpha_matrix = np.zeros([N,N])
    for idx_row in range(N):
        for idx_col in range(N):
            alpha_matrix[idx_row, idx_col] = alpha_vector[idx_row] * alpha_vector[idx_col]
    
    return (1/2) * np.sum(alpha_matrix * P) - np.sum(alpha_vector)
    

def get_non_zero_alpha(learned_alpha_vector):
    non_zero_alpha_X = []
    non_zero_alpha_t = []
    non_zero_alpha = []
    for idx in range(len(learned_alpha_vector)):
        if learned_alpha_vector[idx] > FLOAT_ZERO_TH:
            non_zero_alpha.append(learned_alpha_vector[idx])
            non_zero_alpha_X.append(X[idx])
            non_zero_alpha_t.append(t_vector[idx])
            
    return non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t


def calculate_b(non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t, idx_SV = 0):
    SV_X = non_zero_alpha_X[idx_SV]
    SV_t = non_zero_alpha_t[idx_SV]
    
    K = get_K_new_data_point(SV_X, non_zero_alpha_X, is_kernel_linear=True)
    
    return np.sum(np.asarray(non_zero_alpha) * np.asarray(non_zero_alpha_t) * K) - SV_t
     

def indicator_function(new_data_point, b, non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t):
    K = get_K_new_data_point(new_data_point, non_zero_alpha_X, is_kernel_linear=True)
    return np.sum(np.asarray(non_zero_alpha) * np.asarray(non_zero_alpha_t) * K) - b
    

def get_K_new_data_point(new_data_point, X, is_kernel_linear = False, is_kernel_polynomial = False, is_kernel_RBF = False, polynomial_power = 2, sigma=0.9):
    N = len(X)
    K = np.zeros(N)
    for idx in range(N):
        if is_kernel_linear:
            K[idx] = linear_kernel(X[idx], new_data_point)
        elif is_kernel_polynomial:
            K[idx] = polynomial_kernel(X[idx], new_data_point, polynomial_power)
        elif is_kernel_RBF:
            K[idx] = RBF_kernel(X[idx], new_data_point, sigma)
    return K


## PLOTTING FUNTIONS
def plot_training_data(show_plot = False, fig_name_to_save = "training_data.pdf"):    
    plt.plot([p[0] for p in classA],
            [p[1] for p in classA],
             'b.') 
    
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
     
    
    plt.axis('equal') # Force same scale on both axes 
    if show_plot:
        plt.savefig(fig_name_to_save) # Save a copy in a file 
        plt.show() # Show the plot on the screen


def plot_decision_boundary(b, non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t, fig_name_to_save = "svm.pdf"):
    xgrid = np.linspace(-5, 5) 
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator_function(np.array([x,y]), b, non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t) 
                    for x in xgrid] 
                    for y in ygrid])
    plt.contour(xgrid, ygrid, grid, 
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'), 
                linestyles=['--', '-', '--'])
                #linewidths=(1, 3, 1))
    


def mark_support_vectors(non_zero_alpha_X, fig_name_to_save = "svm.pdf"):
    SV_x = np.asarray([x[0] for x in non_zero_alpha_X])
    SV_y = np.asarray([x[1] for x in non_zero_alpha_X])
    plt.scatter(SV_x, SV_y, s=50, edgecolors='k', lw = 1 , facecolors='none');
    plt.savefig(fig_name_to_save)       
                
    



if __name__ == "__main__":
    
    
    plot_training_data()
    
    alpha_vector = get_alpha_vector_after_optimization(slack=True, C=5)
   
    
    non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t = get_non_zero_alpha(alpha_vector)
    
    
    b = calculate_b(non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t)
    
    ## PLOTTING
    plot_training_data()
    plot_decision_boundary(b, non_zero_alpha, non_zero_alpha_X, non_zero_alpha_t)
    mark_support_vectors(non_zero_alpha_X)
    
   