import numpy as np

def ln_y_squared(giant_matrix, xval_array):
    lny = np.log(giant_matrix @ xval_array)
    return lny.T @ lny

def cost_function(giant_matrix, W, lambd):
    # TODO: use W instead of xval_array; hint: reconstruct xvalues using this W (just kron as below)
    lny = np.log(giant_matrix @ xval_array)
    regularization = lambd * np.log(weigths)
    return 1.0/2.0*(lny.T @ lny +regularization.T @ regularization)

def evaluate_x(giant_matrix, xval_array):
    return giant_matrix @ xval_array




#METRICS
# Caculate metrics
# mean Model Bias
def MB(results):
    return np.mean(results[np.where(results>=0)])-1

#Fractional Bias
def FB(results):
    return 2*((1-np.mean(results[np.where(results>=0)]))/(1+np.mean(results[np.where(results>=0)])))

#Geometric Mean Bias
def MG(results):
    return np.exp(-np.mean(np.log(results[np.where(results>=0)])))

# Fraction of the data where the model is within a fraction 2 correct
def FAC2(results):
    return np.size(np.where((results < 2) & (results > .5)))/np.size(np.where(results>=0)) # Fraction of the data where the model is within a

#log10Error = np.mean(np.log10(results[np.where(results>=0)]))
#log10Error = np.mean(np.log10(results[np.where(results>=0)]))
