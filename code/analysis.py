import numpy as np

def ln_y_squared(giant_matrix, xval_array):
    lny = np.log(giant_matrix @ xval_array)
    return lny.T @ lny

#W is vector of W
def cost_function(giant_matrix, W, reg_lambda):
    # TODO: use W instead of xval_array; hint: reconstruct xvalues using this W (just kron as below)
    lny = np.log(giant_matrix @ xval_array)
    regularization = reg_lambda * np.log(weigths)
    return 1.0/2.0*(lny.T @ lny +regularization.T @ regularization)

def evaluate_x(giant_matrix, xval_array):
    return giant_matrix @ xval_array



#Gradient descent; is returned as a vector instead of a diagonal matrix
#W is assumed to be a vector (corresponds with a diagonal matrix)
def grad_cost_function(y, x_default, giant_matrix, W, reg_lambda, stepsize):
    print("matrix shape_T: ", giant_matrix.T.shape)
    print("y shape: ", y.shape)
    print("matrix_T times y shape: ", (giant_matrix.T @ (np.log(y)/y)).shape)
    print("x_T shape: ", x_default.T.shape)
    #to determine the diagonal of the large matrix, we do not explicitly need to compute this large matrix
    # matrix_cost = np.diag((giant_matrix.T @ (np.log(y)/y)) @ x_default.T)
    #this 'diagonal matrix' is stored in vector format
    matrix_cost = (giant_matrix.T @ (np.log(y)/y)) * x_default
    reg_cost = reg_lambda / W * np.log(W)
    return 1.0/2.0*(matrix_cost+reg_cost)




#METRICS
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
