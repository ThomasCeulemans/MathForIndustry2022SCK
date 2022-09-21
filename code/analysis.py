import numpy as np
import constants as cst
import scipy as sp

def ln_y_squared(giant_matrix, xval_array):
    lny = np.log(giant_matrix @ xval_array)
    return lny.T @ lny

#W is vector
#TODO FIXME
#xval_array = W * default_x
def cost_function(giant_matrix, xval_array, W, reg_lambda):
    # TODO: use W instead of xval_array; hint: reconstruct xvalues using this W (just kron as below)
    lny = np.log(giant_matrix @ xval_array)
    regularization = reg_lambda * np.log(W)
    print("cost: ", 1.0/2.0*(lny.T @ lny +regularization.T @ regularization))
    print("min weight: ", np.min(W))
    return 1.0/2.0*(lny.T @ lny +regularization.T @ regularization)

def evaluate_y(giant_matrix, xval_array):
    return giant_matrix @ xval_array



#Gradient descent; is returned as a vector instead of a diagonal matrix
#W is assumed to be a vector (corresponds with a diagonal matrix)
def grad_cost_function(y, x_default, giant_matrix, W, reg_lambda):
    # print("matrix shape_T: ", giant_matrix.T.shape)
    # print("y shape: ", y.shape)
    # print("matrix_T times y shape: ", (giant_matrix.T @ (np.log(y)/y)).shape)
    # print("x_T shape: ", x_default.T.shape)
    #to determine the diagonal of the large matrix, we do not explicitly need to compute this large matrix
    # matrix_cost = np.diag((giant_matrix.T @ (np.log(y)/y)) @ x_default.T)
    #this 'diagonal matrix' is stored in vector format
    matrix_cost = (giant_matrix.T @ (np.log(y)/y)) * x_default
    reg_cost = reg_lambda / W * np.log(W)
    return (matrix_cost+reg_cost)

#cost function using lnW
def cost_function_ln(giant_matrix, xval_array, lnW, reg_lambda):
    # W = np.exp(lnW)
    lny = np.log(giant_matrix @ xval_array)
    regularization = reg_lambda * lnW
    print("min lnW: ", np.min(lnW))
    print("max lnW: ", np.max(lnW))
    cost = 1.0/2.0*(lny.T @ lny +regularization.T @ regularization)
    print("cost: ", cost)
    return cost


#Gradient descent; is returned as a vector instead of a diagonal matrix
#gradient function using lnW to derive to
def grad_cost_function_ln(y, x_default, giant_matrix, lnW, reg_lambda):
    # print("matrix shape_T: ", giant_matrix.T.shape)
    # print("y shape: ", y.shape)
    # print("matrix_T times y shape: ", (giant_matrix.T @ (np.log(y)/y)).shape)
    # print("x_T shape: ", x_default.T.shape)
    #to determine the diagonal of the large matrix, we do not explicitly need to compute this large matrix
    # matrix_cost = np.diag((giant_matrix.T @ (np.log(y)/y)) @ x_default.T)
    #this 'diagonal matrix' is stored in vector format
    W = np.exp(lnW)
    matrix_cost = W * (giant_matrix.T @ (np.log(y)/y)) * x_default
    reg_cost = reg_lambda * lnW
    return (matrix_cost+reg_cost)


#for all negative values, we transform it into the following
def h_transform(Mx):
    # print("before h_transform: ", Mx)
    negative_indices = Mx < 0
    # print("negative indices: ", negative_indices)
    small_negative_indices = (Mx < 0) & (Mx > -2)
    # print("small negative indices: ", small_negative_indices)
    Mx[negative_indices] = -Mx[negative_indices]
    Mx[small_negative_indices] = np.power(Mx[small_negative_indices], 2) / 4.0 + np.ones(Mx[small_negative_indices].shape)
    # print("h_transform: ", Mx)
    return Mx

#returns gradient multiplication factors
def grad_h_transform(Mx):
    temp_grad = np.ones(Mx.shape)
    negative_indices = Mx < 0

    small_negative_indices = (Mx < 0) & (Mx > -2)
    temp_grad[negative_indices] = -temp_grad[negative_indices]
    temp_grad[small_negative_indices] = Mx[small_negative_indices] / 2.0
    # print("tempgrad: ", temp_grad)
    return temp_grad


#cost function using lnW and the h function transform
def cost_function_ln_hfun(giant_matrix, xval_array, lnW, reg_lambda):
    # W = np.exp(lnW)
    lny = np.log(h_transform(giant_matrix @ xval_array))
    regularization = reg_lambda * lnW
    print("min lnW: ", np.min(lnW))
    print("max lnW: ", np.max(lnW))
    cost = 1.0/2.0*(lny.T @ lny +regularization.T @ regularization)
    print("cost: ", cost)
    return cost


#Gradient descent; is returned as a vector instead of a diagonal matrix
#gradient function using lnW to derive to and using the h function transform
#Note: y should be the original y=MWx, which contains negative values for the non-detections
def grad_cost_function_ln_hfun(y, x_default, giant_matrix, lnW, reg_lambda):
    # print("matrix shape_T: ", giant_matrix.T.shape)
    # print("y shape: ", y.shape)
    # print("matrix_T times y shape: ", (giant_matrix.T @ (np.log(y)/y)).shape)
    # print("x_T shape: ", x_default.T.shape)
    #to determine the diagonal of the large matrix, we do not explicitly need to compute this large matrix
    # matrix_cost = np.diag((giant_matrix.T @ (np.log(y)/y)) @ x_default.T)
    #this 'diagonal matrix' is stored in vector format
    y_trans = h_transform(y)
    W = np.exp(lnW)
    matrix_cost = W * (giant_matrix.T @ (np.log(y_trans)/y_trans * grad_h_transform(y))) * x_default
    reg_cost = reg_lambda * lnW
    return (matrix_cost+reg_cost)


#returns the optimized weights
def gradient_descent_algorithm(reg_lambda, np_sources, giant_matrix, start_W):
    default_x = np.kron(np_sources, np.ones(cst.N_COLS))
    W = start_W

    #optimizing lnW makes sure that W will always be positive
    lnW = np.log(W)

    # #original optimization stuff for W
    # lambda_cost_function = lambda W : cost_function(giant_matrix, W*default_x, W, reg_lambda)
    # lambda_gradient_function = lambda W : grad_cost_function(evaluate_y(giant_matrix, W*default_x), default_x, giant_matrix, W, reg_lambda)
    #
    # #(lazy) derived optimization stuff for ln(W)
    # lnW_cost_function = lambda lnW : lambda_cost_function(np.exp(lnW))
    # lnW_gradient_function = lambda lnW : np.exp(lnW)*lambda_gradient_function(np.exp(lnW))

    #optimization stuff for lnW
    # lnW_cost_function = lambda lnW : cost_function_ln(giant_matrix, np.exp(lnW)*default_x, lnW, reg_lambda)
    # lnW_gradient_function = lambda lnW : grad_cost_function_ln(evaluate_y(giant_matrix, np.exp(lnW)*default_x), default_x, giant_matrix, lnW, reg_lambda)

    #optimization stuff for lnW, also incorporating the non-detections
    lnW_cost_function = lambda lnW : cost_function_ln_hfun(giant_matrix, np.exp(lnW)*default_x, lnW, reg_lambda)
    lnW_gradient_function = lambda lnW : grad_cost_function_ln_hfun(evaluate_y(giant_matrix, np.exp(lnW)*default_x), default_x, giant_matrix, lnW, reg_lambda)


    print("gradient at initial: ", lnW_gradient_function(lnW))
    # optimization for W
    # optimized_result = sp.optimize.minimize(lambda_cost_function, start_W, method = "L-BFGS-B", jac=lambda_gradient_function)

    # optimization for lnW
    optimized_result = sp.optimize.minimize(lnW_cost_function, lnW, method = "L-BFGS-B", jac=lnW_gradient_function)
    print(optimized_result)
    return np.exp(optimized_result['x'])
    # x = default_x * W
    # for i in range(max_n_it):
    #     #compute y
    #     y = evaluate_x(giant_matrix, default_x * W)
    #     #correct W
    #     gradient = grad_cost_function(y, default_x, giant_matrix, W, reg_lambda)
    #     W -= stepsize * gradient * W
    #     print("cost: ", cost_function(giant_matrix, W * default_x, reg_lambda))
    #     if (i%print_every_n_its):
    #         print("it: ", i)
    # return W



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
