import readin as read
import analysis as analysis
import numpy as np
import constants as cst
import matplotlib.pyplot as plt

np_sources = read.import_scaled_sources()
print("importing matrix")
#import the combined matrix
giant_matrix = read.import_combined_matrix()
print("done importing matrix")
#and throw away all zero/negative rows
pruned_giant_matrix = read.prune_matrix(giant_matrix)
negative_rows_giant_matrix = read.negative_rows_of(giant_matrix)


#default source assumes all sources to be constant at all times
default_x = np.kron(np_sources, np.ones(cst.N_COLS))
y_sim = analysis.evaluate_y(pruned_giant_matrix, default_x)

plt.hist(np.log10(y_sim), bins=50)
plt.xlabel("log10(y_sim)")
plt.legend()
plt.title("Simulated observations")
# plt.show()

# nb_rows = giant_matrix.shape[1]
print("giant matrix shape: ", pruned_giant_matrix.shape)
# print("nb cols: ", nb_rows)

# print(analysis.grad_cost_function(np.ones(nb_rows)))
W = np.ones(default_x.shape)
print(analysis.grad_cost_function(y_sim, default_x, pruned_giant_matrix, W, 1))


#just some random value to try out gradient descent
reg_lambda = 1.0
# reg_lambda = 0.0
# reg_lambda = 1.0
start_W = np.ones(default_x.shape)

# print(np.linalg.svd(pruned_giant_matrix))


# grad = analysis.grad_cost_function(y_sim, default_x, pruned_giant_matrix, W, reg_lambda)
#
resulting_W = analysis.gradient_descent_algorithm(reg_lambda, np_sources, pruned_giant_matrix, start_W)
y_corrected = analysis.evaluate_y(pruned_giant_matrix, default_x * resulting_W)

plt.figure()
plt.hist(np.log10(y_corrected), bins=50)
plt.xlabel("log10(y_corrected)")
plt.legend()
plt.title("Simulated observations lambda: "+str(reg_lambda))
# plt.show()


#simulated non-detections after optimization
y_nondetections = analysis.evaluate_y(negative_rows_giant_matrix, default_x * resulting_W)
#also plotting a histogram of the non-detections
plt.figure()
plt.hist(np.log10(-y_nondetections), bins=50)
plt.xlabel("log10(-y_nondetections)")
plt.title("Simulated non-detections lambda: "+str(reg_lambda))

plt.figure()
plt.hist(np.log10(resulting_W), bins=50)
plt.xlabel("log10(weights)")
# plt.hist(resulting_W, bins=50)
# plt.xlabel("weights")
plt.title("Optimized weights lambda: "+str(reg_lambda))
print("weights: ", resulting_W)

plt.show()
