import readin as read
import analysis as analysis
import numpy as np
import constants as cst
import matplotlib.pyplot as plt

#Note: in this file, basic analysis of the optimization method is done by looking at histograms.
# More advanced methods/actual measures can also be implemented instead.


#importing all data
np_sources = read.import_scaled_sources()
print("importing matrix")
#import the combined matrix
giant_matrix = read.import_combined_matrix()
print("done importing matrix")
#and seperate the matrix into the positive and negative rows respectively
# this can be useful for plotting later on
pruned_giant_matrix = read.prune_matrix(giant_matrix)#positive rows
negative_rows_giant_matrix = read.negative_rows_of(giant_matrix)#negative rows


#default source assumes all sources to be constant at all times
#default_x = [N_COLS × source_0, N_COLS × source_1, …]^T
default_x = np.kron(np_sources, np.ones(cst.N_COLS))
#compute for comparison all simulated detections
y_sim = analysis.evaluate_y(pruned_giant_matrix, default_x)

#plot simulated detections
plt.hist(np.log10(y_sim), bins=50)
plt.xlabel("log10(y_sim)")
plt.legend()
plt.title("Simulated observations")

#compute for comparison all simulated non-detections
y_non = analysis.evaluate_y(negative_rows_giant_matrix, default_x)
#also plot the non-detections
plt.figure()
plt.hist(np.log10(-y_non), bins=50)
plt.xlabel("log10(-y_non)")
plt.legend()
plt.title("Simulated non-detections")

# Initialize all weights on x to 1, as initial guess for the optimization procedure
start_W = np.ones(default_x.shape)
resulting_W = analysis.gradient_descent_algorithm(cst.REG_LAMBDA, np_sources, giant_matrix, start_W)

#compute the simulated y for the detections using this computed weight
y_corrected = analysis.evaluate_y(pruned_giant_matrix, default_x * resulting_W)

#plot the optimized results for the detections
plt.figure()
plt.hist(np.log10(y_corrected), bins=50)
plt.xlabel("log10(y_corrected)")
plt.legend()
plt.title("Simulated observations lambda: "+str(cst.REG_LAMBDA))
# plt.show()


#compute the simulated non-detections after optimization
y_nondetections = analysis.evaluate_y(negative_rows_giant_matrix, default_x * resulting_W)

#also plot a histogram of the non-detections
plt.figure()
plt.hist(np.log10(-y_nondetections), bins=50)
plt.xlabel("log10(-y_nondetections)")
plt.title("Simulated non-detections lambda: "+str(cst.REG_LAMBDA))

#Finally, plot a histogram of the weights
plt.figure()
plt.hist(np.log10(resulting_W), bins=50)
plt.xlabel("log10(weights)")
plt.title("Optimized weights lambda: "+str(cst.REG_LAMBDA))

#saving optimized parameters
np.savetxt("x_optimized_lambda_"+str(cst.REG_LAMBDA)+".csv", default_x * resulting_W)
np.savetxt("W_optimized_lambda_"+str(cst.REG_LAMBDA)+".csv", resulting_W)


plt.show()
