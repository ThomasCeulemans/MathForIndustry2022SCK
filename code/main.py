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

#default source assumes all sources to be constant at all times
default_x = np.kron(np_sources, np.ones(cst.N_COLS))
y_sim = analysis.evaluate_x(pruned_giant_matrix, default_x)

plt.hist(np.log10(y_sim), bins=50)
plt.xlabel("log10(y_sim)")
plt.legend()
plt.title("Simulated observations")
# plt.show()

nb_rows = giant_matrix.shape[1]
print(nb_rows)

# print(analysis.grad_cost_function(np.ones(nb_rows)))
W = np.ones(default_x.shape)
print(analysis.grad_cost_function(y_sim, default_x, pruned_giant_matrix, W, 1, 0.001))
