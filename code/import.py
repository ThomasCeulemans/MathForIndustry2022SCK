import pandas
import os
import numpy as np
import matplotlib.pyplot as plt

MATRIX_LOCATION = "./../data/srs_csv_files/"


scalings=pandas.read_csv("./../data/scalings.csv", header=None)
sources=pandas.read_csv("./../data/data_xe133_NH_yearly.csv")#has a header file, not that we actually need it

np_scalings=scalings.to_numpy()
np_sources=sources['emission'].to_numpy()
#rescaling sources
np_sources=np_sources/np_scalings

n_datafiles=0
for file in os.listdir(MATRIX_LOCATION):
    if file.endswith(".csv"):
        n_datafiles+=1
results=np.zeros(n_datafiles)
print(results)

index=0
for file in os.listdir(MATRIX_LOCATION):
    if file.endswith(".csv"):
        matrix_file = pandas.read_csv(os.path.join(MATRIX_LOCATION, file), header=None)
        np_matrix = matrix_file.to_numpy()
        #prune extension from filename
        pruned_name = file.split('.')[0]
        split_name = pruned_name.split('_')
        # print(split_name)
        location = split_name[0]
        date = split_name[1]
        time = split_name[2]

        #checking whether everything sums to 1 (or something negative)

        result=np.sum(np.matmul(np_sources, np_matrix))
        results[index]=result

        index+=1
        # print(os.path.join(MATRIX_LOCATION, file))
        # print(matrix_file)


        #TODO do something with the data

print(scalings)
print(sources)

plt.hist(np.log10(results), bins=50)
plt.show()
