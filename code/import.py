import pandas
import os
import numpy as np
import matplotlib.pyplot as plt

MATRIX_LOCATION = "./../data/srs_csv_files/"


scalings=pandas.read_csv("./../data/scalings.csv", header=None)
sources=pandas.read_csv("./../data/data_xe133_NH_yearly.csv")#has a header file, not that we actually need it

print(scalings)
print(sources)

np_scalings=scalings[0].to_numpy()#/86400#number of seconds in a day
np_sources=sources['emission'].to_numpy()
print(np_sources.size)
print(np_sources.shape)
#rescaling sources
print("scalings: ", np_scalings)
print("before dividing: ", np_sources)
np_sources=np_sources/np_scalings
print("after dividing:", np_sources)
#np.divide(np_sources,np_scalings)


print(np_scalings.size)
print(np_scalings.shape)
print("np sources: ", np_sources.size)

n_datafiles=0
for file in os.listdir(MATRIX_LOCATION):
    if file.endswith(".csv"):
        n_datafiles+=1
results=np.zeros(n_datafiles)
print(results)

index=0

#timesteps times number of emissions
nb_data_cols=np.size(np_sources)
print(nb_data_cols)

combined_dataframe=pandas.DataFrame()

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

        # print("matrix: ", np_matrix.shape)



        #checking whether everything sums to 1 (or something negative)
        # print(np.matmul(np_matrix.T, np_sources))
        result=np.sum(np.matmul(np_matrix.T, np_sources))
        # print()
        results[index]=result

        index+=1
        # print(os.path.join(MATRIX_LOCATION, file))
        # print(matrix_file)


        #TODO do something with the data

print(scalings)
print(sources)

plt.hist(np.log10(results), bins=50)
plt.xlabel("log10(y_sim)")
plt.legend()
plt.title("Simulated observations")
plt.show()
