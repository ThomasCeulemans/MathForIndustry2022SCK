import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import datetime

MATRIX_LOCATION = "./../data/srs_csv_files/"
#TODO: figure out these parameters from the files themselves
LIMIT_BACKWARD_TIME = 15
N_SOURCES = 200
N_DETECTIONS = 4636
#allowing us to use values for 1 entire year
N_COLS=365+LIMIT_BACKWARD_TIME


scalings=pandas.read_csv("./../data/scalings.csv", header=None)
sources=pandas.read_csv("./../data/data_xe133_NH_yearly.csv")#has a header file, not that we actually need it

print(scalings)
print(sources)

np_scalings=scalings[0].to_numpy()
np_sources=sources['emission'].to_numpy()*1000#emission in mBq instead of Bq
# print(np_sources.size)
# print(np_sources.shape)
#rescaling sources
# print("scalings: ", np_scalings)
# print("before dividing: ", np_sources)
np_sources=np_sources/np_scalings
# print("after dividing:", np_sources)
#np.divide(np_sources,np_scalings)


# print(np_scalings.size)
# print(np_scalings.shape)
# print("np sources: ", np_sources.size)

n_datafiles=0
for file in os.listdir(MATRIX_LOCATION):
    if file.endswith(".csv"):
        n_datafiles+=1
results=np.zeros(n_datafiles)
print(results)




#TODO figure out whether this thing needs to be sparse
#probably
# def fill_matrix(y, index, ):
    # row =
    # col =
    # data =
#or just construct from standard numpy array...



index=0

#timesteps times number of emissions
nb_data_cols=np.size(np_sources)
print(nb_data_cols)

combined_dataframe=pandas.DataFrame()
giant_matrix = np.zeros((N_DETECTIONS, N_SOURCES*N_COLS))
measurements_dict = {};



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

        #consistent index, starting at zero
        dayofyear = datetime.datetime.strptime(date, '%Y%m%d').timetuple().tm_yday - 1
        # print(dayofyear)

        # print("matrix: ", np_matrix.shape)
        measurements_dict.update({pruned_name:{"location": location, "date": date, "time": time, "matrix": np_matrix}})

timesort_dict = dict(sorted(measurements_dict.items(), key = lambda item: item[1]["time"]))
locationsort_dict = dict(sorted(timesort_dict.items(), key = lambda item: item[1]["location"]))
datesort_dict = dict(sorted(locationsort_dict.items(), key = lambda item: item[1]["date"]))
# print(locationsort_dict)

index = 0
for (dictel, val) in datesort_dict.items(): # dict(sorted(measurements_dict.items(), key=lambda item: item["date"])):
    print(dictel)
    # print(val["location"])

    np_matrix = val["matrix"]
    #checking whether everything sums to 1 (or something negative)
    # print(np.matmul(np_matrix.T, np_sources))
    result=np.sum(np.matmul(np_matrix.T, np_sources))
    # print()
    results[index]=result

    #setting up some larger matrix for eventually filling in the combined matrix
    largermat=np.zeros((N_SOURCES, N_COLS))
    #flatten the transposed matrix

    date=dayofyear

    #and putting it in in the correct place?
    largermat[:, date:date+LIMIT_BACKWARD_TIME]=np_matrix
    print(largermat.shape)
    flatmat=largermat.flatten()
    print(flatmat.shape)

    #and put it into a giant matrix
    giant_matrix[index, :]=flatmat
    #hmm, did I correctly treat observations on the same day; probably yes, as the observation will be put on another row

    # print(flatmat)

    index+=1
    # print(os.path.join(MATRIX_LOCATION, file))
    # print(matrix_file)

print("size giant matrix: ", giant_matrix.shape)
# print("giant matrix: ", giant_matrix)
#we do not actually want to deal with the negative values, nor with rows of only zeros (as some gap in the data exists temporally)
pruned_giant_matrix = giant_matrix[(giant_matrix.min(axis=1)>=0.0) & (giant_matrix.max(axis=1)>0.0)]
print("size pruned matrix: ", pruned_giant_matrix.shape)
#TODO: prune zero columns?, as these cannot be determineds
print(scalings)
print(sources)

plt.hist(np.log10(results), bins=50)
plt.xlabel("log10(y_sim)")
plt.legend()
plt.title("Simulated observations")
plt.show()
