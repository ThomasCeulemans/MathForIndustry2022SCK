import pandas
import os
import numpy as np

MATRIX_LOCATION = "./../data/srs_csv_files/"


scalings=pandas.read_csv("./../data/scalings.csv", header=None)
sources=pandas.read_csv("./../data/data_xe133_NH_yearly.csv")#has a header file, not that we actually need it

for file in os.listdir(MATRIX_LOCATION):
    if file.endswith(".csv"):
        matrix_file = pandas.read_csv(os.path.join(MATRIX_LOCATION, file), header=None)
        #TODO: find date and location from filename
        #prune extension from filename
        pruned_name = file.split('.')[0]
        split_name = pruned_name.split('_')
        print(split_name)
        location = split_name[0]
        date = split_name[1]
        time = split_name[2]



        # print(os.path.join(MATRIX_LOCATION, file))
        # print(matrix_file)


        #TODO do something with the data

print(scalings)
print(sources)
