import pandas
import os
import numpy as np
import scipy.optimize as opt
import datetime
import constants as cst

MATRIX_LOCATION = "./../data/srs_csv_files/"
#Note: the relative paths for the data are currently hardcoded.

def import_scalings():
    scalings=pandas.read_csv("./../data/scalings.csv", header=None)
    np_scalings=scalings[0].to_numpy()
    return np_scalings

def import_sources():
    sources=pandas.read_csv("./../data/data_xe133_NH_yearly.csv")#has a header file, not that we actually need it
    np_sources=sources['emission'].to_numpy()*1000#emission in mBq instead of Bq
    return np_sources

def import_scaled_sources():
    return import_sources()/import_scalings()

#first check all files for ordering, then combine all matrices
#but we might just limit the max amount of days to get a grasp on the problem
def import_combined_matrix():
    measurements_dict = {};
    giant_matrix = np.zeros((cst.N_DETECTIONS, cst.N_SOURCES*cst.N_COLS))
    #read in all csv files into the dictionary
    for file in os.listdir(MATRIX_LOCATION):
        if file.endswith(".csv"):
            matrix_file = pandas.read_csv(os.path.join(MATRIX_LOCATION, file), header=None)
            np_matrix = matrix_file.to_numpy()
            #prune extension from filename
            #filename is assumed to have the following format: "measurementstation"_"date:YYYYMMDD"_"time:HHMMSS".csv
            pruned_name = file.split('.')[0]
            split_name = pruned_name.split('_')
            location = split_name[0]
            date = split_name[1]
            time = split_name[2]

            measurements_dict.update({pruned_name:{"location": location, "date": date, "time": time, "matrix": np_matrix}})

    #sort the dictionary, such that it is sorted on date, subsorted on measurement location and then subsorted on time of measurement
    timesort_dict = dict(sorted(measurements_dict.items(), key = lambda item: item[1]["time"]))
    locationsort_dict = dict(sorted(timesort_dict.items(), key = lambda item: item[1]["location"]))
    datesort_dict = dict(sorted(locationsort_dict.items(), key = lambda item: item[1]["date"]))
    index = 0
    for (dictel, val) in datesort_dict.items():
        #LIMITATION: this- part assumes that the data is only from a single year. In order to extend this, a final date must be given instead.

        #extract the nth day of the year from the date
        date_name = val["date"]
        #but we do subtract 1, as we want to start counting from 0
        dayofyear = datetime.datetime.strptime(date_name, '%Y%m%d').timetuple().tm_yday - 1
        date = dayofyear

        #in order to get a grasp on the general matrix, we can limit the number of days we use
        #LIMITATION: for now, only the maximal date is constrained; one might also want to constrain the minimal date
        if date<=cst.MAX_DATE:
            #extract the numpy matrix, in order to further manipulate it to place into the large matrix
            np_matrix = val["matrix"]

            #setting up some larger matrix for eventually filling in the combined matrix
            largermat=np.zeros((cst.N_SOURCES, cst.N_COLS))
            #putting the numpy matrix on the correct place, shifted by the date
            # largermat[:, date:date+cst.LIMIT_BACKWARD_TIME]=np_matrix
            # print(cst.MAX_DATE-date-cst.LIMIT_BACKWARD_TIME)
            # print(cst.LIMIT_BACKWARD_TIME-date)
            largermat[:, (cst.MAX_DATE-date):(cst.MAX_DATE-date+cst.LIMIT_BACKWARD_TIME)]=np_matrix
            #flatten the larger matrix in order to place all consecutive rows after each other
            flatmat=largermat.flatten()

            #This flat matrix becomes a new row of the larger matrix
            giant_matrix[index, :]=flatmat
            index+=1

    return giant_matrix


#remove all negative rows from the matrix
def prune_matrix(matrix):
    return matrix[(matrix.min(axis=1)>=0.0) & (matrix.max(axis=1)>0.0)]


#only keep negative row from the matrix
def negative_rows_of(matrix):
    return matrix[matrix.min(axis=1)<0.0]
