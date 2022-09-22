#This is a file defining some constants for the scripts in this folder
#Note: technically these can just be determined by reading the csv files themselves

#how many days the sensitivities go backwards in time
LIMIT_BACKWARD_TIME = 15
#number of sources
N_SOURCES = 200
#number of detections
N_DETECTIONS = 4363
#maximal date (in our case a single year)
MAX_DATE = 365
#if you want to run it using data from less days, just decrease the MAX_DATE
# MAX_DATE = 40
#total number of days to model (year + buffer for sensitivities going backwards in time)
N_COLS=MAX_DATE+LIMIT_BACKWARD_TIME

#constants you want to change
#value for the regularization parameter; higher will for the weights closer to 1, while lower will allow more overfitting
REG_LAMBDA=0.1
