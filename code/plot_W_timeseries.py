import numpy as np
import constants as cst
import matplotlib.pyplot as plt
import constants as cst
import pandas as pd

lambdas = ["0.1", "1.0", "10.0"]
plotsource = 200 - 1 #index of source to plot
startindex = 380 * plotsource
endindexp1 = 380 * (plotsource + 1)#last index + 1

#copied from plot_output.py
day0 = '20140101'
daymin = pd.to_datetime(day0) + pd.DateOffset(-15)
daymax = pd.to_datetime(day0) + pd.DateOffset(364)
dates = pd.date_range(daymin, daymax)

for l in lambdas:
    string = "W_optimized_lambda_"+l+".csv"
    #load W
    W_opt = np.loadtxt(string)
    W_source = W_opt[startindex:endindexp1]
    #and reverse the indices, as the corresponding dates are ordered in reverse
    W_source = np.flip(W_source)
    plt.figure()
    plt.title("Optimized weights lambda="+l)
    plt.plot(dates, np.log10(W_source))
    plt.xlabel("Date")
    plt.ylabel("log10(W)")

plt.show()
