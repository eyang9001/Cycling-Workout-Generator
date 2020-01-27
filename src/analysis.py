import parsers
import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree
from fitparse import FitFile
from datetime import datetime

def find_max_hr(file_prefix='../data/all_data'):
    # Load data
    hrs = np.load(file_prefix + '_hr.npy', allow_pickle=True)
    dates = np.load(file_prefix + '_dates.npy', allow_pickle=True)

    cur_max = 0
    for i in range(len(hrs)):
        if max(hrs[i]) > cur_max:
            cur_max = max(hrs[i])
            max_index = [i]
        elif max(hrs[i]) == cur_max:
            max_index.append(i)
    max_dates = dates[max_index]
    print('Max Heart Rate: ' + str(cur_max))
    print('Dates: ' + str(max_dates))

if __name__ == "__main__":
    find_max_hr()