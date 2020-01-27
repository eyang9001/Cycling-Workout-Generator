import parsers
import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree
from fitparse import FitFile
from datetime import datetime

def find_max_hr(file_prefix='all_data'):
    # Load data
    hrs = np.load('data/' + file_prefix + '_hr.npy')
    cads = np.load('data/' + file_prefix + '_cad.npy')
    pws = np.load('data/' + file_prefix + '_pwr.npy')
    dates = np.load('data/' + file_prefix + '_dates.npy')

    max = max(hrs)
    indices = hrs.index(max)
    max_dates = dates[indices]

    print('Max heart rate: ' + str(max))
    print('Dates: ' + str(max_dates))

if __name__ == "__main__":
    find_max_hr()