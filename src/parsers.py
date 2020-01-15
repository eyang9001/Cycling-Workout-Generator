import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree
from fitparse import FitFile
from datetime import datetime

def parse_fit_file(file_path, rename=False):
# parses fit files
    fitfile = FitFile(file_path)
    hrs = []
    pwrs = []
    cads = []
    times = []
    stime = None
    for record in fitfile.get_messages('record'):
        rec_dict = record.get_values()
        if stime == None:
            stime = rec_dict['timestamp'] #stores the initial datetime for the ride
            if rename:
                new_fname = stime.strftime("%Y-%m-%dT%H_%M_%S")+'.fit'      # renames the file to the datetime
                folder = file_path[:file_path.rfind('/') + 1]
                os.rename(file_path, folder+new_fname)
        if ('power' in rec_dict) & ('cadence' in rec_dict) & ('heart_rate' in rec_dict):
            pwrs.append(rec_dict['power'])
            cads.append(rec_dict['cadence'])
            hrs.append(rec_dict['heart_rate'])
            times.append(rec_dict['timestamp'])
    # df = pd.DataFrame(list(zip(times, hrs, pwrs, cads)), columns=['time', 'hr', 'pwr', 'cad'])
    return (hrs, pwrs, cads, stime)

def read_xml_file(fileName, rename=False):
# parses xml files
    full_file = os.path.abspath(os.path.join(fileName))
    dom = ElementTree.parse(full_file)
    loc = '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}'

    root = dom.getroot()
    timestamp = root[0][0][0].text                  # Gets the datetime from file
    timestamp = timestamp.replace(':', '_')[:-1]
    if rename:
        folder = fileName[:fileName.rfind('/') + 1]
        os.rename(fileName, folder+timestamp+'.xml')    # renames the file to the datestamp
    datestring = timestamp[:10]
    dt = datetime.strptime(datestring, '%Y-%m-%d')

    hr = []
    pwr = []
    cad = []

    for trackpoints in dom.iter(loc + 'Trackpoint'):
        try:
            cur_cad = int(trackpoints.find(loc + 'Cadence').text)
        except AttributeError:
            cur_cad = -1
        try:
            rate = trackpoints.find(loc + 'HeartRateBpm')
            cur_hr = int(rate.find(loc + 'Value').text)
        except AttributeError:
            cur_hr = -1
        extensions = trackpoints.find(loc + 'Extensions')
        try:
            TPX = extensions.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX')
            cur_pwr = int(TPX.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts').text)
        except AttributeError:
            cur_pwr = -1
        # getting rid of dropout from the power meter
        if cur_pwr > 0 and cur_hr > 0 and cur_cad > 0:
            cad.append(cur_cad)
            hr.append(cur_hr)
            pwr.append(cur_pwr)
    data = np.hstack((np.reshape(cad, (len(cad), 1)), np.reshape(pwr, (len(pwr), 1)), np.reshape(hr, (len(hr), 1))))
    # Remove erroneous power data
    # data = [time for time in data if time[2] !=0 ]
    cad_data = np.array([i[0] for i in data])
    pwr_data = np.array([i[1] for i in data])
    hr_data = np.array([i[2] for i in data])
    return (hr_data, cad_data, pwr_data, dt)



#(hr_data, cad_data, pwr_data, dt) = read_xml_file('../data/2019-12-07T18_54_21.xml')
(hr_data, cad_data, pwr_data, dt) = parse_fit_file('../data/2075244199.fit')