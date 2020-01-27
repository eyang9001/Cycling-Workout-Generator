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
    try:
        for record in fitfile.get_messages('record'):
            rec_dict = record.get_values()
            if stime == None:
                stime = rec_dict['timestamp'] #stores the initial datetime for the ride
                if rename:
                    new_fname = stime.strftime("%Y-%m-%dT%H_%M_%S")+'.fit'      # renames the file to the datetime
                    folder = file_path[:file_path.rfind('/') + 1]
                    os.rename(file_path, folder+new_fname)
            if ('power' in rec_dict) & ('cadence' in rec_dict) & ('heart_rate' in rec_dict):
                if (rec_dict['power'] is not None) & (rec_dict['cadence'] is not None) & (rec_dict['heart_rate'] is not None): # only include timestamps that have all 3 values
                    pwrs.append(rec_dict['power'])
                    cads.append(rec_dict['cadence'])
                    hrs.append(rec_dict['heart_rate'])
                    times.append(rec_dict['timestamp'])
    except AttributeError: # for corrupt files
        pass
    return (np.array(hrs), np.array(pwrs), np.array(cads), stime)

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
        except AttributeError: # if there is no cadence for this record, don't save
            cur_cad = -1
        try:
            rate = trackpoints.find(loc + 'HeartRateBpm')
            cur_hr = int(rate.find(loc + 'Value').text)
        except AttributeError: # if there is no heart rate for this record, don't save
            cur_hr = -1
        extensions = trackpoints.find(loc + 'Extensions')
        try:
            TPX = extensions.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX')
            cur_pwr = int(TPX.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts').text)
        except AttributeError: # if there is no power for this record, don't save
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
    return (hr_data, pwr_data, cad_data, dt)

def parse_store_all(folderpath='../data/', save_prefix='all_data', clean=True):
    # This parses all the .fit and .xml files in a folder and saves them as a .npy array
    # Clean will clean the data
    hrs = []
    pws = []
    cads = []
    dates = []
    count = 0
    for item in os.listdir(folderpath):
        print(item)
        if item.endswith('.xml'):
            try:
                (hr_data, pwr_data, cad_data, dt) = read_xml_file(folderpath + item)
            except AttributeError:
                pass
        if item.endswith('.fit'):
            try:
                (hr_data, pwr_data, cad_data, dt) = parse_fit_file(folderpath + item)
            except AttributeError:
                pass
        if len(hr_data) > 0:        # only save if the file stored all hr, cad and power
            print(count)
            if clean:
                df = pd.DataFrame()
                df['HR'] = hr_data
                df['CAD'] = cad_data
                df['PWR'] = pwr_data
                # deletes instances of heart rate below resting heart rate
                df = df[df['HR'] > 75]

                hr_data = np.array(df['HR'])
                cad_data = np.array(df['CAD'])
                pwr_data = np.array(df['PWR'])

            hrs.append(hr_data)
            cads.append(cad_data)
            pws.append(pwr_data)
            dates.append(dt)
            count += 1

    # Sort by date
    df = pd.DataFrame()
    df['Date'] = dates
    df['HR'] = hrs
    df['CAD'] = cads
    df['PWR'] = pws
    df = df.sort_values(by=['Date'])

    # Save to .npy files
    np.save(folderpath + save_prefix + '_hr.npy', np.array(df['HR']))
    np.save(folderpath + save_prefix + '_cad.npy', np.array(df['CAD']))
    np.save(folderpath + save_prefix + '_pwr.npy', np.array(df['PWR']))
    np.save(folderpath + save_prefix + '_dates.npy', np.array(df['Date']))

if __name__ == "__main__":
    #(hr_data, cad_data, pwr_data, dt) = read_xml_file('../data/2019-12-07T18_54_21.xml')
    # (hr_data, cad_data, pwr_data, dt) = parse_fit_file('../data/2751540892.fit') # This was the troublesome file
    # (hr_data, cad_data, pwr_data, dt) = parse_fit_file('../data/2019-12-23T00_01_43.fit')
    # print(hr_data)
    # File 2019-08-31T12_10_14 deleted because of bad hr data
    # File 2019-12-23T00_01_43 also deleted because of bad hr data
    # File 2019-09-17T21_41_02 deleted bad hr data
    # FIle 2019-06-30T12_13_11 bad hr

    parse_store_all(save_prefix='all_data')