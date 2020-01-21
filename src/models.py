import os
from xml.etree import ElementTree
import time
import copy
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import value_and_grad
from autograd.misc.flatten import flatten_func
import some_plotters as plotter
import datetime as datetime
from xml.etree import ElementTree
from fitparse import FitFile

def readin(fileName):
    full_file = os.path.abspath(os.path.join(fileName))
    dom = ElementTree.parse(full_file)
    loc = '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}'

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
    return (hr_data, cad_data, pwr_data)


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
                stime = rec_dict['timestamp']  # stores the initial datetime for the ride
                if rename:
                    new_fname = stime.strftime("%Y-%m-%dT%H_%M_%S") + '.fit'  # renames the file to the datetime
                    folder = file_path[:file_path.rfind('/') + 1]
                    os.rename(file_path, folder + new_fname)
            if ('power' in rec_dict) & ('cadence' in rec_dict) & ('heart_rate' in rec_dict):
                if (rec_dict['power'] is not None) & (rec_dict['cadence'] is not None) & (
                        rec_dict['heart_rate'] is not None):  # only include timestamps that have all 3 values
                    pwrs.append(rec_dict['power'])
                    cads.append(rec_dict['cadence'])
                    hrs.append(rec_dict['heart_rate'])
                    times.append(rec_dict['timestamp'])
    except AttributeError:  # for corrupt files
        pass
    return (np.array(hrs), np.array(pwrs), np.array(cads), stime)


def read_xml_file(fileName, rename=False):
    # parses xml files
    full_file = os.path.abspath(os.path.join(fileName))
    dom = ElementTree.parse(full_file)
    loc = '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}'

    root = dom.getroot()
    timestamp = root[0][0][0].text  # Gets the datetime from file
    timestamp = timestamp.replace(':', '_')[:-1]
    if rename:
        folder = fileName[:fileName.rfind('/') + 1]
        os.rename(fileName, folder + timestamp + '.xml')  # renames the file to the datestamp
    datestring = timestamp[:10]
    dt = datetime.strptime(datestring, '%Y-%m-%d')

    hr = []
    pwr = []
    cad = []

    for trackpoints in dom.iter(loc + 'Trackpoint'):
        try:
            cur_cad = int(trackpoints.find(loc + 'Cadence').text)
        except AttributeError:  # if there is no cadence for this record, don't save
            cur_cad = -1
        try:
            rate = trackpoints.find(loc + 'HeartRateBpm')
            cur_hr = int(rate.find(loc + 'Value').text)
        except AttributeError:  # if there is no heart rate for this record, don't save
            cur_hr = -1
        extensions = trackpoints.find(loc + 'Extensions')
        try:
            TPX = extensions.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX')
            cur_pwr = int(TPX.find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts').text)
        except AttributeError:  # if there is no power for this record, don't save
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


# Creates a dataset in the format [hr, cad, pwr, next_hr]
def split_data(hr, cad, pwr, dt):
    dataset = []
    for i in range(len(hr) - 1):
        dataset.append([hr[i], cad[i], pwr[i], hr[i + 1], dt])
    return dataset


# Reads in data from all data files in data folder
def read_all_files(data_folder):
    dataset = []
    for item in os.listdir(data_folder):
        if item.endswith(".xml"):
            #             print(item)
            try:
                (hr, pwr, cad, dt) = read_xml_file(data_folder + item)
                dataset = dataset + split_data(hr, pwr, cad, dt)
            except AttributeError:
                pass
        if item.endswith(".fit"):
            try:
                (hr, pwr, cad, dt) = parse_fit_file(data_folder + item)
                dataset = dataset + split_data(hr, pwr, cad, dt)
            except AttributeError:
                pass
    return dataset


### For modeling
# gradient descent
def gradient_descent(g, w_init, alpha, max_its, verbose):
    # flatten the input function
    g_flat, unflatten, w = flatten_func(g, w_init)

    # compute gradient of flattened input function
    # when evaluated this returns both the evaluation of the gradient and the original function
    grad = value_and_grad(g_flat)
    cost_eval, grad_eval = grad(w)
    grad_eval.shape = np.shape(w)

    # record history
    w_hist = [unflatten(w)]
    train_hist = [cost_eval]

    # gradient descent loop
    for k in range(max_its):
        # take descent step with momentum
        w = w - alpha * grad_eval

        # plug in updated w into func and gradient
        cost_eval, grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        # store updates
        w_hist.append(unflatten(w))
        train_hist.append(cost_eval)

        # print update
        if verbose == True:
            print('step ' + str(k + 1) + ' complete, train cost = ' + str(np.round(train_hist[-1], 4)[0]))

    # print update and return
    if verbose == True:
        print('finished all ' + str(max_its) + ' steps')
    return w_hist, train_hist


import matplotlib.pyplot as plt


def plot_series(cost_history, title='cost value', xlabel='iteration'):
    figure = plt.figure(figsize=(10, 3))
    plt.plot(cost_history)
    plt.xlabel(xlabel)
    plt.ylabel(title, rotation=90)
    plt.show()


# standard normalization function
def standard_normalizer(x, axis=1):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x, axis=axis)
    x_stds = np.std(x, axis=axis)

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10 ** (-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # create standard normalizer function
    normalizer = lambda data: (data - x_means) / x_stds

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data * x_stds + x_means

    # return normalizer 
    return normalizer, inverse_normalizer


def plot_data(hr, pwr, cad, compare=None):
    figure = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    x = np.arange(0, np.shape(hr)[0])
    if compare is not None:
        plt.plot(x, compare, color='blueviolet', label='Correct')
        plt.plot(x, hr, c='b', linestyle='--', label='Prediction')
        plt.legend()
    else:
        plt.plot(x, hr, color='blueviolet')
    plt.title('Heart Rate')
    plt.subplot(3, 1, 2)
    plt.plot(x, cad, color='c')
    plt.title('Cadence')
    plt.subplot(3, 1, 3)
    plt.plot(x, pwr, color='b')
    plt.title('Power')


def plot_data_and_model(hr, cad, pwr, trained_model, rev_norm):
    # produce figure, plot data
    figure = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    x = np.arange(0, np.shape(hr)[0])
    plt.plot(x, hr, color='blueviolet', label='Original')
    # show fit from trained model
    nhr = trained_model(hr[:-1], cad[:-1], pwr[:-1])
    nhr = rev_norm(nhr)
    cost = np.sum((nhr[:] - hr[1:]) ** 2) / nhr.size
    plt.plot(np.arange(0, len(nhr)), nhr, c='b', linestyle='--', label='Prediction')
    plt.title('Heart Rates')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x, cad)
    plt.title('Cadence')

    plt.subplot(3, 1, 3)
    plt.plot(x, pwr)
    plt.title('Power')
    plt.show()
    return nhr, cost


### For the different models

# Using a linear model with only the previous one timestep
def model(w, hr, cad, pwr):
    hrnext = w[0] + w[1] * hr + w[2] * cad + w[3] * pwr
    return hrnext


def least_squares(w, hr, cad, pwr):
    hr1 = model(w, hr, cad, pwr)
    cost = np.sum((hr1[:-1] - hr[1:]) ** 2)
    return cost / hr.size


# Using a SEQUENTIAL linear model where only the first HR value is given
def seq_model(w, hr_init, cad, pwr):
    hr = [hr_init]
    for i in range(len(cad[0]) - 1):
        next_hr = w[0] + w[1] * hr[i] + w[2] * cad[0][i] + w[3] * pwr[0][i]
        hr.append(next_hr)
    return np.asarray([hr])


def least_squares_seq(w, hr, cad, pwr):
    hr_pred = seq_model(w, hr[0][0], cad, pwr)
    cost = np.sum((hr[1:] - hr_pred[1:]) ** 2)
    return cost / hr.size


def predict_cost(orig, pred):
    cost = np.sum((orig - pred) ** 2)
    return cost / orig.size


# Pipeline using a lookback of 10 timesteps

# Takes in the data as separate channels and formats it in a dataset with each next heart rate 
# having a set # of steps worth of previous timesteps of data
def seq_dataset(hr, cad, pwr, steps):
    dataset = []
    hr_out = []
    cad_out = []
    pwr_out = []
    hr_next = []
    for i in range(steps, len(hr) - 1):
        c_hr = []
        c_cad = []
        c_pwr = []
        for ii in range(0, steps):
            c_hr.append(hr[i - (steps - ii)])
            c_cad.append(cad[i - (steps - ii)])
            c_pwr.append(pwr[i - (steps - ii)])
        hr_out.append(c_hr)
        cad_out.append(c_cad)
        pwr_out.append(c_pwr)
        hr_next.append(hr[i])
    dataset = [hr_out, cad_out, pwr_out, hr_next]
    return dataset


# Reads in data from all data files in data folder and formats it with many previous timesteps
def read_all_files_back(data_folder, steps):
    dataset = [[], [], [], []]
    for item in os.listdir(data_folder):
        if item.endswith(".xml"):
            (hr, pwr, cad) = readin(data_folder + item)
            fileset = seq_dataset(hr, pwr, cad, steps)
            dataset[0] = dataset[0] + fileset[0]
            dataset[1] = dataset[1] + fileset[1]
            dataset[2] = dataset[2] + fileset[2]
            dataset[3] = dataset[3] + fileset[3]
    return dataset


# Using a linear model with ten previous timesteps
def back_model(w, hr, cad, pwr, steps):
    hrnext = w[0] + np.dot(np.transpose(w[1:steps + 1]), np.transpose(hr)) + np.dot(
        np.transpose(w[steps + 1:1 + (steps * 2)]), np.transpose(cad)) + np.dot(
        np.transpose(w[(steps * 2) + 1:(steps * 3) + 1]), np.transpose(pwr))
    #     hrnext = w[0] + w[1]*hr[0] + w[2]*hr[1] + w[3]*hr[2] + w[4]*hr[3] + w[5]*hr[4] + w[6]*hr[5] + w[7]*hr[6] + w[8]*hr[7] + w[9]*hr[8] + w[10]*hr[9] +
    #     w[11]*cad[0] + w[12]*cad[1] + w[13]*cad[2] + w[14]*cad[3] + w[15]*cad[4] + w[16]*cad[5] + w[17]*cad[6] + w[18]*cad[7] + w[19]*cad[8] + w[20]*cad[9] +
    #     w[21]*pwr[0] + w[22]*pwr[1] + w[23]*pwr[2] + w[24]*pwr[3] + w[25]*pwr[4] + w[26]*pwr[5] + w[27]*pwr[6] + w[28]*pwr[7] + w[29]*pwr[8] + w[30]*pwr[9]
    return hrnext


def back_least_squares(w, hr, cad, pwr, hr_next, steps):
    hr1 = back_model(w, hr, cad, pwr, steps)
    cost = np.sum((hr1[0] - hr_next[0]) ** 2)
    return cost / hr.size

def model_all_data(data_folder):
    # Runs the whole pipeline for taking in all data and generating a linear model to predict the next heart rate
    dataset = read_all_files(data_folder)
    # Save the gathered data in a .npy file for later use
    np.save(data_folder + 'dataset.npy', np.array(dataset))


    dataset_t = np.transpose(dataset, (1, 0))
    hr = dataset_t[0]
    cad = dataset_t[1]
    pwr = dataset_t[2]
    hr_next = dataset_t[3]

    # least_squares using dataset input
    def least_squares_set(w, hr, cad, pwr, hr_next):
        hr1 = model(w, hr, cad, pwr)
        cost = np.sum((hr1 - hr_next) ** 2)
        return cost / hr.size

    hr_normalizer, hr_inverse_normalizer = standard_normalizer(hr, 0)
    cad_normalizer, cad_inverse_normalizer = standard_normalizer(cad, 0)
    pwr_normalizer, pwr_inverse_normalizer = standard_normalizer(pwr, 0)

    hr_normalized = hr_normalizer(hr)
    cad_normalized = cad_normalizer(cad)
    pwr_normalized = pwr_normalizer(pwr)
    hr_next_normalized = hr_normalizer(hr_next)

    g = lambda w, hr=hr_normalized, cad=cad_normalized, pwr=pwr_normalized, hr_next=hr_next_normalized: least_squares_set(w, hr, cad, pwr, hr_next)
    w_size = 4  # The number of weights
    w_init = 0.1 * np.random.randn(w_size, 1)
    max_its = 30
    alpha = 10 ** (-1)
    w_hist, train_hist = gradient_descent(g, w_init, alpha, max_its, verbose=True)

    # print out that cost function history plot
    plot_series(train_hist)

    # Get best weights and trained model
    ind = np.argmin(train_hist)
    w_best = w_hist[ind]
    g_best = train_hist[ind]
    print(w_best)

    all_data_model = lambda hr, cad, pwr, w=w_best: model(w, hr_normalizer(hr), cad_normalizer(cad), pwr_normalizer(pwr))

    return all_data_model, hr_inverse_normalizer