import os
import seaborn as sns; sns.set()
from autograd import value_and_grad
from autograd.misc.flatten import flatten_func
# import autograd functionality
import autograd.numpy as np

from parsers import *
from conv_files import *


# Creates a dataset in the format [hr, cad, pwr, next_hr]
def split_data(hr, cad, pwr):
    dataset = []
    for i in range(len(hr) - 1):
        dataset.append([hr[i], cad[i], pwr[i], hr[i + 1]])
    return dataset


# Reads in data from all data files in data folder
def read_all_files(data_folder):
    dataset = []
    for item in os.listdir(data_folder):
        if item.endswith(".xml"):
            #             print(item)
            (hr, pwr, cad) = readin(data_folder + item)
            dataset = dataset + split_data(hr, pwr, cad)
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
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x, axis=1)[:, np.newaxis]
    x_stds = np.std(x, axis=1)[:, np.newaxis]

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


def plot_data_and_model(hr, cad, pwr, trained_model):
    # produce figure, plot data
    figure = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    x = np.arange(0, np.shape(hr)[0])
    plt.plot(x, hr, color='blueviolet', label='Original')
    # show fit from trained model
    nhr = trained_model(hr[:-1], cad[:-1], pwr[:-1])
    nhr = hr_inverse_normalizer(nhr)
    cost = np.sum((nhr[:, :] - hr[1:]) ** 2) / nhr.size
    plt.plot(np.arange(0, np.shape(nhr)[1]), nhr[0], c='b', linestyle='--', label='Prediction')
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
    cost = np.sum((hr1[:, :-1] - hr[:, 1:]) ** 2)
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

