import copy
import seaborn as sns; sns.set()
from autograd.misc.flatten import flatten_func

# import autograd functionality
import autograd.numpy as np

from parsers import *
from conv_files import *

# least_squares using dataset input
def least_squares_set(w,hr,cad,pwr,hr_next):
    hr1 = model(w,hr,cad,pwr)
    cost = np.sum((hr1[0]-hr_next[0])**2)
    return cost/hr.size

def test_predict(s_hr, cad, pwr, trained_model):
    hr = [s_hr]
    for i in range(len(cad)-1):
        next_hr = trained_model(hr[i], cad[i], pwr[i])[0]
        next_hr = hr_inverse_normalizer(next_hr)
#         print(next_hr)
        hr.append(next_hr)
    return np.asarray([hr])

def test_predict_steps(s_hr, cad, pwr, steps, trained_model):
    hr_in = [[s_hr] * steps]
    hr = [s_hr]
    for i in range(len(cad)-1):
        next_hr = trained_model(np.transpose(hr_in[i]), np.transpose(cad[i]), np.transpose(pwr[i]))[0][0]
        next_hr = hr_inverse_normalizer(next_hr)[0][0]
        hr.append(next_hr)
        hr_in.append(hr_in[i][1:] + [next_hr])
    return np.asarray([hr])


# zero order coordinate search
def coordinate_descent(g, w_init, alpha, max_its, max_pwr, verbose):
    # flatten the input function
    g_flat, unflatten, w = flatten_func(g, w_init)

    # record history
    w_hist = [unflatten(w)]
    cost_eval = g_flat(w)
    train_hist = [cost_eval]

    # start loop
    N = np.size(w)
    for k in range(1, max_its + 1):
        # random shuffle of coordinates
        c = np.random.permutation(N)

        # forming the direction matrix out of the loop
        train_cost = train_hist[-1]

        # loop over each coordinate direction
        for n in range(N):
            direction = np.zeros((N, 1)).flatten()
            direction[c[n]] = 1

            # evaluate all candidates
            evals = [g_flat(w + alpha * direction)]
            evals.append(g_flat(w - alpha * direction))
            evals = np.array(evals)

            # if we find a real descent direction take the step in its direction
            ind = np.argmin(evals)
            if evals[ind] < train_cost:
                # take step
                w = w + ((-1) ** (ind)) * alpha * direction
                train_cost = evals[ind]

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)

        # print update
        if verbose == True:
            print('step ' + str(k + 1) + ' complete, train cost = ' + str(np.round(train_hist[-1], 4)[0]))

    # print update and return
    if verbose == True:
        print('finished all ' + str(max_its) + ' steps')
    return w_hist, train_hist


def PID_controller(e_t, h_t, d_t, w):
    # note here in terms of inputs
    # e_t = current error
    # h_t = integral of error
    # d_t = derivative of error
    return w[0] + w[1] * e_t + w[2] * h_t + w[3] * d_t


# loop for evaluating control model over all input/output action/state pairs
# Our inputs here:
# s_1 - the initial condition state
# x - sequence of training set points
# w - the control model parameters
def control_loop(x, w, trained_model, max_pwr):
    # initialize key variables and containers
    hr_t = copy.deepcopy(hr_1)
    h_t = 0
    d_t = 0
    frac = 1 / float(np.size(x))
    pwr_history = []
    hr_history = [hr_t]
    error_history = []

    # loop over training set points and run through controller, then
    # system models
    for t in range(np.size(x) - 1):
        # get current set point
        x_t = x[t]

        # update error
        e_t = x_t - hr_t
        error_history.append(e_t)

        # update integral of error
        h_t = h_t + frac * e_t

        # update derivative of error
        if t > 0:
            d_t = frac * (error_history[-1] - error_history[-2])

        # send error, integral, and derivative to PID controller
        pwr_t = PID_controller(e_t, h_t, d_t, w)

        # clip power to match my power abilities
        if pwr_t > max_pwr:
            pwr_t = max_pwr
        elif pwr_t < 0:
            pwr_t = 0

        # send action to system model
        # Assuming cadence is maintained at 95
        hr_t = trained_model(hr_t, 95, pwr_t)
        hr_t = hr_inverse_normalizer(hr_t)[0][0]

        # store state output, and actions (for plotting)
        hr_history.append(hr_t)
        pwr_history.append(pwr_t)

    # transition to arrays
    hr_history = np.array(hr_history)[np.newaxis, :]
    pwr_history = np.array(pwr_history)[np.newaxis, :]

    # return velocities and control history
    return hr_history, pwr_history


# an implementation of the least squares cost for PID controller tuning
# note here: s is an (1 x T) array and a an (1 x T-1) array
def least_squares_PID(w, x, trained_model, max_pwr):
    # system_loop - runs over all action-state pairs and produces entire
    # state prediction set
    hr_history, pwr_history = control_loop(x, w, trained_model, max_pwr)

    # compute least squares error between real and predicted states
    cost = np.sum((hr_history[0][1:] - x[1:]) ** 2)
    return cost / float(x.shape[0] - 1)