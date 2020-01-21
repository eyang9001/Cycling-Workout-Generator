import matplotlib.pyplot as plt
from matplotlib import gridspec
from autograd import numpy as np 
import copy
from collections import OrderedDict

# plot a pair of input (seq1) output (seq2) pairs
def plot_setpoints(seq1):   
    # initialize figure
    fig = plt.figure(figsize = (10,3))
 
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1,1) 
    ax1 = plt.subplot(gs[0]);
 
    # plot
    ax1.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'k',linewidth = 2.5,linestyle ='--')
 
    # label axes and title
    ax1.set_title('set point sequence')
    ax1.set_xlabel('step')
    
    # set viewing limits
    s1min = np.min(copy.deepcopy(seq1))
    s1max = np.max(copy.deepcopy(seq1))
    s1gap = (s1max - s1min)*0.1
    s1min -= s1gap
    s1max += s1gap
    ax1.set_ylim([s1min,s1max])
 
    plt.show()
   
def plot_multiple_sequences(seq1,seq2,seq3):            
    # initialize figure
    fig = plt.figure(figsize = (10,5))
 
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(2,1) 
    ax1 = plt.subplot(gs[1]); 
    ax2 = plt.subplot(gs[0]);

    ax1.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'k',linewidth = 2.5)
    ax2.plot(np.arange(np.size(seq2)),seq2.flatten(),c = 'lime',linewidth = 2.5,label = 'sequence 1',zorder = 2)
    ax2.plot(np.arange(np.size(seq3)),seq3.flatten(),c = 'm',linewidth = 2.5,label = 'sequence 2',zorder = 1)
 
    # label axes and title
    ax1.set_title('input sequence')
    ax1.set_xlabel('step')
    ax2.set_title('output sequences')
    ax2.set_xlabel('step')
     
    # set viewing limits
    s1min = np.min(copy.deepcopy(seq1))
    s1max = np.max(copy.deepcopy(seq1))
    s1gap = (s1max - s1min)*0.1
    s1min -= s1gap
    s1max += s1gap
    ax1.set_ylim([s1min,s1max])
     
    s2min = np.min(copy.deepcopy(seq2))
    s2max = np.max(copy.deepcopy(seq2))
    s2gap = (s2max - s2min)*0.1
    s2min -= s2gap
    s2max += s2gap
    ax2.legend(loc = 1)
 
    plt.show()
    
# plot a pair of input (seq1) output (seq2) pairs
def plot_pair(seq1,seq2):   
    # initialize figure
    fig = plt.figure(figsize = (10,5))
 
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(2,1) 
    ax1 = plt.subplot(gs[1]); 
    ax2 = plt.subplot(gs[0]);
 
    # plot
    ax1.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'k',linewidth = 2.5)
    ax2.plot(np.arange(np.size(seq2)),seq2.flatten(),c = 'k',linewidth = 2.5)
 
    # label axes and title
    ax1.set_title('input sequence')
    ax1.set_xlabel('step')
    ax2.set_title('output sequence')
    ax2.set_xlabel('step')
     
    # set viewing limits
    s1min = np.min(copy.deepcopy(seq1))
    s1max = np.max(copy.deepcopy(seq1))
    s1gap = (s1max - s1min)*0.1
    s1min -= s1gap
    s1max += s1gap
    ax1.set_ylim([s1min,s1max])
     
    s2min = np.min(copy.deepcopy(seq2))
    s2max = np.max(copy.deepcopy(seq2))
    s2gap = (s2max - s2min)*0.1
    s2min -= s2gap
    s2max += s2gap
    ax2.set_ylim([s2min,s2max])
 
    plt.show()
    
    
def plot_3fer(seq1,seq2,seq3,**kwargs):            
    # initialize figure
    fig = plt.figure(figsize = (10,5))
 
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(2,1) 
    ax1 = plt.subplot(gs[1]); 
    ax2 = plt.subplot(gs[0]);
 
    # scatter
    ax1.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'k',linewidth = 3.5)
    ax1.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'r',linewidth = 2.5,label='actions')
    
    
    ax2.plot(np.arange(np.size(seq3)),seq3.flatten(),c = 'b',linewidth = 2.5,label = 'states',zorder = 0)
    ax2.plot(np.arange(np.size(seq2)),seq2.flatten(),c = 'k',linestyle='--',linewidth = 2.5,label = 'set points',zorder = 1)
 
    # label axes and title
    ax1.set_title('actions')
    ax1.set_xlabel('step')
    ax2.set_title('states/set points')
    ax2.set_xlabel('step')
     
    # set viewing limits
    s1min = np.min(copy.deepcopy(seq1))
    s1max = np.max(copy.deepcopy(seq1))
    s1gap = (s1max - s1min)*0.1
    s1min -= s1gap
    s1max += s1gap
    ax1.set_ylim([s1min,s1max])
     
    s2min = np.min(copy.deepcopy(seq2))
    s2max = np.max(copy.deepcopy(seq2))
    s2gap = (s2max - s2min)*0.1
    s2min -= s2gap
    s2max += s2gap
    
    ax1.legend()
    ax2.legend()
    
    #ax2.legend(loc = 1)
    #ax2.set_ylim([s2min,s2max])
 
    plt.show()