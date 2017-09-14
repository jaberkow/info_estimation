import numpy as np
from numpy.random import choice
import conversion_utilities as cu
from scipy.special import entr
from scipy.stats import linregress
import timeit
import itertools

def block_binarizer(data_block):
    """
    Converts a block with possible non-binary entries into a resample block of binary spike trains
    
    Inputs:
        data_block - data_block[t,i,j] is the number of spikes that neuron j output for stimulus t in repeat i
    Returns:
        binarized_block - resampled binary block.  Last two dimensions are the same as for data_block but first will increase
        by a factor of the maximum value of data_block
    
    """
    
    max_number = np.amax(data_block)
    (num_stimuli_orig,num_repeats,num_neurons) = np.shape(data_block)
    if max_number <= 1:
        #nothing to be done, all patterns are binary
        return data_block
    else:
        #resample each time stimuli response into max_number sub bins
        num_stimuli = max_number * num_stimuli_orig
        sub_block = np.zeros((max_number,num_repeats,num_neurons))
        ind_possible = np.arange(max_number)
        sub_block_list = []
        for t in range(num_stimuli_orig):
            sub_block *= 0.0
            for i in range(num_repeats):
                for j in range(num_neurons):
                    original_num = data_block[t,i,j]
                    ind_list = choice(ind_possible,size=original_num,replace=False)
                    sub_block[ind_list,i,j] += 1
            sub_block_list.append(np.copy(sub_block))
        binarized_block = np.concatenate(sub_block_list,axis=0)
        return np.sign(binarized_block)

def get_binary_inds(data_block):
    """
    Converts a block of binary response vectors into sample indicators over the set of 2^N patterns
    
    Inputs:
        data_block - A binary tensor of shape (num_stimuli,num_repeats,num_neurons), where data_block[t,i,j] = 1 if
        neuron j spiked in time interval in time interval t on repeat i and 0 otherwise.
    Ouputs:
        samples - A tensor of shape (num_stimuli,num_repeats) where samples[t,i] is the integer index (between 0 and
        2^num_neurons - 1) of the binary pattern observed at time interval t and repeat i.
    """
    
    clipped_data_block = np.clip(data_block,0,1.0)
    (num_stimuli_orig,num_repeats,num_neurons) = np.shape(clipped_data_block)
    weights = 2**np.arange(num_neurons)
    
    return np.sum(clipped_data_block*weights[np.newaxis,np.newaxis,:],axis=2).astype(int)

def get_spike_count_inds(data_block):
    """
    Converts a block of spike response vectors into sample indicators over the set of population spike counts
    
    Inputs:
        data_block - A tensor of shape (num_stimuli,num_repeats,num_neurons),  data_block[t,i,j] 
        is the number of spikes that neuron j output for stimulus t in repeat i
    Ouputs:
        samples - A tensor of shape (num_stimuli,num_repeats) where samples[t,i] is the integer index of the spike count.  Spike
        count state indices are conveniently the same as their value
    """
    
    return np.sum(data_block,axis=2).astype(int)


def get_2d_hist_inds(data_block,w,lp_transform=False,Nx_bins=10,Ny_bins=10):
    """
    Converts a block of spike response vectors into sample indicators over the set of values of a discretized two dimensional vector.
    The two dimensional vector is first formed as np.dot(data_block,w).  These vectors are then optionally passed through a log-polar
    transform.  Either way the results are then binned with uniform cartesian bins along the range of their x and y values with Nx_bins
    along the x-axis and Ny_bins along the y-axis
    
    Inputs:
        data_block - A tensor of shape (num_stimuli,num_repeats,num_neurons),  data_block[t,i,j] 
        is the number of spikes that neuron j output for stimulus t in repeat i
        w - a matrix of shape (num_neurons,2), the two-dimensional weight vectors.  w[i,:] is the weight vector for neuron i.
        lp_transform - Whether or not to perform a log-polar transform on the two-dimensional vectors
        Nx_bins - Number of bins along x-axis
        Ny_bins - Number of bins along y-axis
    Ouputs:
        samples - A tensor of shape (num_stimuli,num_repeats) where samples[t,i] is the integer index of the spike count.  Spike
        count state indices are conveniently the same as their value
    """
    
    
    
    
    
    