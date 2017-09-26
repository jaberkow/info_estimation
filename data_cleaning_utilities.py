import numpy as np
from numpy.random import choice
import conversion_utilities as cu
from scipy.special import entr
from scipy.stats import linregress
import timeit
import itertools
import pyentropy
from pyentropy import DiscreteSystem,SortedDiscreteSystem

def pattern_maker(N):
    """
    Outputs the arrays holding all the binary patterns
    
    Inputs:
        N - The number of neurons
    Outputs:
        P_unsigned - The array holding all the {0,1} patterns.  Of shape (2^N,N)
        P_signed - The {-1,1} version
    """
    
    lst = map(list, itertools.product([0, 1], repeat=N))
    P_unsigned = np.array(lst)
    #P_signed = 2.0*P_unsigned - 1.0
    return P_unsigned

def block_binarizer(data_block):
    """
    Converts a block with possible non-binary entries into a resample block of binary spike trains
    
    Inputs:
        data_block - data_block[t,i,j] is the number of spikes that neuron j output for stimulus t in repeat i
    Outputs:
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
    Outputs:
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
    Outputs:
        samples - A tensor of shape (num_stimuli,num_repeats) where samples[t,i] is the integer index of the spike count.  Spike
        count state indices are conveniently the same as their value
    """
    
    return np.sum(data_block,axis=2).astype(int)


def get_2d_hist_inds(data_block,w,lp_transform=False,Nx_bins=10,Ny_bins=10):
    """
    Converts a block of spike response vectors into sample indicators over the set of values of a discretized two dimensional vector.
    The two dimensional vector is first formed as np.dot(data_block,w).  These vectors are then optionally passed through a log-polar
    transform.  Either way the results are then binned with uniform cartesian bins along the range of their x and y values with
    Nx_bins along the x-axis and Ny_bins along the y-axis
    
    Inputs:
        data_block - A tensor of shape (num_stimuli,num_repeats,num_neurons),  data_block[t,i,j] 
        is the number of spikes that neuron j output for stimulus t in repeat i.  Doesn't need to be binary, can be signed (+1,-1)
        w - a matrix of shape (num_neurons,2), the two-dimensional weight vectors.  w[i,:] is the weight vector for neuron i.
        lp_transform - Whether or not to perform a log-polar transform on the two-dimensional vectors
        Nx_bins - Number of bins along x-axis
        Ny_bins - Number of bins along y-axis
    Outputs:
        samples - A tensor of shape (num_stimuli,num_repeats) where samples[t,i] is the integer index of the spike count.  Spike
        count state indices are conveniently the same as their value
    """
    
    vector_samples = np.dot(data_block,w)
    if lp_transform:
        new_x_vals = 0.5*np.log(vector_samples[:,:,0]**2 + vector_samples[:,:,1]**2)
        new_y_vals = np.arctan2(vector_samples[:,:,1],vector_samples[:,:,0])
        vector_samples = np.stack([new_x_vals,new_y_vals],axis=2)
    
    max_x = np.amax(vector_samples[:,:,0])
    min_x = np.amin(vector_samples[:,:,0])
    max_y = np.amax(vector_samples[:,:,1])
    min_y = np.amin(vector_samples[:,:,1])
    
    bin_x_vals = np.linspace(min_x,max_x,Nx_bins + 1)
    bin_y_vals = np.linspace(min_y,max_y,Ny_bins + 1)
    
    bin_x_vals[-1] += 1e-5
    bin_y_vals[-1] += 1e-5
    
    num_bins = Nx_bins * Ny_bins
    
    x_hashes = np.digitize(vector_samples[:,:,0],bin_x_vals) - 1
    y_hashes = np.digitize(vector_samples[:,:,1],bin_y_vals) - 1
    
    samples = x_hashes + Nx_bins*y_hashes
    return samples
    
def get_w(beta,theta):
    """
    Get matrix of weight vectors from set of preferred orientations (theta) and neural sensitivities (beta).
    
    Inputs:
        beta - Vector of neural sensitivities.  Shape = (,number of neurons)
        theta - Vector of preferred orientations.  Shape = (,number of neurons)
    Outputs:
        w - Matrix of weight vectors.  Shape = (number of neurons,2)
    """
    
    w_x = beta*np.cos(theta)
    w_y = beta*np.sin(theta)
    return np.stack([w_x,w_y],axis=1)

def get_worst_case_theta(theta,delta):
    """
    Get a realization of the preferred orientations under the worst case given the "mean values" (theta) and "uncertainties" (delta)
    
    Inputs:
        theta - Vector of preferred orientations.  Shape = (,number of neurons)
        delta - Vector of orientation uncertainties.  Shape = (,number of neurons)
    Outputs:
        theta_small - Vector of worst case reduced resolutions.  Shape = (,number of neurons)
    """
    
    iters = 0
    num_neur = np.size(theta)
    theta_temp = np.copy(theta)
    delta_temp = np.copy(delta)
    not_converged = True
    while  not_converged and iters < 40:
        #get individual vectors
        iters += 1
        #print theta_temp
        x = np.cos(theta_temp)
        y = np.sin(theta_temp)
        #print x,y
        norms = np.sqrt(x**2 + y**2)
        #print norms
        temp = x[np.newaxis,:]*x[:,np.newaxis] + y[np.newaxis,:]*y[:,np.newaxis]
        #print temp/(norms[np.newaxis,:]*norms[:,np.newaxis])
        theta_diff = np.arccos(np.clip(temp/(norms[np.newaxis,:]*norms[:,np.newaxis]),-1.0,1.0))
        #print theta_diff
        denominator = 0.5*(delta_temp[np.newaxis,:] + delta_temp[:,np.newaxis])
        #print denominator
        values = theta_diff/denominator
        #print values
        values[values==0.0] += 400.0
        #print values
        min_val = np.amin(values)
        #print min_val
        if min_val > 1:
            not_converged= False
        else:
            (ind1,ind2) = np.unravel_index(np.argmin(values),(num_neur,num_neur))
            #print ind1,ind2
            theta1,theta2 = theta_temp[[ind1,ind2]]
            #print theta1,theta2
            theta1_inds = np.where(theta_temp==theta1)
            theta2_inds = np.where(theta_temp==theta2)
            size1 = float(np.size(theta1_inds))
            size2 = float(np.size(theta2_inds))
            vec_x = size1*np.cos(theta1) + size2*np.cos(theta2)
            vec_y = size1*np.sin(theta1) + size2*np.sin(theta2)
            #print vec_x, vec_y
            new_theta = np.arctan2(vec_y,vec_x)
            new_delta =  (np.sum(delta_temp[theta1_inds]) + np.sum(delta_temp[theta2_inds]))/ (size1 + size2)
            #print new_theta
            #print "merging {} and {} at iter {}".format(deg1_inds+1,deg2_inds+1,iters)
            theta_temp[theta1_inds] = new_theta
            theta_temp[theta2_inds] = new_theta
            delta_temp[theta1_inds] = new_delta
            delta_temp[theta2_inds] = new_delta
            #break
    return theta_temp

def get_exact_vector_inds_map(w):
    """
    Get a mapping that maps binary vectors to indices of the vector valued variable.  This function is made separate
    for debugging purposes.
    
    Inputs:
        w - The matrix of weight vectors.  Of shape (number of neurons,D).
    Outputs:
        ind_map - Vector of length (2^number_of_neurons) where ind_map[i] is the the index of the vector pattern applied to
        the i^th firing rate pattern.
    """
    
    N = np.shape(w)[0]
    pattern = pattern_maker(N)
    s_pattern = 2.0*pattern - 1.0
    
    vectors = np.dot(s_pattern,w)
    
    list_of_u = []
    ind_map = np.zeros(2**N)
    for i in range(2**N):
        current_u = np.copy(vectors[i,:])
        if list(current_u) in list_of_u:
            ind_map[i] = list_of_u.index(list(current_u))
        else:
            list_of_u.append(list(current_u))
            ind_map[i] = len(list_of_u) - 1
    
    return ind_map.astype(int)

def get_exact_vector_inds(binary_inds,w):
    """
    Converts a block of binary response indicators into indicators over different states of the two-dimensional vector
    
    Inputs:
        binary_inds - Sample indicators over the binary response patters.  Shape is (num_stimuli,num_repeats)
        beta - the vector of neural sensitives.  Shape is (num neurons)
        theta - the ector of preferred orientations.  Shape is (num neurons)
    Outputs:
        vector_samples 
    """
    
    my_map = get_exact_vector_inds_map(w)
    
    vector_inds = my_map[binary_inds]
    
    return vector_inds

def compute_information(sample_inds,to_bps,compress_states=False,est_method='pt'):
    """
    Computes the mutual information from the block of samples
    
    Inputs:
        sample_inds - The samples, represented as integer valued indicators over the different states
        to_bps - The conversion from bits to bits per second.  How many samples are taken per second essentially.
        compress_states - Boolean.  If true, remap the indicators so that only states that are actually observed are
        considered.  Useful if est_method = 'pt' or 'nsb' and many of the states are not actually possible.
        method - The information estimation method.  Choices are 'qe', 'pt', or 'nsb'.  Note that 
        nsb is not recommended due to time requirements.
    Outputs:
        Information - The estimate of mutual information, in bits per second
    """
    
    (num_stimuli,num_reps) = np.shape(sample_inds)
    reshaped_data = np.concatenate(sample_inds.tolist())
    
    if compress_states:
        temp = np.copy(reshaped_data)
        seen_states,reshaped_data np.unique(temp,return_inverse=True)
    
    num_states = np.amax(reshaped_data) + 1
    
    Ny = np.zeros(num_stimuli) + num_reps
    Ny.astype(int)
    s = SortedDiscreteSystem(reshaped_data,(1,num_states),num_stimuli,Ny)
    s.calculate_entropies(method=est_method, calc=['HX', 'HXY'])
    return s.I()*to_bps