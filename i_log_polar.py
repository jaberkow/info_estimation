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


def get_maps(beta,theta,Nybins,Nxbins):
    """
    Let m(r) be the sufficient statistic m(\vec{r}) = \sum_i \beta_i r_i
    Let u(r) be the unweighted pooling u(\vec{r}) = \sum_i r_i
    
    Num_m is the number of unique values of m(r).  Num_u is the number of unique values of u(r).
    
    
    Inputs:
        beta:  The set of gains (weights)
        theta: The set of preferred orientations
        Nybins: The number of bins along the y axis
        Nxbins:  The number of bins along the x axis
    
    Returns:
        m_hashes:  Map from \vec{r} to bins of \vec{m} in log-polar space
        u_hashes:  Map from \vec{r} to bins of \vec{u} in log-polar space
    
    """
    
    
    N = np.size(beta)
    pattern = pattern_maker(N)
    s_pattern = 2.0*pattern - 1.0
    
    w_x = np.cos(theta)
    w_y = np.sin(theta)
    
    #get the cartesian coordinates
    m_x_vals_cart = np.sum(s_pattern*beta[np.newaxis,:]*w_x[np.newaxis,:],axis=1)
    m_y_vals_cart = np.sum(s_pattern*beta[np.newaxis,:]*w_y[np.newaxis,:],axis=1)
    u_x_vals_cart = np.sum(s_pattern*w_x[np.newaxis,:],axis=1)
    u_y_vals_cart = np.sum(s_pattern*w_y[np.newaxis,:],axis=1)
    
    #convert to log-polar coordinates
    m_x_vals = 0.5*np.log(m_x_vals_cart**2 + m_y_vals_cart**2)
    m_y_vals = np.arctan2(m_y_vals_cart,m_x_vals_cart)
    u_x_vals = 0.5*np.log(u_x_vals_cart**2 + u_y_vals_cart**2)
    u_y_vals = np.arctan2(u_y_vals_cart,u_x_vals_cart)
    
    max_m_x = np.amax(m_x_vals)
    max_m_y = np.amax(m_y_vals)
    min_m_x = np.amin(m_x_vals)
    min_m_y = np.amin(m_y_vals)
    
    
    max_u_x = np.amax(u_x_vals)
    max_u_y = np.amax(u_y_vals)
    min_u_x = np.amin(u_x_vals)
    min_u_y = np.amin(u_y_vals)
    
    mbin_x_vals = np.linspace(min_m_x,max_m_x,Nxbins + 1)
    mbin_y_vals = np.linspace(min_m_y,max_m_y,Nybins + 1)
    ubin_x_vals = np.linspace(min_u_x,max_u_x,Nxbins + 1)
    ubin_y_vals = np.linspace(min_u_y,max_u_y,Nybins + 1)
    
    mbin_x_vals[-1] += 0.0001
    mbin_y_vals[-1] += 0.0001
    ubin_x_vals[-1] += 0.0001
    ubin_y_vals[-1] += 0.0001
    
    num_hashes = Nybins * Nxbins
    
    m_x_hashes = np.digitize(m_x_vals,mbin_x_vals) - 1
    m_y_hashes = np.digitize(m_y_vals,mbin_y_vals) - 1
    u_x_hashes = np.digitize(u_x_vals,ubin_x_vals) - 1
    u_y_hashes = np.digitize(u_y_vals,ubin_y_vals) - 1
    
    #print u_x_hashes, u_y_hashes
    
    m_hashes = m_x_hashes + Nxbins*m_y_hashes
    u_hashes = u_x_hashes + Nxbins*u_y_hashes
    
    #print m_hashes, u_hashes
    
    return  m_hashes,u_hashes

def check_trivial(beta,theta,Nybins,Nxbins):
    N = np.size(beta)
    if(N==1):
        return 2,2,2
    else:
        m_hashes,u_hashes = get_maps(beta,theta,Nybins,Nxbins)
        unique_m = np.unique(m_hashes)
        unique_u = np.unique(u_hashes)
    return 2**N,np.size(unique_m),np.size(u_hashes)

def I_comp(data_block,m_hashes,u_hashes):
    """
    Computes Information w.r.t patterns, spike count, and 2-d binned m_hat and u_hat
    
    Inputs:
    data_block - np array, boolean.  Shape is (num_stimuli,repeats,num_neurons)
    m_hashes - What bin the pattern goes into
    u_hashes - same as above
    
    Ouputs:
    I_r - Information from patterns in bits
    I_u - information from spike count
    I_mbin - 
    I_ubin - 
    """
    
    #data_block = np.sign(data_block) #make sure it is binarized
    
    (num_stimuli,num_repeats,num_neurons) = np.shape(data_block)
    num_patterns = 2**num_neurons
    pattern_converter = 2**(np.arange(num_neurons))
    
    r_marg = np.zeros(num_patterns)
    r_cur = np.zeros(num_patterns)
    r_ce = 0.0
    
    u_marg = np.zeros(num_neurons + 1)
    u_cur = np.zeros(num_neurons + 1)
    u_ce = 0.0
    
    num_mhash = np.amax(m_hashes) + 1
    mhash_marg = np.zeros(num_mhash)
    mhash_cur = np.zeros(num_mhash)
    mhash_ce = 0.0
    
    num_uhash = np.amax(u_hashes) + 1
    uhash_marg = np.zeros(num_uhash)
    uhash_cur = np.zeros(num_uhash)
    uhash_ce = 0.0
    
    
    r_hits = np.sum(data_block*pattern_converter[np.newaxis,np.newaxis,:],axis=2).astype(int)
    u_hits = np.sum(data_block,axis=2).astype(int)
    mhash_hits = m_hashes[r_hits]
    uhash_hits = u_hashes[r_hits]
    
    for t in range(num_stimuli):
        #at every stimulus we count the number of times each value of r,u,m, or mbin is observed and form the empirical histogram
        r_y = np.bincount(r_hits[t,:])/float(num_repeats)
        r_ii = np.nonzero(r_y)[0]
        r_cur *= 0.0
        r_cur[r_ii] += r_y[r_ii]
        r_marg += r_cur
        r_ce += np.sum(entr(r_cur))
        
        u_y = np.bincount(u_hits[t,:])/float(num_repeats)
        u_ii = np.nonzero(u_y)[0]
        u_cur *= 0.0
        u_cur[u_ii] += u_y[u_ii]
        u_marg += u_cur
        u_ce += np.sum(entr(u_cur))
        
        mhash_y = np.bincount(mhash_hits[t,:])/float(num_repeats)
        mhash_ii = np.nonzero(mhash_y)[0]
        mhash_cur *= 0.0
        mhash_cur[mhash_ii] += mhash_y[mhash_ii]
        mhash_marg += mhash_cur
        mhash_ce += np.sum(entr(mhash_cur))
        
        uhash_y = np.bincount(uhash_hits[t,:])/float(num_repeats)
        uhash_ii = np.nonzero(uhash_y)[0]
        uhash_cur *= 0.0
        uhash_cur[uhash_ii] += uhash_y[uhash_ii]
        uhash_marg += uhash_cur
        uhash_ce += np.sum(entr(uhash_cur))
    
    #compute marginals
    r_marg /= float(num_stimuli)
    u_marg /= float(num_stimuli)
    mhash_marg /= float(num_stimuli)
    uhash_marg /= float(num_stimuli)
    
    #compute conditional entropy
    r_ce /= float(num_stimuli)
    u_ce /= float(num_stimuli)
    mhash_ce /= float(num_stimuli)
    uhash_ce /= float(num_stimuli)
     
    
    #marginal entropy
    r_H = np.sum(entr(r_marg))
    u_H = np.sum(entr(u_marg))
    mhash_H = np.sum(entr(mhash_marg))
    uhash_H = np.sum(entr(uhash_marg))
    
    
    #compute information in bits
    I_r = (r_H - r_ce)*np.log2(np.exp(1)) 
    I_u = (u_H - u_ce)*np.log2(np.exp(1))
    I_mhash = (mhash_H - mhash_ce)*np.log2(np.exp(1)) 
    I_uhash = (uhash_H - uhash_ce)*np.log2(np.exp(1))
    
    
    return I_r,I_u,I_mhash,I_uhash

def linear_extrap_I_full(raw_data_block,beta,theta,Nybins=10,Nxbins=10,number_of_tries=10):
    
    N_max = np.amax(raw_data_block)
    to_seconds = float(N_max)/0.03
    data_block = block_binarizer(raw_data_block)
    m_hashes,u_hashes = get_maps(beta,theta,Nybins,Nxbins)
    (num_stimuli,total_repeats,num_neurons) = np.shape(data_block)
    
    repeat_inds = np.arange(total_repeats)
    
    ideal_fractions = np.array([1.0,0.95,0.9,0.85,0.8])
    repeats_to_use = np.array(total_repeats*ideal_fractions).astype('int')
    real_fractions = repeats_to_use/float(total_repeats)
    regression_x = 1.0/real_fractions
    
    I_r_final_array = np.zeros_like(ideal_fractions)
    I_u_final_array = np.zeros_like(ideal_fractions)
    I_mhash_final_array = np.zeros_like(ideal_fractions)
    I_uhash_final_array = np.zeros_like(ideal_fractions)
    
    I_r_std_array = np.zeros_like(ideal_fractions)
    I_u_std_array = np.zeros_like(ideal_fractions)
    I_mhash_std_array = np.zeros_like(ideal_fractions)
    I_uhash_std_array = np.zeros_like(ideal_fractions)
    
    I_temp_r = np.zeros(number_of_tries)
    I_temp_u = np.zeros(number_of_tries)
    I_temp_mhash = np.zeros(number_of_tries)
    I_temp_uhash = np.zeros(number_of_tries)
    
    for f_ind in range(np.size(real_fractions)):
        num_repeats = repeats_to_use[f_ind]
        I_temp_r *= 0.0
        I_temp_u *= 0.0
        I_temp_mhash *= 0.0
        I_temp_uhash *= 0.0
        if num_repeats == total_repeats:
            #use all data
            I_r,I_u,I_mhash,I_uhash = I_comp(data_block,m_hashes,u_hashes)
            I_r_final_array[f_ind] = np.copy(I_r)
            I_u_final_array[f_ind] = np.copy(I_u)
            I_mhash_final_array[f_ind] = np.copy(I_mhash)
            I_uhash_final_array[f_ind] = np.copy(I_uhash)
            I_r_std_array[f_ind] = 0.0
            I_u_std_array[f_ind] = 0.0
            I_mhash_std_array[f_ind] = 0.0
            I_uhash_std_array[f_ind] = 0.0
            #print I_r,I_m
        else:
            #subsample data
            for t_ind in range(number_of_tries):
                #sample indices of repeats
                inds_to_use = choice(repeat_inds,size=num_repeats,replace=False)
                data_sub_block = np.copy(data_block[:,inds_to_use,:])
                I_temp_r[t_ind],I_temp_u[t_ind],I_temp_mhash[t_ind],I_temp_uhash[t_ind] = I_comp(data_sub_block,
                                                                                                 m_hashes,u_hashes)
            I_r_final_array[f_ind] = np.mean(I_temp_r)
            I_u_final_array[f_ind] = np.mean(I_temp_u)
            I_mhash_final_array[f_ind] = np.mean(I_temp_mhash)
            I_uhash_final_array[f_ind] = np.mean(I_temp_uhash)
            
            I_r_std_array[f_ind] = np.std(I_temp_r)
            I_u_std_array[f_ind] = np.std(I_temp_u)
            I_mhash_std_array[f_ind] = np.std(I_temp_mhash)
            I_uhash_std_array[f_ind] = np.std(I_temp_uhash)
    
    #convert to bits per second
    I_r_final_array *= to_seconds
    I_u_final_array *= to_seconds
    I_mhash_final_array *= to_seconds
    I_uhash_final_array *= to_seconds
    
    I_r_std_array *= to_seconds
    I_u_std_array *= to_seconds
    I_mhash_std_array *= to_seconds
    I_uhash_std_array *= to_seconds
    
    
    #get extrapolated values
    r_slope,r_intercept,r_r_value,r_p_value,r_std_err = linregress(regression_x,I_r_final_array)
    u_slope,u_intercept,u_r_value,u_p_value,u_std_err = linregress(regression_x,I_u_final_array)
    mhash_slope,mhash_intercept,mhash_r_value,mhash_p_value,mhash_std_err = linregress(regression_x,I_mhash_final_array)
    uhash_slope,uhash_intercept,uhash_r_value,uhash_p_value,uhash_std_err = linregress(regression_x,I_uhash_final_array)
    
    r_std  = np.mean(np.sqrt(real_fractions[1:])*I_r_std_array[1:])
    u_std  = np.mean(np.sqrt(real_fractions[1:])*I_u_std_array[1:])
    mhash_std  = np.mean(np.sqrt(real_fractions[1:])*I_mhash_std_array[1:])
    uhash_std  = np.mean(np.sqrt(real_fractions[1:])*I_uhash_std_array[1:])
    
    return r_intercept,u_intercept,mhash_intercept,uhash_intercept,r_std,u_std,mhash_std,uhash_std

def I_loop_full(trial_data,beta_dict,theta_dict,Nybins=10,Nxbins=10):
    list_of_neurons = [key for key in trial_data.keys() if key != 'all movies']
    N_val_list = []
    I_r_list = []
    I_u_list = []
    I_mhash_list = []
    I_uhash_list = []
    name_list = []
    max_N = len(list_of_neurons)
    for N in range(2,max_N+1):
        list_of_possible_subsets = list(itertools.combinations(list_of_neurons,N))
        for subset in list_of_possible_subsets:
            subset_list = list(subset)
            beta = np.zeros(len(subset_list))
            theta = np.zeros(len(subset_list))
            for t in range(len(subset_list)):
                beta[t] = beta_dict[subset_list[t]]
                theta[t] = theta_dict[subset_list[t]]
            num_movies = len(cu.get_available_movies(subset_list,trial_data))
            if num_movies > 0:
                data_block = cu.build_data_set(subset_list,trial_data)
                I_r,I_u,I_mhash,I_uhash,r_std,u_std,mhash_std,uhash_std = linear_extrap_I_full(data_block,beta,theta
                                                                                     ,Nybins=Nybins,Nxbins=Nxbins)
                #make sure all the returned informations are statistically significantly greater than zero
                sig_list = [(I_r > r_std),(I_u > u_std),(I_mhash > mhash_std),(I_uhash > uhash_std)]
                if all(sig_list):
                    I_r_list.append(I_r)
                    I_u_list.append(I_u)
                    I_mhash_list.append(I_mhash)
                    I_uhash_list.append(I_uhash)
                    N_val_list.append(len(subset_list))
                    name_list.append(subset_list)
    return name_list,N_val_list,I_r_list,I_u_list,I_mhash_list,I_uhash_list