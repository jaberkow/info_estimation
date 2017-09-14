import numpy as np
from operator import itemgetter
import sys
import os

def get_full_site(site_number,path_to_data,stim_type='natural scenes',stim_length=330):
        
    if stim_type=='natural scenes':
        stim_code = 212
    else:
        stim_code = 229
    
    gm_list = []
    for file_name in os.listdir(path_to_data):
        if file_name[2:5] == str(site_number):
            gm_list.append(file_name)
    Num_Neurons = len(gm_list)
    
    if Num_Neurons == 0:
        print "Site {} does not appear in ".format(site_number) + path_to_data
        return None
    
    dict_of_dicts = {}
    
    all_keys = []
    print "{} neurons for site {}".format(len(gm_list),site_number)
    for gm_file_name in gm_list:
        neuron_key = gm_file_name[2:-4]
        gm_file = open(path_to_data + gm_file_name,"rb")
        #print gm_file_name
        data_dict  = {}
        for line in gm_file:
            [code,isk_key,isk_index] = line.split()
            if int(code) == stim_code:
                #isk_dict.update({isk_key:isk_key + '_' + isk_index + '.isk'})
                isk_file_name = isk_key + '_' + isk_index + '.isk'
                if(isk_file_name in os.listdir(path_to_data)):
                    cur_data = scan_through_isk(isk_file_name,path_to_data)
                    data_dict.update({isk_key:cur_data})
                    if not(isk_key in all_keys):
                        all_keys.append(isk_key)
                
        gm_file.close()
        #print isk_dict
        if len(data_dict) > 0:
            dict_of_dicts[neuron_key] = data_dict.copy()
    
    Number_surviving = len(dict_of_dicts)
    
    """
    #Now keep only protocols that appear in every neuron's dictionary
    #get all keys
    
    #all_keys  = list(set().union(*(d.keys() for d in dict_of_dicts.values())))
    
    
    
    list_of_valid_keys = []
    
    for current_key in all_keys:
        if all(current_key in d for d in dict_of_dicts.values()):
            list_of_valid_keys.append(current_key)
           
    """
    dict_of_dicts.update({'all movies':all_keys})
    return dict_of_dicts

def scan_through_isk(file_name, path_to_data, stim_length=330):
    isk_file = open(path_to_data + file_name ,'rb')
    my_list = list(isk_file)
    isk_file.close()
    my_vals = [int(el.replace('\r\n','')) for el in my_list[2:]]
    my_array = np.array(my_vals)
    N_repeats = int(my_array.shape[0]/stim_length)
    my_array = np.reshape(my_array,(stim_length,N_repeats))
    return my_array

def get_available_movies(list_of_neurons,trial_data):
    #get a list of movies that every neuron in list_of_neurons was exposed too
    
    possible_keys = trial_data['all movies']
    subset = itemgetter(*list_of_neurons)(trial_data)
    valid_key_list = []
    for cur_key in possible_keys:
        if all(cur_key in d for d in subset):
            valid_key_list.append(cur_key)
    
    #key_subset = list(set().union(*(d.keys() for d in subset)))
    
    
    return valid_key_list

def build_data_set(list_of_neurons,trial_data):
    
    """
    Resulting data set is of shape (num_stimuli,total repeats,num_neurons)
    """
    
    #creates a numpy array of all the data with requisite number of repeats and stims
    valid_movie_list = get_available_movies(list_of_neurons,trial_data)
    
    Num_neurons = len(list_of_neurons)
    
    neur_data_list = []
    
    for neuron in list_of_neurons:
        data_list = []
        current_neur_dict = trial_data[neuron]
        for movie in valid_movie_list:
            data_list.append(current_neur_dict[movie])
        if len(valid_movie_list) > 1:
            #stack the different movies by axis 1
            neur_data_list.append(np.concatenate(data_list,axis=1))
        else:
            neur_data_list.append(data_list[0])
    
    #Now we have a list of arrays, we shall stack them
    
    
    return np.stack(neur_data_list,axis=-1)