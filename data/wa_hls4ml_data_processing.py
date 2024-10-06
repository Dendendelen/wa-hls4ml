import pandas as pd
import numpy as np
import torch
import sklearn.model_selection

import torch_geometric as pyg
from torch_geometric.data import Data
        
from data.wa_hls4ml_json_to_csv import parse_file

import os
import sys
import math

# current I/O:
#   inputs: d_in, d_2,	d_out, prec, rf, strategy
#   outputs: TargetClockPeriod_hls,	WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls, hls_synth_success

def preprocess_data_from_csv(model_folder, csv_file, input_features, output_features, _binary_feature_names, _numeric_feature_names, _categorical_feature_names, special_feature_names, presaved_mean = None, presaved_stdev = None):
    ''' Extract data from a CSV, and preprocess that data '''

    # Step 1: Read the CSV file
    df = pd.read_csv(csv_file)
    df.fillna(-1)
    preprocessed_data = []
    processing_input = True
    for sel_feature_names in [input_features, output_features, input_features]: # Do input features twice to get normalized and non-normalized values
        binary_feature_names = [item for item in _binary_feature_names if item in sel_feature_names]
        numeric_feature_names = [item for item in _numeric_feature_names if item in sel_feature_names]
        categorical_feature_names = [item for item in _categorical_feature_names if item in sel_feature_names]

        # Step 2: Split the DataFrame into input and output DataFrames
        input_data = df[sel_feature_names]

        # Steps 3-6: Process binary, numeric, and categorical features
        preprocessed_inputs = preprocess_features(input_data, binary_feature_names, numeric_feature_names, categorical_feature_names, presaved_mean, presaved_stdev, model_folder, processing_input)
        processing_input = False

        # Step 7: Convert the preprocessed data to numpy arrays
        preprocessed_inputs = preprocessed_inputs.numpy()
        preprocessed_data.append(preprocessed_inputs)

    special_data = []

    # Step 8: Extract special features
    # these are features which we do not want to process at all
    for i in range (preprocessed_data[0].shape[0]):
        special_datapoint = []
        for name in special_feature_names:
            special_feature = df[name][i]
            special_datapoint.append(special_feature)
        special_data.append(special_datapoint)

    return np.nan_to_num(preprocessed_data[0], nan=-1), np.nan_to_num(preprocessed_data[1], nan=-1), np.nan_to_num(preprocessed_data[2], nan=-1), special_data


def preprocess_features(data, binary_feature_names, numeric_feature_names, categorical_feature_names, presaved_mean, presaved_stdev, model_folder, processing_input=True):
    ''' Preprocess features '''
    preprocessed = []

    # Step 3: Process numeric features
    if numeric_feature_names:
        for name in numeric_feature_names:
            data[name] = pd.to_numeric(data[name], errors='coerce')
        print("Numerical features processed, top values:")
        print(data[numeric_feature_names].head())
        if processing_input:
            tensorized_val = torch.tensor(data[numeric_feature_names].astype('float32').values)
            if presaved_mean is not None and presaved_stdev is not None:
                overall_mean = torch.tensor(presaved_mean)
                overall_stdev = torch.tensor(presaved_stdev)
            else:
                overall_mean, overall_stdev = tensorized_val.mean(dim=0), tensorized_val.std(dim=0)
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                np.save(model_folder+"/mean.npy", overall_mean.numpy())
                np.save(model_folder+"/stdev.npy", overall_stdev.numpy())

            mean, stdev = overall_mean.broadcast_to(tensorized_val.shape), overall_stdev.broadcast_to(tensorized_val.shape)

            numeric_normalized = (tensorized_val-mean)/stdev
        else:
            numeric_normalized = torch.tensor(data[numeric_feature_names].values.astype('float32'))
        preprocessed.append(numeric_normalized)

    # Step 4: Process binary features
    if binary_feature_names:
        for name in binary_feature_names:
            value = data[name].astype(bool).astype('float32')
            value = torch.tensor(value).unsqueeze(1)
            # value = torch.tensor(2*value - 1).unsqueeze(1)
            
            preprocessed.append(value)

    # Step 5: Process categorical features
    if categorical_feature_names:
        for name in categorical_feature_names:
            vocab = sorted(set(data[name][1:])) #Exclude header

            # skip this if we trivially have but one data point
            if len(vocab) == 1: 
                continue
            if type(vocab[0]) is str:
                # change strings to integers
                i = 0
                for word in vocab:
                    data[name] = data[name].replace(word, i)
                    i += 1

            numbered_data = torch.tensor(data[name])

            one_hot = torch.zeros(numbered_data.shape[0], len(vocab))
            one_hot.scatter_(1, numbered_data.unsqueeze(1), 1.0)

            print("Categorical feature processed, shape:")
            print(data[name].shape)
            preprocessed.append(one_hot)

    # Step 6: Concatenate all processed features
    preprocessed_data = torch.cat(preprocessed, dim=1)
    return preprocessed_data


def parse_json_string(json, mean_val, stdev_val):
    ''' Parse the model information out of a JSON string ''' 

    json_list = json.split('-')

    layer_number = len(json_list)
    
    raw_nodes_count = np.empty((layer_number,))
    nodes_count = np.empty((layer_number,))
    source = np.empty((layer_number-1,))
    target = np.empty((layer_number-1))

    for i in range(layer_number):

        raw_nodes_count[i] = float(json_list[i])
        nodes_count[i] = (float(json_list[i])-mean_val)/stdev_val

        if i < layer_number-1:
            source[i] = i
            target[i] = i+1

    return nodes_count, source, target, raw_nodes_count


def create_graph_tensor(input_values, input_raw_values, input_json, mean, stdev, dev):
    ''' Turn the data into the form of a GraphTensor to allow for GNN use ''' 

    input_values_2 = np.asarray(input_values[2:]).astype('float32') # for resource and latency

    # parse model string into distinct nodes and edges
    nodes_count, source, target, raw_nodes_count = parse_json_string(input_json, (mean[0]+mean[1])/2, (stdev[0]+stdev[1])/2)

    # concatenate and transpose the adjacency list
    adjacency_list = torch.einsum('ij -> ji', torch.cat((torch.tensor(source).unsqueeze(1), torch.tensor(target).unsqueeze(1)), dim = 1)).to(dev)

    bops_features = np.empty(nodes_count.shape)

    for i in range(nodes_count.shape[0]):
        curr_nodes = raw_nodes_count[i]
        if i+1 == nodes_count.shape[0]:
            next_nodes = 1
        else:
            next_nodes = raw_nodes_count[i+1]
        
        p = input_raw_values[3]
        
        bops = curr_nodes * next_nodes * ( p**2 + 2 * p + math.log2(curr_nodes) )

        # arrange these to the same or similar order of magnitude as other values
        bops = math.sqrt(bops)/1000
    
        bops_features[i] = bops

    bops_features = bops_features.astype('float32')

    # node features are number of nodes in layer, and BOPs estimated
    nodes = torch.cat((torch.tensor(nodes_count).unsqueeze(1).to(dev), torch.tensor(bops_features).unsqueeze(1).to(dev)), dim=1)

    # edge vector at present is all ones, no training occurs on it besides implicit adjacency list
    edges = torch.tensor(np.ones((nodes_count.shape[0]-1),)).unsqueeze(1).to(dev)
    global_features = torch.tensor(input_values_2).to(dev)

    # add the number of edges itself as a global feature    
    global_features = torch.cat((global_features, torch.tensor(source.shape[0]).unsqueeze(0).to(dev)))
    graph_datapoint = Data(x=nodes, edge_index=adjacency_list, edge_attr=edges, y = global_features)

    return graph_datapoint   


def preprocess_data(model_folder, is_graph = False, input_folder="../results/results_combined.csv", needs_json_parsing = False, is_already_serialized = False, mean = None, stdev = None, doing_train_test_split = True, dev = "cpu"):
    ''' Preprocess the data '''

    input_features = ["d_in", "d_out", "prec", "rf", "strategy", "rf_times_precision"]
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls", "hls_synth_success"]
    binary_feature_names = ['hls_synth_success']
    numeric_feature_names = ["d_in", "d_2", "d_out", "prec", "rf", "WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls",
                             "BRAM_18K_hls", "DSP_hls", "rf_times_precision"]
    categorical_feature_names = ["strategy"]
    special_feature_names = ["model_string"]

    if needs_json_parsing:
        parse_file(input_folder)
        input_folder = 'auto_parsed_json.csv'

    _X, y, X_raw, special_data = preprocess_data_from_csv(model_folder, input_folder, input_features, output_features,
                             binary_feature_names, numeric_feature_names,
                             categorical_feature_names, special_feature_names, presaved_mean=mean, presaved_stdev=stdev)

    # load in the preprocessed mean and stdev values
    mean = np.load(model_folder + "/mean.npy")
    stdev = np.load(model_folder + "/stdev.npy")

    if (is_graph and not is_already_serialized):
        i = 0
        graph_tensor_list = []

        for datapoint in special_data:
            # tensorize this data into the torch graph-based data format
            graph_tensor = create_graph_tensor(_X[i], X_raw[i], datapoint, mean, stdev, dev)
            graph_tensor_list.append(graph_tensor)
            i += 1
            if i % 5000 == 0:
                print("Processing special feature " + str(i))

        X = graph_tensor_list
    else:
        X = _X
        print(X.shape, y.shape)

    # Split the data 70 - 20 - 10 train test val
    # Train and test
    print("X Data: ",input_features)
    print("Y Data: ",output_features)

    if doing_train_test_split:
        X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = sklearn.model_selection.train_test_split(X, y, X_raw,  test_size=0.2, random_state=42, shuffle=True)
    else:
        # in this case, the user is responsible for the splitting
        X_train = X
        X_test = X
        y_train = y
        y_test = y
        X_raw_train = X_raw
        X_raw_test = X_raw

    return X_train, X_test, y_train, y_test, X_raw_train, X_raw_test
