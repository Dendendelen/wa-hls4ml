import pandas as pd
import numpy as np
import torch
import sklearn.model_selection

import torch_geometric as pyg
from torch_geometric.data import Data

import sys
import math

# current I/O:
#   inputs: d_in, d_2,	d_out, prec, rf, strategy
#   outputs: TargetClockPeriod_hls,	WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls, hls_synth_success

def preprocess_data_from_csv(csv_file, input_features, output_features, _binary_feature_names, _numeric_feature_names, _categorical_feature_names, special_feature_names):
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
        preprocessed_inputs = preprocess_features(input_data, binary_feature_names, numeric_feature_names, categorical_feature_names, processing_input)
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


def preprocess_features(data, binary_feature_names, numeric_feature_names, categorical_feature_names,processing_input=True):
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
            mean, stdev = tensorized_val.mean(dim=0).broadcast_to(tensorized_val.shape), tensorized_val.std(dim=0).broadcast_to(tensorized_val.shape)
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


def parse_json_string(json):
    ''' Parse the model information out of a JSON string ''' 
    #TODO: implement

    if np.isnan(json[1]):
        nodes_count = np.asarray([json[0], json[2]])

        source = np.zeros((1,)).astype('int64')
        target = np.ones((1,)).astype('int64')

        activation = np.zeros((1,3)).astype('int64')
        # activation[0, 0] = 1

        density = np.ones((1,)).astype('float32')
        dropout = np.zeros((1,)).astype('float32')
    else:
        nodes_count = np.asarray([json[0], json[1], json[2]])

        source = np.empty((2,))
        source[0] = 0
        source[1] = 1

        source = source.astype('int64')

        target = np.empty((2,))
        target[0] = 1
        target[1] = 2

        target = target.astype('int64')

        activation = np.zeros((2,3)).astype('int64')
        density = np.ones((2,)).astype('float32')
        dropout = np.zeros((2,)).astype('float32')

    return nodes_count, source, target, activation, density, dropout


def create_graph_tensor(input_values, input_raw_values, input_json, dev):
    ''' Turn the data into the form of a GraphTensor to allow for GNN use ''' 

    # ------------------ temporary while using CSV rather than JSON ---------------

    # input_values_2 = np.asarray(input_values[3:-1]).astype('float32') # for only resource
    input_values_2 = np.asarray(input_values[3:]).astype('float32') # for resource and latency

    input_json = input_values[:3]
    raw_input_json = input_raw_values[:3]

    if input_raw_values[1] == -1:
        input_json[1] = None
        raw_input_json[1] = None
 
    # input_values_2 = np.asarray(input_values[3:]).astype('float32')

    #TODO: parse json into distinct nodes and edges
    nodes_count, source, target, activation, density, dropout = parse_json_string(input_json)
    raw_nodes_count, _, _, _, _, _ = parse_json_string(raw_input_json)

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
        
        bops = curr_nodes * next_nodes * ( p**2 + p + math.log2(curr_nodes) )
    
        bops_features[i] = bops

    bops_features = bops_features.astype('float32')

    # node features are number of nodes in layer, and BOPs estimated
    nodes = torch.cat((torch.tensor(nodes_count).unsqueeze(1).to(dev), torch.tensor(bops_features).unsqueeze(1).to(dev)), dim=1)

    edges = torch.tensor(density).unsqueeze(1).to(dev)
    global_features = torch.tensor(input_values_2).to(dev)

    # add the number of edges itself as a global feature    
    global_features = torch.cat((global_features, torch.tensor(source.shape[0]).unsqueeze(0).to(dev)))
    graph_datapoint = Data(x=nodes, edge_index=adjacency_list, edge_attr=edges, y = global_features)

    return graph_datapoint   


def preprocess_data(is_graph = False, input_folder="../results/results_combined.csv", is_already_serialized = False, dev = "cpu"):
    ''' Preprocess the data '''

    input_features = ["d_in", "d_2", "d_out", "prec", "rf", "strategy", "rf_times_prec"]
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls", "hls_synth_success"]
    binary_feature_names = ['hls_synth_success']
    numeric_feature_names = ["d_in", "d_2", "d_out", "prec", "rf", "WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls",
                             "BRAM_18K_hls", "DSP_hls", "rf_times_prec"]
    categorical_feature_names = ["strategy"]
    # special_feature_names = ["json"]
    special_feature_names = ["model_name"]

    _X, y, X_raw, special_data = preprocess_data_from_csv(input_folder, input_features, output_features,
                             binary_feature_names, numeric_feature_names,
                             categorical_feature_names, special_feature_names)

    if (is_graph and not is_already_serialized):
        i = 0
        graph_tensor_list = []

        for datapoint in special_data:
            # tensorize this data into the torch graph-based data format
            graph_tensor = create_graph_tensor(_X[i], X_raw[i], datapoint, dev)
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

    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = sklearn.model_selection.train_test_split(X, y, X_raw,  test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test, X_raw_train, X_raw_test
