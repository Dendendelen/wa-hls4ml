import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data

from data.wa_hls4ml_plotly import plot_results
from model.wa_hls4ml_model import save_model, load_model
from model.wa_hls4ml_train import train_classifier, train_regressor
from model.wa_hls4ml_test import calculate_metrics, display_results_classifier,display_results_regressor, test_regression_classification_union
from data.wa_hls4ml_data_processing import preprocess_data


def perform_train_and_test(train, test, regression, classification, skip_intermediates, is_graph, folder_name = "model_1", input_file = "../results/results_combined.csv", dev="cpu"):

    features_without_classification = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    feature_classification_task = ["hls_synth_success"]

    # get raw data out
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = preprocess_data(is_graph, input_file, dev = dev)

    # get just the classification task as its own variable
    y_train_classifier = y_train[:, -1]
    y_test_classifier = y_test[:, -1]

    # train the classifier
    if train and classification:
        print("Training the classifier...")
        train_classifier(X_train, y_train_classifier, folder_name, is_graph, dev)
    if test and classification and not skip_display_intermediate:
        display_results_classifier(X_test, X_raw_test, y_test_classifier, feature_classification_task, folder_name, is_graph)

    # This is all we do if we are doing classification alone
    if not regression:
        return

    # find which synth has succeeded in groundtruth
    succeeded_synth_gt_test = np.nonzero(y_test_classifier)
    succeeded_synth_gt_train = np.nonzero(y_train_classifier)

    # only train regressor on successes
    if is_graph:
        X_succeeded_train = []
        X_raw_succeeded_train = []
        for i in succeeded_synth_gt_train[0]:
            X_succeeded_train.append(X_train[i])
            X_raw_succeeded_train.append(X_raw_train[i])
    else:
        X_succeeded_train = X_train[succeeded_synth_gt_train]
        X_raw_succeeded_train = X_raw_train[succeeded_synth_gt_train]
    y_succeeded_train = (y_train[succeeded_synth_gt_train])[:, :-1]

    # only test regressor alone on successes
    if is_graph:
        X_succeeded_test = []
        X_raw_succeeded_test = []
        for i in succeeded_synth_gt_test[0]:
            X_succeeded_test.append(X_test[i])
            X_raw_succeeded_test.append(X_raw_test[i])
    else:
        X_succeeded_test = X_test[succeeded_synth_gt_test]
        X_raw_succeeded_test = X_raw_test[succeeded_synth_gt_test]
    y_succeeded_test = (y_test[succeeded_synth_gt_test])[:, :-1]

    # train the regressor
    if train and regression:
        print("Training the regressor...")
        train_regressor(X_succeeded_train, y_succeeded_train, features_without_classification, folder_name, is_graph, dev)
    if test and regression and not skip_intermediates:
        display_results_regressor(X_succeeded_test, X_raw_succeeded_test, y_succeeded_test, features_without_classification, folder_name, is_graph)

    # if we are not doing both regression and classification, we are done
    if not regression or not classification or not test:
        return
    
    print("Testing the models in union...")
    test_regression_classification_union(X_test, X_raw_test, y_test, features_without_classification, feature_classification_task, folder_name, is_graph)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='wa-hls4ml', description='Train or test model for wa-hls4ml', add_help=True)

    parser.add_argument('--gpu', action='store_true', help='Use CUDA GPU processing for training')
    
    parser.add_argument('--train', action='store_true', help='Train a new surrogate model from the data')
    parser.add_argument('--test', action='store_true', help='Test existing models')

    parser.add_argument('-c', '--classification', action='store_true', help='Train/test the classifier')
    parser.add_argument('-r','--regression', action='store_true', help='Train/test the regressor')

    parser.add_argument('-g','--gnn', action='store_true', help='Use a graph neural network to model the layers of the ml model')

    parser.add_argument('-i', '--input', action='store', help="What file to use for input data")

    parser.add_argument('-f', '--folder', action='store', help='Set the folder you want the model outputs to be created within', required=True)

    
    args = parser.parse_args()
    args_dict = vars(args)

    # # allow for GPU use if availible
    # if os.system('nvidia-smi') == 0:
    #     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #     device = "/GPU:0"
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     if gpus:
    #         try:
    #             for gpu in gpus:
    #                 tf.config.experimental.set_memory_growth(gpu, True)
    #         except RuntimeError as e:
    #             print(e)
    # else:
    #     device = "/CPU:0"
    # torch.cuda.device("cuda")

    train = args_dict['train']
    test = args_dict['test']

    # if neither flag is assigned, assume both
    if not train and not test:
        train = True
        test = True

    classification = args_dict['classification']
    regression= args_dict['regression']

    skip_display_intermediate = False

    # if neither flag is assigned, assume both, but that we do not care about the intermediate classification/regression results
    if not classification and not regression:
        classification = True
        regression = True
        skip_display_intermediate = True

    # whether or not to use the GNN
    is_graph = args_dict['gnn']

    # folder to store all the outputs into
    folder = args_dict['folder']

    # redirect the input into a models folder
    folder = 'models/'+folder

    input_file = args_dict['input']

    is_gpu = args_dict['gpu']

    # allow using CUDA
    if is_gpu:
        dev = "cuda"
    else:
        dev = "cpu"

    # perform the testing and training depending on arguments
    print("Beginning...")
    perform_train_and_test(train, test, regression, classification, skip_display_intermediate, is_graph, folder, input_file, dev)
    print("Done")
    



    