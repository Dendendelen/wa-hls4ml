import numpy as np
import torch

import sys

from torch_geometric import loader as gloader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

from model.wa_hls4ml_model import load_model
from data.wa_hls4ml_plotly import plot_results
from data.wa_hls4ml_data_plot import plot_histograms

def calculate_metrics(y_test, y_pred):
    ''' Calculate out MAE, MSE, RMSE, and R^2 '''

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')


def display_results_classifier(X_test, X_raw_test, y_test, output_features, folder_name, is_graph=False):
    ''' Display the results of the classification model '''

    model = load_model(folder_name+'/classification').to("cpu")
    model.switch_device("cpu")
    model.eval()

    with torch.no_grad():

        # Predict the output for X_test
        if is_graph:
            X_loader = gloader.DataLoader(X_test, batch_size=len(X_test))
            X = next(iter(X_loader))
            y_pred = model(X).detach().numpy()
        else:
            y_pred = model(torch.tensor(X_test)).detach().numpy()

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # Calculate confusion matrix
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    confusion = confusion_matrix(y_pred_binary, y_test)
    print('Confusion matrix:')
    print(confusion)

    # plot our classification results
    y_test_2d = np.reshape(y_test, (y_test.shape[0], 1))
    plot_results("classifier", False, y_test_2d, y_pred, X_raw_test, output_features, folder_name)


def display_results_regressor(X_test, X_raw_test, y_test, output_features, folder_name, is_graph):
    ''' Display the results of the regression models '''

    y_pred = np.empty(y_test.shape)

    i = 0
    for feature in output_features:

        model = load_model(folder_name+'/regression_'+feature).to("cpu")
        model.switch_device("cpu")
        model.eval()

        with torch.no_grad():
            # Predict the output of this specific feature for X_test
            if is_graph:
                X_loader = gloader.DataLoader(X_test, batch_size=len(X_test))
                X = next(iter(X_loader))
                y_pred_part = model(X).detach().numpy()
                from model.wa_hls4ml_train import bounded_percentile_loss
                print(torch.mean(torch.nn.functional.l1_loss(torch.tensor(y_pred_part[:,0]), torch.tensor(y_test[:, i]))))
                # print(torch.nn.functional.huber_loss(torch.tensor(y_pred_part[:,0]), torch.tensor(y_test[:, i])))
                # print("saving...")
                # np.save("dump_lut_pred.npy",y_pred_part[:,0])
                # np.save("dump_lut_gt.npy", y_test[:,i])
                # sys.exit(0)
                # # np.save("dump_dsp.npy", y_pred_part)
                
                # print(X_test)
            else:
                y_pred_part = model(torch.tensor(X_test)).detach().numpy()

            print("Part " + feature + ": " + str(y_pred_part.shape))

        # Consolidate feature predictions
        y_pred[:, i] = y_pred_part[:, 0]
            
        i += 1

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # plot our regression results
    plot_results("regression_all", False, y_test, y_pred, X_raw_test, output_features, folder_name)


def test_regression_classification_union(X_test, X_raw_test, y_test, features_without_classification, feature_classification_task, folder_name, is_graph = False):
    '''Test the effectiveness of the whole model, first doing classification, and using that result to help with regression'''

    features_with_classification = features_without_classification + feature_classification_task

    model_classifier = load_model(folder_name+"/classification").to("cpu")
    model_classifier.switch_device("cpu")

    # predict the classes of the test dataset, then convert to binary 1/0
    if is_graph:
        X_loader = gloader.DataLoader(X_test, batch_size=len(X_test))
        X = next(iter(X_loader))
        class_pred = model_classifier(X).detach().numpy()
    else:
        class_pred = model_classifier(torch.tensor(X_test)).detach().numpy()
    class_binary = np.where(class_pred > 0.5, 1, 0)
    print("Binary classification created, shape:")
    print(class_binary.shape)

    # get only the predicted successes for regression prediction
    succeeded_idx = np.nonzero(class_binary)[0]
    print("Indices of success created, shape:")
    print(succeeded_idx.shape)

    if is_graph:
        X_test_only_success = []
        for i in succeeded_idx:
            X_test_only_success.append(X_test[i])
        y_regression_pred = np.empty((len(X_test_only_success), 6))
    else:
        X_test_only_success = X_test[succeeded_idx]
        y_regression_pred = np.empty((X_test_only_success.shape[0], 6))

    i = 0
    for feature in features_without_classification:
        model_regressor = load_model(folder_name+'/regression_'+feature+'/').to("cpu")
        model_regressor.switch_device("cpu")

        if is_graph:
            X_loader = gloader.DataLoader(X_test_only_success, batch_size=len(X_test_only_success))
            X = next(iter(X_loader))
            y_regression_pred_slice = model_regressor(X).detach().numpy()
        else:
            y_regression_pred_slice = model_regressor(torch.tensor(X_test_only_success)).detach().cpu().numpy()
        y_regression_pred[:, i] = y_regression_pred_slice[:, 0]
        i += 1

    # Add back the classification as a prediction index (does not yet need to have the data)
    y_regression_pred_with_classification_column = np.append(y_regression_pred, np.full((y_regression_pred.shape[0], 1), 1), axis=1)

    # Restore the indices that fail classification
    y_regression_pred_reshaped = np.empty((class_binary.shape[0], y_regression_pred_with_classification_column.shape[1]))
    y_regression_pred_reshaped[succeeded_idx] = y_regression_pred_with_classification_column

    y_pred = np.where(class_binary > 0, y_regression_pred_reshaped, -1)

    # set the classification predictions as the 0/1 binary for error calc
    y_pred[:, -1] = class_binary[:, 0]

    print("Added the classification predictions to the output, shape:")
    print(y_pred.shape)

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # Generate histograms for all
    plot_histograms(y_pred, y_test, features_with_classification, folder_name)

    # now set classification predictions to be the classification model's raw outputs
    y_pred[:, 6] = class_pred[:,0]

    # use plotly to create result graph
    plot_results("both", False, y_test, y_pred, X_raw_test, features_with_classification, folder_name)
    
