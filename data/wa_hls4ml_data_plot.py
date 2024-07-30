import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(name, history, folder_name):
    ''' plot losses during training of a model '''

    fig, ax1 = plt.subplots(figsize=(10,6))

    lns1 = ax1.plot(history['train'], label='Training Loss')
    lns2 = ax1.plot(history['val'], label='Validation Loss')
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')
    lns3 = ax2.plot(history['lr'], 'r', label='Learning Rate')

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    directory = folder_name+'/plots/training/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+name+'_train_loss.png')
    plt.show()


def plot_histogram(y, name, x_axis, filename, folder_name, log=False, color="blue"):
    ''' plot a single histogram '''

    plt.figure(figsize=(10,6))
    plt.hist(y, 20, color=color)
    plt.title(name)
    plt.ylabel('Number')
    plt.xlabel(x_axis)

    directory = folder_name+'/plots/histograms/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    if log:
        plt.yscale('log')
        plt.savefig(directory+filename+'_log_hist.png')
    else:
        plt.savefig(directory+filename+'_hist.png')

    plt.close()


def plot_scatter(x, y, name, x_axis, y_axis, filename, folder_name, log=False):
    ''' Plot a 2d histogram '''

    plt.figure(figsize=(10,6))
    if log:
        plt.hist2d(x, y, bins=(20,15), density=True, norm='log')
    else:
        plt.hist2d(x, y, bins=(20,15), density=True)

    plt.title(name)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)

    directory = folder_name+'/plots/histograms/'

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if log:
        plt.savefig(directory+filename+'_log_2dhist.png')
    else:
        plt.savefig(directory+filename+'_2dhist.png')
    plt.close()

def plot_histograms(y_predicted, y_actual, output_features, folder_name):
    '''Plot all histograms for features '''

    y_difference = y_predicted - y_actual
    y_abs_diff = np.abs(y_difference)

    i = 0

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']
    for feature in output_features:

        feature_diff = y_difference[:, i]
        abs_feature_diff = y_abs_diff[:, i]

        rms = np.sqrt(np.mean(np.square(feature_diff)))

        plot_scatter(y_actual[:, i], feature_diff, "Residual vs Value of "+feature, feature, 'Error', feature, folder_name, False)
        plot_scatter(y_actual[:, i], abs_feature_diff, "Log Absolute Residual vs Value of "+feature, feature, 'Error', feature, folder_name, True)

        plot_histogram(abs_feature_diff, 'Absolute Residual of '+feature, 'Absolute Error of '+feature, feature+"_abs", folder_name, False, colors[i])
        plot_histogram(abs_feature_diff, 'Log Absolute Residual of '+feature, 'Absolute Error of '+feature, feature+"_abs", folder_name, True, colors[i])

        plot_histogram(feature_diff, 'Residual of '+feature, 'Error of '+feature +" (RMSE = " + str(rms) +")", feature, folder_name, False, colors[i])
        plot_histogram(feature_diff, 'Log Residual of '+feature, 'Error of '+feature +" (RMSE = " + str(rms) +")", feature, folder_name, True, colors[i])

        print("Finished plots for "+feature)

        i += 1