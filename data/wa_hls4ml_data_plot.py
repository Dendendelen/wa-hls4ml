import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

        # save residuals for later plotting
        np.save(folder_name+"/plots/"+feature+".npy", feature_diff)
        print("Finished plots for "+feature)

        i += 1


def plot_box_plots(y_pred, y_test, folder_name):
    # Box plot
    # prediction_labels =  ['Cycles', 'FF', 'LUT', 'BRAM', 'DSP']
    prediction_labels =  ['BRAM', 'DSP', 'FF', 'LUT', 'Cycles']
    

    prediction_errors = []
    for i in range(0, len(prediction_labels)):
        # Percent error
        # SMALLN = 1e-15
        # y_test[:, i] = np.where(y_test[:, i] == 0, SMALLN, y_test[:, i])
        # prediction_errors.append(np.abs((y_test[:,i] - y_pred[:,i])/y_test[:,i])*100)

        # Relative Percent error
        prediction_errors.append((y_test[:,i] - y_pred[:,i])/(y_test[:,i]+1)*100)
        
        # Absolute error
        # prediction_errors.append(np.abs(y_test[:,i] - y_pred[:,i]))

        # # Relative Error
        # prediction_errors.append(y_test[:,i] - y_pred[:,i])
    
    prediction_errors=[prediction_errors[3],prediction_errors[4],prediction_errors[1],
                       prediction_errors[2],prediction_errors[0]]
    
    plt.rcParams.update({"font.size": 16})
    fig, axis = plt.subplots(1, len(prediction_labels), figsize=(12, 8))
    axis = np.reshape(axis, -1)
    fig.subplots_adjust(hspace=0.1, wspace=0.6)
    iqr_weight = 1.5
    colors = ["pink", "yellow", "lightgreen", "lightblue", "#FFA500"] #plum
    for i, errors in enumerate(prediction_errors):
        label = prediction_labels[i]
        ax = axis[i]
        bplot = ax.boxplot(
            errors,
            whis=iqr_weight,
            tick_labels=[label.upper()],
            showfliers=True,
            showmeans=True,
            meanline=True,
            vert=True,
            patch_artist=True
        )
        for j, patch in enumerate(bplot["boxes"]):
            patch.set_facecolor(colors[(i + j) % len(colors)])
        ax.yaxis.grid(True)
        ax.spines.top.set_visible(False)
        ax.xaxis.tick_bottom()
    median_line = Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="Median")
    mean_line = Line2D([0], [0], color="green", linestyle="--", linewidth=1.5, label="Mean")
    handles = [median_line, mean_line]
    labels = ["Median", "Mean"]
    legends = fig.legend(
        handles,
        labels,
        bbox_to_anchor=[0.9, 1],
        loc="upper right",
        ncol=len(labels) // 2,
    )
    # ytext = fig.text(0.02, 0.5, "Absolute Error", va="center", rotation="vertical", size=18)
    ytext = fig.text(0.02, 0.5, "Relative Percent Error", va="center", rotation="vertical", size=18)
    # ytext = fig.text(0.02, 0.5, "Relative Error", va="center", rotation="vertical", size=18)
    suptitle = fig.suptitle("Prediction Errors - Boxplots", fontsize=20, y=0.95)
    fig.savefig(
        "./box_plot_exemplar.pdf",
        dpi=300,
        bbox_extra_artists=(legends, ytext, suptitle),
        bbox_inches="tight",
    )

    directory = folder_name+'/plots/'

    plt.savefig(directory+'_box.pdf')

    plt.close()