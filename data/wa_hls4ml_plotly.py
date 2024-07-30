import matplotlib.pyplot as plt
import numpy as np
import math
import os, sys


def plot_results(name, mpl_plots, y_test, y_pred, X_raw_test, output_features, folder_name):

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']

    if mpl_plots:
        # Iterate over each column
        print(y_test[0])
        print(y_test[1])
        for i in range(y_test.shape[1]):
            plt.figure(figsize=(10, 6))  # Create a new figure for each plot
            print(output_features[i])
            plt.scatter(y_test[:, i], y_pred[:, i], s=20, label=output_features[i])
            plt.title('Actual vs Predicted for ' + output_features[i])
            plt.xlabel('Actual Value')
            plt.ylabel('Predicted Value')
            plt.legend()

            # Plot a diagonal (for reference)
            plt.plot([np.min(y_test[:, i]), np.max(y_test[:, i])], [np.min(y_test[:, i]), np.max(y_test[:, i])], 'r')
            plt.tight_layout()
            plt.savefig(folder_name+"/plots/" + output_features[i] + '_predicted_vs_true.png')
            plt.show()


    import plotly.graph_objects as go
    import plotly.io as pio

    # Create a figure
    fig = go.Figure()

    marker_shapes = {0: 'star', 1: 'square'}
    strat_dict = {0: 'Latency', 1: 'Resource'}

    # Calculate the overall min and max values for the reference line
    overall_min = np.min(y_test)
    overall_max = np.max(y_test)
    n_features = len(output_features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)

    # Create a subplot
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=n_rows, cols=n_cols,
                           vertical_spacing=0.03, horizontal_spacing=0.03,
                            x_title='Actual Value',
                            y_title='Predicted Value',
                           subplot_titles=output_features)

    # Iterate over each column
    for i in range(n_features):
        # Calculate the current row and column
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Iterate over each strategy
        for strategy in [0, 1]:
            # Create a mask for the current strategy
            X_raw_test = np.asarray(X_raw_test)
            mask = X_raw_test[:,
                   strategy + 5] == 1  # assuming the 'strategy' is the 6th and 7th feature in your input data TODO: change this back for the fewer features
            text_arr = [
                f"{int(point[3])}-bit {int(point[0])}x{int(point[1])}x{int(point[2])} @ RF={int(point[4])} ({strat_dict[strategy]})" for
                point in X_raw_test if point[strategy + 5] == 1]

            # Create a scatter plot for each output feature and strategy
            scatter = go.Scatter(
                x=y_test[mask, i],
                y=y_pred[mask, i],
                mode='markers',
                name=f'{output_features[i]} - {strat_dict[strategy]}',
                legendgroup=f'{output_features[i]}',
                marker=dict(
                    symbol=marker_shapes[strategy],  # Use different marker shapes for different strategies
                    color=colors[i],
                    size=10,
                    opacity=0.7,
                ),
                hovertemplate=
                '%{text}<br>' +
                '<i>Actual</i>: %{x}<br>' +
                '<b>Predicted</b>: %{y}<br><extra></extra>',
                text=text_arr
            )
            # Add the scatter plot to the subplot
            fig.add_trace(scatter, row=row, col=col)
        #fig.update_yaxes(scaleanchor=f"x{col}", scaleratio=1, row=row, col=col)

        # Add a reference line
        fig.add_trace(
            go.Scatter(
                x=[np.min(y_test[:, i]), np.max(y_test[:, i])],
                y=[np.min(y_test[:, i]), np.max(y_test[:, i])],
                mode='lines',
                line=dict(color='black'),
                showlegend=False
            ),
            row=row, col=col
        )

    # Set the layout
    fig.update_layout(height=1900, width=1900, title='wa-hls4ml 1-Layer Dense Toy Model - Actual vs Predicted')

    directory = folder_name+'/plots/scatterplots/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    pio.write_html(fig, file=directory+name+'_wa-hls4ml_outputs.html', auto_open=False)
    # py.plot(fig, filename='wa-hls4ml_outputs', auto_open=False)
    # Show the plot
    # fig.show()
