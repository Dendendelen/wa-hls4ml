# wa-hls4ml

wa-hls4ml: A Graph Neural Network surrogate model for hls4ml

File structure:

wa_hls4ml/py: Command line interface

data/
> wa_hls4ml_data_processing.py: Input data formatting and graph-conversion

> wa_hls4ml_data_plot.py: Plotting model learning and output, and residual histograms

> wa_hls4ml_plotly.py: Plotly code for generating interactive prediction scatterplots

model/
> wa_hls4ml_model.py: Actual GNN and control MLP model

> wa_hls4ml_train.py: Training a model

> wa_hls4ml_test.py: Testing a model
