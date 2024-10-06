# data = categorized_models["Quarks"]

import json
import csv
import numpy as np
import re 

def data_reader(file):
    with open(file, 'r') as json_file:
        data = json.load(json_file)
    return data

def produce_model_string(json):

    layers = []
    first_layer = None
    last_layer = None

    for layer in json:
        if layer['class_name'] != 'QDense':
            continue

        if first_layer is None:
            first_layer = layer['input_shape'][1]

        layers.append(str(layer['input_shape'][1]))
        last_layer = layer['output_shape'][1]

    layers.append(str(last_layer))

    assert(first_layer != None and last_layer != None)

    return "-".join(layers), first_layer, last_layer


def parse_file(file):

    data = data_reader(file)

    print("Read in " + str(len(data)) + "models from JSON, parsing...")

    csv_header = [
        'model_name', 'd_in', 'd_out', 'prec', 'model_file', 'model_string', 'rf', 'strategy', 
        'TargetClockPeriod_hls', 'EstimatedClockPeriod_hls', 
        'BestLatency_hls', 'WorstLatency_hls', 'IntervalMin_hls', 'IntervalMax_hls', 
        'BRAM_18K_hls', 'DSP_hls', 'FF_hls', 'LUT_hls', 'URAM_hls', "rf_times_precision", "hls_synth_success"]

    prec = data[0]['hls_config']["LayerName"]
    i=True 
    for key in prec:
        if i: 
            i = False
            continue
        prec = prec[key]["Precision"]['weight']
        break
    prec = re.sub(",[0-9]+\>", "", prec)
    prec = re.sub("fixed\<", "", prec)

    with open("auto_parsed_json.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

        for i in range(len(data)):

            # Access the first element of the data
            data_point = data[i]
            meta_data = data_point['meta_data']

            model_name = meta_data['model_name']

            model_config = data_point['model_config']

            # get a string representing all layers of the model
            model_string, d_in, d_out = produce_model_string(model_config)

            # extract values from the HLS configuration
            hls_config = data_point['hls_config']

            prec = hls_config["LayerName"]
            d=True 
            for key in prec:
                if d: 
                    d = False
                    continue
                prec = prec[key]["Precision"]['weight']
                break
            prec = re.sub(",[0-9]+\>", "", prec)
            prec = re.sub("fixed\<", "", prec)

            model_file = meta_data['artifacts_file']

            rf = hls_config['Model']['ReuseFactor']

            strategy = hls_config['Model']['Strategy']

            # extract values from latency report
            latency_report = data_point['latency_report']

            target_clock = latency_report['target_clock']

            estimated_clock = latency_report['estimated_clock']

            best_latency = latency_report['cycles_min']

            worst_latency = latency_report['cycles_max']

            # extract values from resource report
            resource_report = data_point['resource_report']

            if 'BRAM' in resource_report:
                bram = resource_report['BRAM']
            else:
                bram = resource_report['bram']


            if 'DSP' in resource_report:
                dsp = resource_report['DSP']
            else:
                dsp = resource_report['dsp']


            if 'FF' in resource_report:
                ff = resource_report['FF']
            else:
                ff = resource_report['ff']


            if 'LUT' in resource_report:
                lut = resource_report['LUT']
            else:
                lut = resource_report['lut']


            if 'URAM' in resource_report:
                uram = resource_report['URAM']
            else:
                uram = resource_report['uram']

            rf_times_precision = int(prec) * int(rf)

            # only successes are in this dataset for now, trivially true for bookkeeping
            hls_synth_success = "TRUE"

            csv_row = [
            model_name, d_in, d_out, prec, model_file, model_string, rf, strategy,
            target_clock, estimated_clock, best_latency, worst_latency, best_latency, worst_latency,
            bram, dsp, ff, lut, uram, rf_times_precision, hls_synth_success]

            writer.writerow(csv_row)

    print('Parsing successful, file saved as "auto_parsed_json.csv"')