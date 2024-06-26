from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import io
import subprocess
import config
from run_model_projection import WebAnalyzer

app = Flask(__name__)


roofline_data = {}
projection_data = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choices')
def choices():
    devices = list(config.HardwareParameters.keys())
    device_types = list(config.DeviceType2Ratio.keys())
    models = list(config.ModelDict.keys())
    data_types = list(config.DType2Bytes.keys())
    return jsonify({"devices": devices, "models": models, "device_types": device_types, "data_types": data_types})


@app.route('/plot', methods=['GET'])
def plot():
    global roofline_data
    device = "IntelGaudi2"
    device_type = "B"
    model = "Llama2-13B"
    data_type = "BF16"
    cfg = config.HardwareConfig(device, device_type, data_type)
    peak_flops = cfg.flops_mme
    peak_bandwidth = cfg.hbm_bandwidth
    arithmetic_intensity = peak_flops / peak_bandwidth
    attainable_tops = peak_flops
    if roofline_data:
        device = roofline_data["device"]
        device_type = roofline_data["device_type"]
        peak_flops = roofline_data["peak_flops"]
        peak_bandwidth = roofline_data["peak_bandwidth"]
        model = roofline_data["model"]
        data_type = roofline_data["data_type"]
        arithmetic_intensity = roofline_data["arithmetic_intensity"]
        attainable_tops = roofline_data["attainable_tops"]

        operational_intensity = np.logspace(-3, 3, 512)
        roofline = np.minimum(
            peak_flops, peak_bandwidth * operational_intensity)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=operational_intensity, y=roofline,
                      mode='lines', name=f'{device}{device_type} Roofline'))
        fig.add_trace(go.Scatter(x=[arithmetic_intensity], y=[attainable_tops], mode='markers+text', text=[
                      f"AI: {arithmetic_intensity:.2f}FLOPS/Bytes<br>\nTops: {attainable_tops/config.TFLOPS:.2f}TFLOPs"], textposition="top left", marker=dict(color='red')))
        # fig.add_trace(go.Scatter(x=[arithmetic_intensity], y=[attainable_tops], mode='markers+text', text=[
        #               f"Model: {model}<br>\nAI: {arithmetic_intensity:.2f}FLOPS/Bytes<br>\nTops: {attainable_tops/config.TFLOPS:.2f}TFLOPs"], textposition="top left", marker=dict(color='red')))

        fig.update_layout(
            xaxis=dict(type='log', title='Arithmetic Intensity (FLOPs/Byte)'),
            yaxis=dict(type='log', title='Performance (FLOPs/sec)'),
            title=f'Roofline Model for {device}{device_type}_{data_type}'
        )

        plot_html = fig.to_html(full_html=False)

        return plot_html
        '''
        plt.figure(figsize=(10, 6))
        plt.loglog(operational_intensity, roofline,
                   label=f'{device}{device_type}_{data_type} Roofline')
        plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
        plt.ylabel('Performance (FLOPs/sec)')
        plt.title(f'Roofline Model for {device}{device_type}')
        plt.legend()

        plt.scatter(arithmetic_intensity, attainable_tops,
                    color='red', zorder=5)
        # plt.text(arithmetic_intensity,  attainable_tops, f'{model}', fontsize=12, ha='right')
        plt.text(arithmetic_intensity,  attainable_tops,
                 f'Model: {model}\nArithmeticIntensity: {round(arithmetic_intensity, 2)}FLOPS/Bytes\nAttainableTops: {round(attainable_tops/config.TFLOPS, 2)}TFLOPs', fontsize=12, ha='left')
        plt.legend()
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return send_file(img, mimetype='image/png')
        '''
    else:
        return "No Roofline Data", 404


@app.route('/roofline', methods=['POST'])
def roofline():
    global roofline_data
    global projection_data
    device = request.form.get('device', 'IntelGaudi2')
    device_type = request.form.get('device_type', 'B')
    model = request.form.get('model', 'Llama2-7B')
    data_type = request.form.get('data_type', 'BF16')
    cfg = config.HardwareConfig(device, device_type, data_type)

    peak_flops = cfg.flops_mme
    peak_bandwidth = cfg.hbm_bandwidth

    operational_intensity = np.logspace(-3, 3, 512)
    roofline = np.minimum(peak_flops, peak_bandwidth * operational_intensity)

    arithmetic_intensity = peak_flops / peak_bandwidth
    attainable_tops = peak_flops
    if projection_data:
        arithmetic_intensity = projection_data[-1][9]
        attainable_tops = min(
            peak_flops, peak_bandwidth * arithmetic_intensity)

    roofline = {
        "device": device,
        "device_type": device_type,
        "peak_flops": peak_flops,
        "peak_bandwidth": peak_bandwidth,
        "model": model,
        "data_type": data_type,
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_tops": attainable_tops,
    }
    roofline_data = roofline  # Update global roofline data

    return jsonify({"message": "Roofline model completed successfully.", "data": roofline_data})


@app.route('/data', methods=['GET'])
def data():
    global projection_data
    if projection_data:
        columns = projection_data[0]  # First row as column headers
        data = projection_data[1:]    # Remaining rows as data

        # Convert to list of dictionaries for JSON serialization
        data_json = []
        for row in data:
            data_json.append(dict(zip(columns, row)))

        return jsonify(data_json)
    else:
        return jsonify([])


@app.route('/run_projection', methods=['POST'])
def run_projection():
    global projection_data
    device = request.form.get('device', 'IntelGaudi2')
    device_type = request.form.get('device_type', 'B')
    model = request.form.get('model', 'Llama2-7B')
    data_type = request.form.get('data_type', 'BF16')
    batch_size = int(request.form.get('batch_size', 64))
    context_input = int(request.form.get('context_input', 512))
    context_output = int(request.form.get('context_output', 1024))
    kvcache_bucket = int(request.form.get('kvcache_bucket', 256))

    proj_cfg = {
        "device_list": [device],
        "type_list": [device_type],
        "model_list": [model],
        "dtype_list": [data_type],
        "parallel": {
            "pp_list": [1],
            "tp_list": [1],
        },
        "context": {
            "input_list": [context_input],
            "output_list": [context_output],
        },
        "bs_list": [batch_size],
        "optims": {
            "kvcache_bucket": kvcache_bucket,
            "flash_attention": False,  # Todo
            "enable_vec_bmm": False,
        }
    }

    try:
        analyzer = WebAnalyzer(proj_cfg)
        proj_data = analyzer.analyze()
        projection_data = proj_data  # Update global projection data
        return jsonify({"message": "Model projection completed successfully.", "data": projection_data})
    except Exception as e:
        return jsonify({"message": f"Error running model projection: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
