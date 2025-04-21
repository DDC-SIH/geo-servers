import os
import json
import tempfile
import shutil
from urllib.parse import unquote
import requests
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.plot import reshape_as_image
from flask import Flask, request, jsonify, send_file
from concurrent.futures import ThreadPoolExecutor
import zipfile
import matplotlib.pyplot as plt

app = Flask(__name__)

def download_from_titiler(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download {url}")

@app.route('/download/raw', methods=['POST'])
def download_raw_layers():
    try:
        urls = request.json
        if not urls or not isinstance(urls, list):
            return jsonify({"error": "Expected a list of direct TiTiler URLs."}), 400

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "raw_layers.zip")

        def download_single(index, direct_url):
            path = os.path.join(temp_dir, f"layer_{index}.tiff")
            download_from_titiler(direct_url, path)
            return path

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_single, i, url) for i, url in enumerate(urls)]
            downloaded_files = [f.result() for f in futures]

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in downloaded_files:
                zipf.write(file_path, os.path.basename(file_path))

        return send_file(zip_path, as_attachment=True, download_name="raw_layers.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/download/layered', methods=['POST'])
def stack_layers():
    try:
        layers = request.json
        if not layers or not isinstance(layers, list):
            return jsonify({"error": "Expected a list of layer objects."}), 400

        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "stacked_output.tiff")

        sorted_layers = sorted(layers, key=lambda x: x.get('zIndex', 0))

        processed_layers = []
        ref_transform = ref_crs = ref_width = ref_height = None

        def download_layer(index, layer):
            url = layer.get('directURL')
            trans = float(layer.get('transparency', 1.0))
            path = os.path.join(temp_dir, f"layer_{index}.tiff")
            download_from_titiler(url, path)
            return path, trans

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_layer, i, l) for i, l in enumerate(sorted_layers)]
            results = [f.result() for f in futures]

        for path, transparency in results:
            with rasterio.open(path) as src:
                if ref_transform is None:
                    ref_transform = src.transform
                    ref_crs = src.crs
                    ref_width = src.width
                    ref_height = src.height
            processed_layers.append({"file": path, "transparency": transparency})

        stacked_data = np.zeros((ref_height, ref_width, 4), dtype=np.uint8)

        for layer_info in processed_layers:
            with rasterio.open(layer_info['file']) as src:
                if src.count >= 3:
                    rgb = src.read([1, 2, 3])
                    layer_rgb = reshape_as_image(rgb)
                else:
                    data = src.read(1)
                    layer_rgb = np.stack([data]*3, axis=2)

                if layer_rgb.dtype != np.uint8:
                    layer_rgb = ((layer_rgb / layer_rgb.max()) * 255).astype(np.uint8)

                alpha = np.ones((ref_height, ref_width), dtype=np.uint8) * 255
                alpha = (alpha * layer_info['transparency']).astype(np.uint8)

                src_alpha = alpha.astype(float) / 255
                dst_alpha = stacked_data[..., 3].astype(float) / 255
                out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
                mask = out_alpha > 0

                for c in range(3):
                    stacked_data[..., c] = np.where(
                        mask,
                        (layer_rgb[..., c] * src_alpha + stacked_data[..., c] * dst_alpha * (1 - src_alpha)) / out_alpha,
                        stacked_data[..., c]
                    ).astype(np.uint8)

                stacked_data[..., 3] = (out_alpha * 255).astype(np.uint8)

        rgb_data = stacked_data[..., :3].transpose(2, 0, 1)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ref_height,
            width=ref_width,
            count=3,
            dtype=rgb_data.dtype,
            crs=ref_crs,
            transform=ref_transform
        ) as dst:
            dst.write(rgb_data)

        return send_file(output_path, as_attachment=True, download_name="stacked_layers.tiff")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)