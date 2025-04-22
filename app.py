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
from flasgger import Swagger, swag_from
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import os
import requests
import datetime
from PIL import Image
from typing import List


app = Flask(__name__)
CORS(app)

# Configure Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs/"
}

swagger_template = {
    "info": {
        "title": "Geo Server API",
        "description": "API for geospatial data operations, including raster processing and GIF generation",
        "version": "1.0.0",
        "contact": {
            "email": "souradip1000@gmail.com"
        }
    },
    "tags": [
        {
            "name": "Download",
            "description": "Operations for downloading and processing raster data"
        },
        {
            "name": "GIF Generation",
            "description": "Operations for creating animated GIFs from raster time series"
        }
    ],
    "servers": [
        {
            "url": "http://74.226.242.56:5000",
            "description": "Production server"
        }
    ],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)
TITILER_BASE = "http://74.226.242.56:8000/cog"
METADATA_API = "http://74.226.242.56:7000/api/metadata/{sat}/cog/range"

def convert_epoch_to_datetime(epoch_ms: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(epoch_ms / 1000)

def get_filtered_cogs(input_data: dict) -> List[dict]:
    from dateutil.parser import isoparse
    start = isoparse(input_data["startDateTime"]).replace(tzinfo=None)
    end = isoparse(input_data["endDateTime"]).replace(tzinfo=None)
    interval = int(input_data["interval"])
    band_names = input_data["bandName"] if isinstance(input_data["bandName"], list) else [input_data["bandName"]]

    url = METADATA_API.format(sat=input_data["SatelliteId"])
    params = {
        "start": input_data["startDateTime"],
        "end": input_data["endDateTime"],
        "processingLevel": input_data["processingLevel"],
        "productCode": input_data["productType"]
    }
    if len(band_names) == 1:
        params["type"] = band_names[0]
    
    print(f"Fetching metadata from: {url} with params: {params}")

    print(f"Fetching metadata from: {url} with params: {params}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    cogs = response.json().get("cogs", [])
    print(response.json())
    print(f"Received {len(cogs)} COGs")

    sorted_cogs = sorted(cogs, key=lambda c: c["aquisition_datetime"])
    selected = []
    current_time = start

    for cog in sorted_cogs:
        cog_time = convert_epoch_to_datetime(cog["aquisition_datetime"])
        if cog_time >= current_time and cog_time <= end:
            if len(band_names) > 1 and cog.get("type") != "MULTI":
                continue
            selected.append(cog)
            current_time = cog_time + datetime.timedelta(hours=interval)

    print(f"Selected {len(selected)} COGs for processing")
    return selected

def generate_titiler_url(cog: dict, input_data: dict) -> str:
    bbox = input_data.get("bbox")
    print(f"Using bbox: {bbox}" if bbox else "No bbox provided, using preview")
    color_map = input_data.get("colourmap")  # Make colourmap optional
    file_path = os.path.join(cog["filepath"], cog["filename"])

    # Build the base url with either bbox or preview endpoint
    if bbox:
        bbox_str = f"{bbox['minx']},{bbox['miny']},{bbox['maxx']},{bbox['maxy']}"
        base_url = f"{TITILER_BASE}/bbox/{bbox_str}.png"
    else:
        base_url = f"{TITILER_BASE}/preview"

    # Build query parameters
    params = [f"url={file_path}"]

    # Add band indices from COG metadata
    band_params = []
    if len(input_data["bandName"]) > 1:
        for band_name in input_data["bandName"]:
            for band in cog["bands"]:
                if band["description"] == band_name or band["description"] == f"IMG_{band_name}":
                    band_params.append(f"bidx={band['bandId']}")
    else:
        for band in cog["bands"]:
            band_params.append(f"bidx={band['bandId']}")

    params.extend(band_params)

    # Add optional colormap if provided
    if color_map:
        params.append(f"colormap_name={color_map}")

    # Combine base url with query parameters
    url = f"{base_url}?{'&'.join(params)}"
    print(f"Generated TiTiler URL: {url}")
    return url

def download_image(url: str) -> Image.Image:
    print(f"Downloading image from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    print("✅ Image downloaded and converted")
    return img

def build_gif(images: List[Image.Image]) -> BytesIO:
    gif_bytes = BytesIO()
    if images:
        print(f"Building GIF with {len(images)} frames")
        images[0].save(
            gif_bytes,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )
    else:
        print("⚠️ No images to include in GIF")
    gif_bytes.seek(0)
    return gif_bytes


def download_from_titiler(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download {url}")

@app.route('/download/raw', methods=['POST'])
@swag_from({
    'tags': ['Download'],
    'description': 'Download multiple raster layers from direct TiTiler URLs and zip them.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'string',
                    'example': 'http://127.0.0.1:8000/cog/bbox/72.0,15.0,78.0,25.0.tif?url=/path/to/file.cog.tif&bidx=1&bidx=2&bidx=3'
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Zipped file of all layers',
            'content': {
                'application/zip': {
                    'schema': {
                        'type': 'string',
                        'format': 'binary'
                    }
                }
            }
        }
    }
})
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
@swag_from({
    'tags': ['Download'],
    'description': 'Download and stack multiple raster layers using alpha transparency and zIndex.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'format': {
    'type': 'string',
    'example': 'png',
    'enum': ['tiff', 'tif', 'png', 'jpeg', 'jpg', 'webp', 'npy'],
    'description': 'Supported formats: tiff, tif, png, jpeg, jpg, webp, npy'
},
                    'data': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'directURL': {'type': 'string'},
                                'zIndex': {'type': 'integer'},
                                'transparency': {'type': 'number', 'format': 'float'}
                            },
                            'required': ['directURL']
                        }
                    }
                },
                'required': ['data']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Stacked raster image',
            'content': {
                '*/*': {
                    'schema': {
                        'type': 'string',
                        'format': 'binary'
                    }
                }
            }
        }
    }
})
def stack_layers():
    try:
        req = request.json
        if not req or not isinstance(req, dict) or 'data' not in req:
            return jsonify({"error": "Expected a JSON object with 'data' list and optional 'format'."}), 400

        layers = req['data']
        output_format = req.get('format', 'tiff').lower()

        if not layers or not isinstance(layers, list):
            return jsonify({"error": "'data' should be a list of layer objects."}), 400

        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"stacked_output.{output_format if output_format != 'tif' else 'tiff'}")

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

        rgb_data = stacked_data[..., :3]

        if output_format in ["tiff", "tif"]:
            rgb_data = rgb_data.transpose(2, 0, 1)
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

        elif output_format in ["jpeg", "jpg", "png", "webp"]:
            image = Image.fromarray(rgb_data)
            save_format = "JPEG" if output_format == "jpg" else output_format.upper()
            image.save(output_path, format=save_format)


        elif output_format == "npy":
            np.save(output_path, rgb_data)

        else:
            return jsonify({"error": f"Unsupported format: {output_format}"}), 400

        return send_file(output_path, as_attachment=True, download_name=f"stacked_layers.{output_format}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route("/generate-gif", methods=["POST"])
@swag_from({
    'tags': ['GIF Generation'],
    'summary': 'Generate animated GIF from selected COG raster layers',
    'description': 'Generates a GIF by fetching raster images (COG) based on date range, satellite ID, band, and other filtering parameters.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'SatelliteId': {
                        'type': 'string',
                        'example': '3R',
                        'description': 'Satellite identifier (e.g., "3R")'
                    },
                    'startDateTime': {
                        'type': 'string',
                        'format': 'date-time',
                        'example': '2025-03-22T00:00:00',
                        'description': 'Start datetime for filtering COGs (ISO format)'
                    },
                    'endDateTime': {
                        'type': 'string',
                        'format': 'date-time',
                        'example': '2025-03-22T00:05:45',
                        'description': 'End datetime for filtering COGs (ISO format)'
                    },
                    'interval': {
                        'type': 'integer',
                        'example': 1,
                        'description': 'Hour-based interval between selected COGs'
                    },
                    'processingLevel': {
                        'type': 'string',
                        'example': 'L1C',
                        'description': 'Processing level of satellite data'
                    },
                    'productType': {
                        'type': 'string',
                        'example': 'ASIA_MER',
                        'description': 'Product type code (e.g., "ASIA_MER")'
                    },
                    'bandName': {
                        'type': 'string',
                        'example': 'TIR2',
                        'description': 'Spectral band name (e.g., "TIR2")'
                    },
                    'bbox': {
                        'type': 'object',
                        'properties': {
                            'minx': {'type': 'number', 'example': 72.5},
                            'miny': {'type': 'number', 'example': 15.5},
                            'maxx': {'type': 'number', 'example': 88.5},
                            'maxy': {'type': 'number', 'example': 27.5}
                        },
                        'required': ['minx', 'miny', 'maxx', 'maxy'],
                        'description': 'Bounding box for clipping the raster (WGS84 coordinates)'
                    },
                    'colourmap': {
                        'type': 'string',
                        'example': 'viridis',
                        'description': 'Color map name used to render the imagery'
                    }
                },
                'required': [
                    'SatelliteId', 'startDateTime', 'endDateTime', 'interval',
                    'processingLevel', 'productType', 'bandName', 'colourmap'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Animated GIF created from selected raster images',
            'content': {
                'image/gif': {
                    'schema': {
                        'type': 'string',
                        'format': 'binary'
                    }
                }
            }
        },
        400: {
            'description': 'Invalid request format or missing required fields'
        },
        500: {
            'description': 'Internal server error during GIF generation'
        }
    }
})
def generate_gif_endpoint():
    input_data = request.get_json()
    print("Received request with data:", input_data)
    try:
        selected_cogs = get_filtered_cogs(input_data)
        images = []
        for cog in selected_cogs:
            try:
                titiler_url = generate_titiler_url(cog, input_data)
                img = download_image(titiler_url)
                images.append(img)
                print(f"✅ Added image from {cog['filename']} to GIF")
            except Exception as img_err:
                print(f"❌ Error downloading image for {cog['filename']}: {img_err}")
        gif_buffer = build_gif(images)
        return send_file(gif_buffer, mimetype="image/gif", download_name="output.gif")
    except Exception as e:
        print(f"❌ Error in generate_gif_endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
