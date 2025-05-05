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
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import os
import requests
import datetime
from PIL import Image
from typing import List
from urllib.parse import quote_plus
import torch
from torch import nn
import torch.nn.functional as F
from pyproj import Transformer
from rasterio.crs import CRS

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

# Add these imports at the top of the file
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import os.path
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

# Add this function after existing imports
import shapely.ops  # Add this import if not already present

def overlay_shapefile(img, transform, crs, show_country=True, show_states=True, line_width=1.0):
    """
    Overlay shapefile boundaries on an image with improved visibility
    
    Args:
        img: PIL Image to overlay boundaries on
        transform: rasterio transform of the image
        crs: coordinate reference system of the image
        show_country: whether to show country boundary
        show_states: whether to show state boundaries
        line_width: width of boundary lines
        
    Returns:
        PIL Image with shapefile boundaries overlaid
    """
    try:
        # Get image dimensions
        width, height = img.size
        
        # Create a transparent overlay image first
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Correct shapefile paths
        country_path = "/home/sbn/souradip/geo-servers/shapefiles/India_Country_Boundary.shp"
        state_path = "/home/sbn/souradip/geo-servers/shapefiles/India_State_Boundary.shp"
        
        # Ensure CRS is defined
        if crs is None:
            print("Warning: CRS is None, defaulting to EPSG:4326")
            crs = CRS.from_epsg(4326)
        
        # Draw country boundary
        if show_country and os.path.exists(country_path):
            try:
                country_gdf = gpd.read_file(country_path)
                
                # Ensure the shapefile has a CRS set
                if country_gdf.crs is None:
                    print("Setting CRS for country shapefile to EPSG:4326")
                    country_gdf.set_crs(epsg=4326, inplace=True)
                
                # Reproject to the image CRS if needed
                if country_gdf.crs != crs:
                    try:
                        country_gdf = country_gdf.to_crs(crs)
                    except Exception as e:
                        print(f"Error reprojecting country boundary: {e}")
                        # Try using pyproj directly
                        from pyproj import Transformer
                        transformer = Transformer.from_crs(country_gdf.crs, crs, always_xy=True)
                        
                        def transform_geom(geom):
                            if geom is None or not geom.is_valid:
                                return None
                            return shapely.ops.transform(
                                lambda x, y: transformer.transform(x, y), 
                                geom
                            )
                        
                        country_gdf['geometry'] = country_gdf['geometry'].apply(transform_geom)
                        country_gdf = country_gdf[country_gdf.geometry.notnull()]
                        if len(country_gdf) > 0:
                            country_gdf.set_crs(crs, inplace=True)
                
                # Directly draw each polygon's exterior on our overlay using PIL
                for _, row in country_gdf.iterrows():
                    geom = row['geometry']
                    if geom is None or not hasattr(geom, 'exterior'):
                        continue
                        
                    # Get the exterior coordinates of the polygon
                    if hasattr(geom, 'geoms'):  # MultiPolygon
                        for subgeom in geom.geoms:
                            if not hasattr(subgeom, 'exterior'):
                                continue
                            coords = subgeom.exterior.coords
                            pixel_coords = [~transform * (x, y) for x, y in coords]
                            # Draw with thicker blue line for country boundary
                            draw.line(pixel_coords, fill=(0, 0, 255, 255), width=int(line_width*3))
                    else:  # Single Polygon
                        coords = geom.exterior.coords
                        pixel_coords = [~transform * (x, y) for x, y in coords]
                        # Draw with thicker blue line for country boundary
                        draw.line(pixel_coords, fill=(0, 0, 255, 255), width=int(line_width*3))
                
            except Exception as e:
                print(f"Error plotting country boundary: {e}")
        
        # Draw state boundaries
        if show_states and os.path.exists(state_path):
            try:
                state_gdf = gpd.read_file(state_path)
                
                # Ensure the shapefile has a CRS set
                if state_gdf.crs is None:
                    print("Setting CRS for state shapefile to EPSG:4326")
                    state_gdf.set_crs(epsg=4326, inplace=True)
                
                # Reproject to the image CRS if needed
                if state_gdf.crs != crs:
                    try:
                        state_gdf = state_gdf.to_crs(crs)
                    except Exception as e:
                        print(f"Error reprojecting state boundaries: {e}")
                        # Try using pyproj directly
                        from pyproj import Transformer
                        transformer = Transformer.from_crs(state_gdf.crs, crs, always_xy=True)
                        
                        def transform_geom(geom):
                            if geom is None or not geom.is_valid:
                                return None
                            return shapely.ops.transform(
                                lambda x, y: transformer.transform(x, y), 
                                geom
                            )
                        
                        state_gdf['geometry'] = state_gdf['geometry'].apply(transform_geom)
                        state_gdf = state_gdf[state_gdf.geometry.notnull()]
                        if len(state_gdf) > 0:
                            state_gdf.set_crs(crs, inplace=True)
                
                # Directly draw each polygon's exterior on our overlay
                for _, row in state_gdf.iterrows():
                    geom = row['geometry']
                    if geom is None or not hasattr(geom, 'exterior'):
                        continue
                        
                    # Get the exterior coordinates of the polygon
                    if hasattr(geom, 'geoms'):  # MultiPolygon
                        for subgeom in geom.geoms:
                            if not hasattr(subgeom, 'exterior'):
                                continue
                            coords = subgeom.exterior.coords
                            pixel_coords = [~transform * (x, y) for x, y in coords]
                            # Draw with white line for state boundaries
                            draw.line(pixel_coords, fill=(255, 255, 255, 255), width=int(line_width*2))
                    else:  # Single Polygon
                        coords = geom.exterior.coords
                        pixel_coords = [~transform * (x, y) for x, y in coords]
                        # Draw with white line for state boundaries
                        draw.line(pixel_coords, fill=(255, 255, 255, 255), width=int(line_width*2))
                
            except Exception as e:
                print(f"Error plotting state boundaries: {e}")
        
        # Add a simple legend to the bottom right
        if show_country or show_states:
            legend_width = 150
            legend_height = 50
            legend_margin = 10
            legend_x = width - legend_width - legend_margin
            legend_y = height - legend_height - legend_margin
            
            # Draw semi-transparent white background for legend
            draw.rectangle([(legend_x, legend_y), 
                           (legend_x + legend_width, legend_y + legend_height)], 
                          fill=(255, 255, 255, 180))
            
            # Add legend entries
            if show_country:
                # Country boundary line
                draw.line([(legend_x + 10, legend_y + 15), 
                          (legend_x + 40, legend_y + 15)], 
                         fill=(0, 0, 255, 255), width=3)
                # Country label
                draw.text((legend_x + 50, legend_y + 10), 
                         "Country", fill=(0, 0, 0, 255))
            
            if show_states:
                # State boundary line
                draw.line([(legend_x + 10, legend_y + 35), 
                          (legend_x + 40, legend_y + 35)], 
                         fill=(255, 255, 255, 255), width=2)
                # State label
                draw.text((legend_x + 50, legend_y + 30), 
                         "States", fill=(0, 0, 0, 255))
        
        # Composite the original image with our overlay
        result_img = Image.alpha_composite(img.convert("RGBA"), overlay)
        
        return result_img
    
    except Exception as e:
        print(f"Error in overlay_shapefile: {e}")
        import traceback
        traceback.print_exc()
        # Return the original image if overlay fails
        return img

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
            path = os.path.join(temp_dir, f"layer_{index}_original.tiff")
            download_from_titiler(direct_url, path)
            
            # Add white header region with file details and ISRO logo
            try:
                # Open the downloaded image
                with rasterio.open(path) as src:
                    # Get transform and CRS for shapefile overlay
                    transform = src.transform
                    crs = src.crs
                    
                    # Read image data
                    if src.count >= 3:
                        img_data = src.read([1, 2, 3])
                        img_data = reshape_as_image(img_data)
                    else:
                        img_data = src.read(1)
                        img_data = np.stack([img_data]*3, axis=2)
                    
                    # Convert to 8-bit if needed
                    if img_data.dtype != np.uint8:
                        img_data = ((img_data / img_data.max()) * 255).astype(np.uint8)
                    
                    # Convert to PIL image
                    img = Image.fromarray(img_data)
                    
                    # Get image dimensions
                    img_width, img_height = img.size
                    
                    # Create a larger canvas with white header
                    canvas = Image.new('RGBA', (img_width, img_height + 100), (255, 255, 255, 255))
                    canvas.paste(img, (0, 100))  # Paste original image below the header
                    
                    # Create a drawing context
                    draw = ImageDraw.Draw(canvas)
                    
                    # Load a font
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 14)
                        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
                    except IOError:
                        # Fallback to default font
                        font = ImageFont.load_default()
                        small_font = ImageFont.load_default()
                    
                    # Parse URL to get file details
                    filename = f"Layer {index}"
                    url_parts = direct_url.split('?')
                    
                    # Extract product information from URL
                    satellite_id = "Unknown"
                    product_type = "Unknown"
                    band_name = "Unknown"
                    
                    if len(url_parts) > 1:
                        query_params = url_parts[1].split('&')
                        for param in query_params:
                            if param.startswith('url='):
                                path_param = unquote(param[4:])
                                filename = os.path.basename(path_param)
                                
                                # Try to extract product information from filename
                                name_parts = filename.split('_')
                                if len(name_parts) >= 2:
                                    satellite_id = name_parts[0]
                                if len(name_parts) >= 3:
                                    product_type = name_parts[1]
                                if "band" in direct_url.lower() or "bidx" in direct_url.lower():
                                    for p in query_params:
                                        if p.startswith('bidx='):
                                            band_name = f"Band {p.split('=')[1]}"
                                            break
                    
                    # Current date and time
                    import datetime
                    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create file information string
                    file_info = f"Satellite: {satellite_id} | Product: {product_type} | Band: {band_name}"
                    
                    # Draw text for datetime and file info (on the left side)
                    draw.text((10, 10), current_datetime, fill=(0, 0, 0, 255), font=font)
                    draw.text((10, 40), file_info, fill=(0, 0, 0, 255), font=small_font)
                    draw.text((10, 65), f"File: {filename}", fill=(0, 0, 0, 255), font=small_font)
                    
                    # Add the ISRO logo in the top-right corner of the header
                    logo_path = "/home/sbn/souradip/geo-servers/Indian_Space_Research_Organisation_Logo.svg.png"
                    if os.path.exists(logo_path):
                        try:
                            logo_img = Image.open(logo_path).convert("RGBA")
                            # Resize logo to appropriate size
                            logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                            logo_position = (img_width - logo_img.width - 10, 10)  # Top-right position
                            canvas.paste(logo_img, logo_position, logo_img)
                        except Exception as e:
                            print(f"Error loading ISRO logo: {e}")
                    
                    # Apply shapefile overlay to the image portion only (below the header)
                    try:
                        img_with_overlay = overlay_shapefile(
                            img, 
                            transform, 
                            crs,
                            show_country=True,
                            show_states=True
                        )
                        
                        # Replace the image portion in the canvas with the overlaid image
                        canvas.paste(img_with_overlay, (0, 100))
                    except Exception as e:
                        print(f"Error applying shapefile overlay: {e}")
                    
                    # Save the enhanced image
                    enhanced_path = os.path.join(temp_dir, f"layer_{index}.tiff")
                    canvas.save(enhanced_path, format="TIFF")
                    
                    return enhanced_path
                
            except Exception as e:
                print(f"Error adding header to image {index}: {e}")
                # If enhancement fails, return the original file
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

# Now modify the stack_layers function to include shapefile overlay
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
        stacked_path = os.path.join(temp_dir, f"stacked_output_original.{output_format if output_format != 'tif' else 'tiff'}")
        output_path = os.path.join(temp_dir, f"stacked_output.{output_format if output_format != 'tif' else 'tiff'}")

        sorted_layers = sorted(layers, key=lambda x: x.get('zIndex', 0))

        processed_layers = []
        ref_transform = ref_crs = ref_width = ref_height = None
        
        # Extract product information from first URL when available
        satellite_id = "Unknown"
        product_type = "Unknown"
        band_name = "Unknown"
        
        if sorted_layers and 'directURL' in sorted_layers[0]:
            url = sorted_layers[0]['directURL']
            url_parts = url.split('?')
            
            if len(url_parts) > 1:
                query_params = url_parts[1].split('&')
                for param in query_params:
                    if param.startswith('url='):
                        path_param = unquote(param[4:])
                        filename = os.path.basename(path_param)
                        
                        # Try to extract product information from filename
                        name_parts = filename.split('_')
                        if len(name_parts) >= 2:
                            satellite_id = name_parts[0]
                        if len(name_parts) >= 3:
                            product_type = name_parts[1]
                        if "band" in url.lower() or "bidx" in url.lower():
                            for p in query_params:
                                if p.startswith('bidx='):
                                    band_name = f"Band {p.split('=')[1]}"
                                    break

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

        # First create the basic version
        if output_format in ["tiff", "tif"]:
            rgb_data_transposed = rgb_data.transpose(2, 0, 1)
            with rasterio.open(
                stacked_path,
                'w',
                driver='GTiff',
                height=ref_height,
                width=ref_width,
                count=3,
                dtype=rgb_data_transposed.dtype,
                crs=ref_crs,
                transform=ref_transform
            ) as dst:
                dst.write(rgb_data_transposed)

        elif output_format in ["jpeg", "jpg", "png", "webp"]:
            image = Image.fromarray(rgb_data)
            save_format = "JPEG" if output_format == "jpg" else output_format.upper()
            image.save(stacked_path, format=save_format)

        elif output_format == "npy":
            np.save(stacked_path, rgb_data)

        else:
            return jsonify({"error": f"Unsupported format: {output_format}"}), 400
        
        # Now enhance the image with header
        try:
            # Create header region for the stacked image
            if output_format in ["jpeg", "jpg", "png", "webp", "tiff", "tif"]:
                # Load the stacked image
                if output_format in ["tiff", "tif"]:
                    with rasterio.open(stacked_path) as src:
                        # Get transform and CRS for shapefile overlay
                        img_transform = src.transform
                        img_crs = src.crs
                        
                        if src.count >= 3:
                            img_data = src.read([1, 2, 3])
                            img_data = reshape_as_image(img_data)
                        else:
                            img_data = src.read(1)
                            img_data = np.stack([img_data]*3, axis=2)
                        
                        if img_data.dtype != np.uint8:
                            img_data = ((img_data / img_data.max()) * 255).astype(np.uint8)
                        
                        img = Image.fromarray(img_data)
                else:
                    img = Image.open(stacked_path)
                    img_transform = ref_transform
                    img_crs = ref_crs
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Apply shapefile overlay before adding header
                try:
                    img_with_overlay = overlay_shapefile(
                        img, 
                        img_transform, 
                        img_crs,
                        show_country=True,
                        show_states=True
                    )
                    img = img_with_overlay
                except Exception as e:
                    print(f"Error applying shapefile overlay to stacked image: {e}")
                
                # Create a larger canvas with white header
                canvas = Image.new('RGBA', (img_width, img_height + 100), (255, 255, 255, 255))
                canvas.paste(img, (0, 100))  # Paste original image with overlay below the header
                
                # Create a drawing context
                draw = ImageDraw.Draw(canvas)
                
                # Load a font
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 14)
                    small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
                except IOError:
                    # Fallback to default font
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                
                # Current date and time
                import datetime
                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create product information string
                product_info = f"Satellite: {satellite_id} | Product: {product_type} | Band: {band_name}"
                
                # Layer information
                layer_count_info = f"Stacked Image: {len(layers)} layers"
                
                # Get transparency values as a string
                transparencies = [f"{layer.get('transparency', 1.0):.2f}" for layer in sorted_layers]
                transparency_info = f"Transparencies: {', '.join(transparencies)}"
                
                # Draw text information
                draw.text((10, 10), current_datetime, fill=(0, 0, 0, 255), font=font)
                draw.text((10, 40), product_info, fill=(0, 0, 0, 255), font=small_font)
                draw.text((10, 65), f"{layer_count_info} | {transparency_info}", fill=(0, 0, 0, 255), font=small_font)
                
                # Add the ISRO logo in the top-right corner of the header
                logo_path = "/home/sbn/souradip/geo-servers/Indian_Space_Research_Organisation_Logo.svg.png"
                if os.path.exists(logo_path):
                    try:
                        logo_img = Image.open(logo_path).convert("RGBA")
                        # Resize logo to appropriate size
                        logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                        logo_position = (img_width - logo_img.width - 10, 10)  # Top-right position
                        canvas.paste(logo_img, logo_position, logo_img)
                    except Exception as e:
                        print(f"Error loading ISRO logo: {e}")
                
                # Save the enhanced image
                if output_format in ["tiff", "tif"]:
                    # Convert to RGB if needed before saving
                    if canvas.mode != 'RGB':
                        canvas = canvas.convert('RGB')
                    canvas.save(output_path, format="TIFF")
                else:
                    save_format = "JPEG" if output_format == "jpg" else output_format.upper()
                    canvas.save(output_path, format=save_format)
            else:
                # For non-image formats like NPY, just use the original output
                output_path = stacked_path
                
        except Exception as e:
            print(f"Error adding header to stacked image: {e}")
            # If enhancement fails, use the original stacked image
            output_path = stacked_path

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
        
        # Handle case when no COGs are found
        if not selected_cogs:
            print("No COGs found for the specified criteria")
            # Create a blank image with information about the query
            info_img = Image.new('RGBA', (800, 400), (255, 255, 255, 255))
            draw = ImageDraw.Draw(info_img)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
                small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw text with information about the request
            draw.text((50, 50), "No Data Available", fill=(0, 0, 0, 255), font=font)
            draw.text((50, 100), f"Satellite ID: {input_data.get('SatelliteId', 'N/A')}", fill=(0, 0, 0, 255), font=small_font)
            draw.text((50, 130), f"Processing Level: {input_data.get('processingLevel', 'N/A')}", fill=(0, 0, 0, 255), font=small_font)
            draw.text((50, 160), f"Product Type: {input_data.get('productType', 'N/A')}", fill=(0, 0, 0, 255), font=small_font)
            draw.text((50, 190), f"Band: {input_data.get('bandName', 'N/A')}", fill=(0, 0, 0, 255), font=small_font)
            draw.text((50, 220), f"Time Range: {input_data.get('startDateTime', 'N/A')} to {input_data.get('endDateTime', 'N/A')}", 
                     fill=(0, 0, 0, 255), font=small_font)
            draw.text((50, 280), "No data found for the specified time period and parameters.", 
                     fill=(255, 0, 0, 255), font=small_font)
            draw.text((50, 310), "Please adjust your search criteria and try again.", 
                     fill=(255, 0, 0, 255), font=small_font)
            
            # Create GIF from the single info image
            gif_buffer = BytesIO()
            info_img.save(gif_buffer, format="GIF")
            gif_buffer.seek(0)
            
            return send_file(gif_buffer, mimetype="image/gif", download_name="no_data.gif")
        
        enhanced_images = []
        
        # Check if ISRO logo file exists, otherwise download it
        logo_path = "/home/sbn/souradip/geo-servers/Indian_Space_Research_Organisation_Logo.svg.png"
        if not os.path.exists(logo_path):
            # Try a different URL with User-Agent header to avoid 403 error
            logo_urls = [
                "https://www.isro.gov.in/media_isro/image/index/isro-logo.png",
                "https://www.isro.gov.in/sites/default/files/2022-07/isro_logo_0.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg"
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            for logo_url in logo_urls:
                try:
                    logo_response = requests.get(logo_url, headers=headers)
                    if logo_response.status_code == 200:
                        with open(logo_path, 'wb') as f:
                            f.write(logo_response.content)
                        print(f"✅ Downloaded ISRO logo to {logo_path}")
                        break
                    else:
                        print(f"⚠️ Could not download ISRO logo from {logo_url}, status code: {logo_response.status_code}")
                except Exception as logo_err:
                    print(f"⚠️ Error downloading ISRO logo from {logo_url}: {logo_err}")
            
            # If all download attempts fail, create a basic placeholder logo
            if not os.path.exists(logo_path):
                print("Creating placeholder ISRO logo")
                placeholder_logo = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
                draw = ImageDraw.Draw(placeholder_logo)
                
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                
                draw.text((40, 80), "ISRO", fill=(0, 0, 255, 255), font=font)
                draw.text((25, 110), "Indian Space", fill=(0, 0, 0, 255), font=font)
                draw.text((25, 140), "Research Org.", fill=(0, 0, 0, 255), font=font)
                
                # Draw a blue circle around the text
                draw.ellipse((10, 10, 190, 190), outline=(0, 0, 255, 255), width=3)
                
                placeholder_logo.save(logo_path)
                print("✅ Created placeholder ISRO logo")
        
        # Load the ISRO logo as a PIL Image
        logo_img = None
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                # Resize logo to appropriate size for header (smaller than before)
                logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                print("✅ Loaded ISRO logo")
            except Exception as e:
                print(f"⚠️ Error loading ISRO logo: {e}")
                logo_img = None
        
        # Get statistical information for the first COG to create appropriate legend ranges
        cog_stats = None
        legend_img = None
        
        try:
            # Get stats for the first COG to determine value ranges for the legend
            if selected_cogs:
                first_cog = selected_cogs[0]
                first_cog_path = os.path.join(first_cog['filepath'], first_cog['filename'])
                first_cog_url = f"file://{first_cog_path}"
                
                # Get stats from TiTiler
                stats_url = f"{TITILER_BASE}/statistics?url={quote_plus(first_cog_url)}"
                print(f"Fetching statistics from: {stats_url}")
                stats_response = requests.get(stats_url)
                
                if stats_response.status_code == 200:
                    cog_stats = stats_response.json()
                    print(f"✅ Got statistics for legend: {cog_stats}")
                    
                    # Extract min, max, and histogram information
                    band_key = list(cog_stats.keys())[0]  # Usually 'b1'
                    band_stats = cog_stats[band_key]
                    
                    # Get min/max values directly from statistics
                    min_val = band_stats.get('min', 0)
                    max_val = band_stats.get('max', 255)
                    print(f"Min value: {min_val}, Max value: {max_val}")
                    
                    # Create value ranges for legend
                    value_ranges = []
                    if 'histogram' in band_stats and len(band_stats['histogram']) > 1:
                        bins = band_stats['histogram'][1]
                        value_ranges = [[bins[i], bins[i+1]] for i in range(len(bins)-1)]
                    else:
                        # Create default ranges if histogram not available
                        step = (max_val - min_val) / 10
                        value_ranges = [[min_val + step*i, min_val + step*(i+1)] for i in range(10)]
                    
                    # Generate colors for each range based on the selected colormap
                    colormap = input_data.get('colourmap', 'viridis')
                    
                    # Import matplotlib for colormap generation
                    import matplotlib.pyplot as plt
                    from matplotlib import cm as mpl_cm
                    
                    # Get the colormap from matplotlib
                    cmap = plt.get_cmap(colormap)
                    
                    # Create colors for each bin
                    # Use the start value of each range to get a color
                    start_values = [r[0] for r in value_ranges]
                    
                    # Normalize values to 0-1 range for colormap
                    if min_val != max_val:
                        norm_values = [(v - min_val) / (max_val - min_val) for v in start_values]
                    else:
                        norm_values = [0.5] * len(start_values)
                    
                    # Get RGBA colors from colormap
                    rgba_colors = [cmap(v) for v in norm_values]
                    
                    # Convert to 0-255 range integers for PIL
                    colors = [
                        [int(r*255), int(g*255), int(b*255), 255] 
                        for r, g, b, _ in rgba_colors
                    ]
                    
                    # Create a custom legend image
                    legend_width = 200
                    legend_height = 40
                    
                    # Create the base legend image
                    legend_img = Image.new('RGBA', (legend_width, legend_height), (255, 255, 255, 255))
                    draw = ImageDraw.Draw(legend_img)
                    
                    # Draw color blocks
                    block_width = legend_width / len(colors)
                    for i, color in enumerate(colors):
                        x0 = i * block_width
                        x1 = (i + 1) * block_width
                        y0 = 0
                        y1 = legend_height - 20  # Leave space for text
                        draw.rectangle([(x0, y0), (x1, y1)], fill=tuple(color))
                    
                    # Add min and max values text
                    try:
                        small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
                    except IOError:
                        small_font = ImageFont.load_default()
                    
                    # Format values to be more readable (avoid long decimals)
                    def format_value(val):
                        if isinstance(val, (int, float)):
                            if val == int(val):
                                return str(int(val))
                            return f"{val:.2f}"
                        return str(val)
                    
                    # Use the actual min/max values from statistics
                    min_text = format_value(min_val)
                    max_text = format_value(max_val)
                    
                    # Draw min/max labels
                    draw.text((0, legend_height - 15), min_text, fill=(0, 0, 0, 255), font=small_font)
                    
                    # Position max value on the right, accounting for text width
                    max_text_width = draw.textlength(max_text, small_font)
                    draw.text((legend_width - max_text_width, legend_height - 15), 
                             max_text, fill=(0, 0, 0, 255), font=small_font)
                    
                    print("✅ Created custom interval-based legend image with min/max values")
                else:
                    print(f"⚠️ Could not get statistics, status code: {stats_response.status_code}")
                    # Fall back to default legend creation
        except Exception as e:
            print(f"⚠️ Error creating custom legend: {e}")
            # Will fall back to default legend creation
        
        # If we couldn't create a custom legend, use the default TiTiler approach
        if legend_img is None and input_data.get('colourmap'):
            try:
                # Add known valid colormaps supported by TiTiler
                valid_colormaps = [
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                    'terrain', 'rainbow', 'jet', 'turbo', 'hot', 'cool'
                ]
                
                colormap = input_data['colourmap']
                if colormap not in valid_colormaps:
                    colormap = 'viridis'  # Default to viridis if the requested colormap is not supported
                
                # Use the correct endpoint for getting the legend
                legend_url = f"{TITILER_BASE}/colormaps/{colormap}/legend.png?width=200&height=20"
                print(f"Fetching legend from: {legend_url}")
                
                legend_response = requests.get(legend_url)
                
                if legend_response.status_code == 200:
                    legend_img = Image.open(BytesIO(legend_response.content)).convert("RGBA")
                    print("✅ Generated legend image from TiTiler")
                    
                    # Since we're using TiTiler's legend, we should add min/max labels
                    # But we need to ensure we have min/max values available
                    if cog_stats is not None:
                        # Add min and max values text from the statistics
                        legend_height = legend_img.height + 20  # Add space for text
                        extended_legend = Image.new('RGBA', (legend_img.width, legend_height), (255, 255, 255, 255))
                        extended_legend.paste(legend_img, (0, 0))
                        
                        draw = ImageDraw.Draw(extended_legend)
                        
                        try:
                            small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
                        except IOError:
                            small_font = ImageFont.load_default()
                        
                        band_key = list(cog_stats.keys())[0]
                        band_stats = cog_stats[band_key]
                        min_val = band_stats.get('min', 0)
                        max_val = band_stats.get('max', 255)
                        
                        # Format values
                        def format_value(val):
                            if isinstance(val, (int, float)):
                                if val == int(val):
                                    return str(int(val))
                                return f"{val:.2f}"
                            return str(val)
                        
                        min_text = format_value(min_val)
                        max_text = format_value(max_val)
                        
                        # Draw min/max labels
                        draw.text((0, legend_height - 15), min_text, fill=(0, 0, 0, 255), font=small_font)
                        
                        # Position max value on the right
                        max_text_width = draw.textlength(max_text, small_font)
                        draw.text((legend_img.width - max_text_width, legend_height - 15), 
                                 max_text, fill=(0, 0, 0, 255), font=small_font)
                        
                        legend_img = extended_legend
                else:
                    print(f"⚠️ Could not get legend, status code: {legend_response.status_code}")
                    # Create a fallback legend with min/max
                    legend_img = Image.new('RGBA', (200, 40), (255, 255, 255, 255))
                    draw = ImageDraw.Draw(legend_img)
                    
                    # Draw a gradient based on the colormap name
                    if colormap == 'viridis':
                        # Blue to yellow-green gradient for viridis
                        for x in range(200):
                            r = min(255, int((x/200) * 255))
                            g = min(255, int((x/200) * 200))
                            b = max(0, 255 - int((x/200) * 200))
                            draw.line([(x, 0), (x, 20)], fill=(r, g, b, 255))
                    else:
                        # Generic gradient for other colormaps
                        for x in range(200):
                            import math  # Make sure math is imported
                            draw.line([(x, 0), (x, 20)], fill=(int((x/200) * 255), 
                                                              int((1-(x/200)) * 255), 
                                                              int(127 * abs(math.sin(x/32))), 255))
                            
                    # Add min/max labels if stats are available
                    if cog_stats is not None:
                        try:
                            small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
                        except IOError:
                            small_font = ImageFont.load_default()
                        
                        band_key = list(cog_stats.keys())[0]
                        band_stats = cog_stats[band_key]
                        min_val = band_stats.get('min', 0)
                        max_val = band_stats.get('max', 255)
                        
                        def format_value(val):
                            if isinstance(val, (int, float)):
                                if val == int(val):
                                    return str(int(val))
                                return f"{val:.2f}"
                            return str(val)
                        
                        min_text = format_value(min_val)
                        max_text = format_value(max_val)
                        
                        # Draw min/max labels
                        draw.text((0, 25), min_text, fill=(0, 0, 0, 255), font=small_font)
                        
                        # Position max value on the right
                        max_text_width = draw.textlength(max_text, small_font)
                        draw.text((200 - max_text_width, 25), 
                                 max_text, fill=(0, 0, 0, 255), font=small_font)
                    
                    print("✅ Created fallback legend image with min/max values")
            except Exception as legend_err:
                print(f"⚠️ Error getting legend: {legend_err}")
                # Create a simple grayscale legend as fallback
                legend_img = Image.new('RGBA', (200, 40), (255, 255, 255, 255))
                draw = ImageDraw.Draw(legend_img)
                for x in range(200):
                    color = int(x * 255 / 200)
                    draw.line([(x, 0), (x, 20)], fill=(color, color, color, 255))
                
                # Add min/max labels if stats are available
                if cog_stats is not None:
                    try:
                        small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
                    except IOError:
                        small_font = ImageFont.load_default()
                    
                    band_key = list(cog_stats.keys())[0]
                    band_stats = cog_stats[band_key]
                    min_val = band_stats.get('min', 0)
                    max_val = band_stats.get('max', 255)
                    
                    def format_value(val):
                        if isinstance(val, (int, float)):
                            if val == int(val):
                                return str(int(val))
                            return f"{val:.2f}"
                        return str(val)
                    
                    min_text = format_value(min_val)
                    max_text = format_value(max_val)
                    
                    # Draw min/max labels
                    draw.text((0, 25), min_text, fill=(0, 0, 0, 255), font=small_font)
                    
                    # Position max value on the right
                    max_text_width = draw.textlength(max_text, small_font)
                    draw.text((200 - max_text_width, 25), 
                             max_text, fill=(0, 0, 0, 255), font=small_font)
                
                print("✅ Created simple grayscale legend as fallback with min/max values")
        
        # Process each COG
        for cog in selected_cogs:
            try:
                # Generate and download the base image from TiTiler
                titiler_url = generate_titiler_url(cog, input_data)
                img = download_image(titiler_url)
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Create a larger canvas to add text and legend
                # Add 100px at the top for datetime and file info
                canvas = Image.new('RGBA', (img_width, img_height + 100), (255, 255, 255, 255))
                canvas.paste(img, (0, 100))  # Paste original image below the header
                
                # Create a drawing context
                draw = ImageDraw.Draw(canvas)
                
                # Load a font
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 14)
                    small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
                except IOError:
                    # Fallback to default font
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                
                # Format datetime for display
                datetime_str = convert_epoch_to_datetime(cog["aquisition_datetime"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                
                # Format band name correctly whether it's a string or a list
                band_name = input_data['bandName']
                if isinstance(band_name, list):
                    band_name = ", ".join(band_name)
                
                # Add file information
                file_info = f"Satellite: {input_data['SatelliteId']} | Product: {input_data['productType']}"
                
                # Draw text for datetime and file info (on the left side)
                draw.text((10, 10), datetime_str, fill=(0, 0, 0, 255), font=font)
                draw.text((10, 40), file_info, fill=(0, 0, 0, 255), font=small_font)
                draw.text((10, 65), f"File: {cog['filename']}", fill=(0, 0, 0, 255), font=small_font)
                
                # Add the ISRO logo in the top-right corner of the header
                if logo_img:
                    logo_position = (img_width - logo_img.width - 10, 10)  # Top-right position
                    canvas.paste(logo_img, logo_position, logo_img)
                
                # Add legend in the top-middle of the header
                if legend_img:
                    # Center horizontally
                    legend_position_x = (img_width - legend_img.width) // 2
                    # Place in the middle of the header area
                    legend_position_y = 50
                    canvas.paste(legend_img, (legend_position_x, legend_position_y), legend_img)
                    
                    # Only add min/max labels if not already part of the custom legend
                    if legend_img.height < 30:  # The custom legend already includes the labels
                        # Add min/max labels to the legend
                        draw.text(
                            (legend_position_x - 30, legend_position_y + 5), 
                            "Min", 
                            fill=(0, 0, 0, 255), 
                            font=small_font
                        )
                        draw.text(
                            (legend_position_x + legend_img.width + 5, legend_position_y + 5), 
                            "Max", 
                            fill=(0, 0, 0, 255), 
                            font=small_font
                        )
                
                enhanced_images.append(canvas)
                print(f"✅ Added enhanced image from {cog['filename']} to GIF")
            except Exception as img_err:
                print(f"❌ Error processing image for {cog['filename']}: {img_err}")
        
        # Build the GIF with the enhanced images
        if not enhanced_images:
            print("⚠️ No enhanced images were created successfully")
            # Create a default "error" image
            error_img = Image.new('RGBA', (800, 400), (255, 255, 255, 255))
            draw = ImageDraw.Draw(error_img)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
                small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            draw.text((50, 50), "Error Processing Images", fill=(255, 0, 0, 255), font=font)
            draw.text((50, 100), "COGs were found but could not be processed into images.", 
                     fill=(0, 0, 0, 255), font=small_font)
            
            enhanced_images = [error_img]
        
        gif_buffer = build_gif(enhanced_images)
        return send_file(gif_buffer, mimetype="image/gif", download_name="output.gif")
    except Exception as e:
        print(f"❌ Error in generate_gif_endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/pointprobe", methods=["POST"])
@swag_from({
    "tags": ["Probe"],
    "summary": "Sample pixel values at a lon/lat across a time‐series",
    "description": (
        "Fetches COG metadata using the same filter rules as `/generate-gif`, "
        "then for each COG calls TiTiler’s `/point/{lon},{lat}` endpoint to "
        "sample pixel values for the requested band(s)."
    ),
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "required": [
                    "SatelliteId","startDateTime","endDateTime","interval",
                    "processingLevel","productType","bandName","coordinate"
                ],
                "properties": {
                    "SatelliteId":     {"type": "string", "example": "3R"},
                    "startDateTime":   {"type": "string","format":"date-time",
                                        "example":"2025-03-22T00:00:00"},
                    "endDateTime":     {"type": "string","format":"date-time",
                                        "example":"2025-03-22T06:00:00"},
                    "interval":        {"type": "integer", "example": 1},
                    "processingLevel": {"type": "string",  "example": "L1C"},
                    "productType":     {"type": "string",  "example": "ASIA_MER"},
                    "bandName":        {
                        "oneOf": [
                            {"type":"string","example":"TIR2"},
                            {"type":"array","items":{"type":"string"},"example":["TIR2","VIS0"]}
                        ]
                    },
                    "coordinate": {
                        "type": "object",
                        "required": ["lat","lon"],
                        "properties": {
                            "lat": {"type": "number","example": 22.5726},
                            "lon": {"type": "number","example": 88.3639}
                        },
                        "description": "Point coordinate for sampling"
                    }
                }
            }
        }
    ],
    "responses": {
        "200": {
            "description": "List of per-file samples",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file":   {"type": "string"},
                        "date":   {"type": "string", "format":"date-time"},
                        "values": {
                            "type": "object",
                            "additionalProperties": {"type":"number"}
                        }
                    }
                }
            }
        },
        "400": {
            "description": "Bad request",
            "schema": {"type":"object","properties":{"error":{"type":"string"}}}
        },
        "500": {
            "description": "Server error",
            "schema": {"type":"object","properties":{"error":{"type":"string"}}}
        }
    }
})
def pointprobe():
    data = request.get_json(force=True)
    lat = data.get("coordinate", {}).get("lat")
    lon = data.get("coordinate", {}).get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Missing coordinate.lat or coordinate.lon"}), 400

    try:
        cogs = get_filtered_cogs(data)
        
        # Normalize bandName to a list
        bands = (
            data["bandName"] if isinstance(data["bandName"], list)
            else [data["bandName"]]
        )
        
        def process_cog(cog):
            try:
                # 1️⃣ build and encode a file:// URL
                file_url = f"file://{cog['filepath']}/{cog['filename']}"
                encoded = quote_plus(file_url)

                # 2️⃣ use the /point/{lon},{lat} path (no .json)
                url = f"{TITILER_BASE}/point/{lon},{lat}?url={encoded}"

                # 3️⃣ tell TiTiler exactly which band(s) to sample
                for b in bands:
                    # find the matching bandId from metadata
                    for band in cog["bands"]:
                        if band["description"] in (b, f"IMG_{b}"):
                            url += f"&bidx={band['bandId']}"

                # 4️⃣ fetch and parse
                resp = requests.get(url)
                resp.raise_for_status()
                js = resp.json()

                # 5️⃣ TiTiler returns "band_names" + "values"
                names = js.get("band_names", [])
                values = js.get("values", [])

                vals = {name: val for name, val in zip(names, values)}

                return {
                    "file": cog["filename"],
                    "date": convert_epoch_to_datetime(cog["aquisition_datetime"]).isoformat(),
                    "values": vals
                }
            except Exception as e:
                print(f"Error processing COG {cog['filename']}: {str(e)}")
                return None

        # Use ThreadPoolExecutor to process COGs in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(cogs))) as executor:
            results = list(executor.map(process_cog, cogs))
        
        # Filter out any None results (failed processing)
        out = [result for result in results if result is not None]
        
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import torch
from torch import nn
import torch.nn.functional as F

# First, add the model class definition
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        
        # Add center cropping to handle size mismatch
        # Make sure upconv3 and conv2 have the same spatial dimensions
        if upconv3.size() != conv2.size():
            # Calculate center crop dimensions
            target_size = upconv3.size()[-2:]  # Get height, width of upconv3
            current_size = conv2.size()[-2:]   # Get height, width of conv2
            
            # Calculate how much to crop from each side
            diff_h = (current_size[0] - target_size[0]) // 2
            diff_w = (current_size[1] - target_size[1]) // 2
            
            # Crop conv2 to match upconv3
            if diff_h > 0 or diff_w > 0:
                conv2 = conv2[:, :, 
                              diff_h:diff_h + target_size[0], 
                              diff_w:diff_w + target_size[1]]
            
            # If upconv3 is bigger, we need to crop it instead
            elif diff_h < 0 or diff_w < 0:
                diff_h, diff_w = abs(diff_h), abs(diff_w)
                upconv3 = upconv3[:, :, 
                                 diff_h:diff_h + current_size[0], 
                                 diff_w:diff_w + current_size[1]]
        
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        
        # Same cropping for upconv2 and conv1
        if upconv2.size() != conv1.size():
            target_size = upconv2.size()[-2:]
            current_size = conv1.size()[-2:]
            
            diff_h = (current_size[0] - target_size[0]) // 2
            diff_w = (current_size[1] - target_size[1]) // 2
            
            if diff_h > 0 or diff_w > 0:
                conv1 = conv1[:, :, 
                              diff_h:diff_h + target_size[0], 
                              diff_w:diff_w + target_size[1]]
            elif diff_h < 0 or diff_w < 0:
                diff_h, diff_w = abs(diff_h), abs(diff_w)
                upconv2 = upconv2[:, :, 
                                 diff_h:diff_h + current_size[0], 
                                 diff_w:diff_w + current_size[1]]
        
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    # Rest of the methods remain the same
    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
        )
        return expand

# Initialize the model at application startup
cloud_model = None

@app.before_first_request
def load_cloud_model():
    global cloud_model
    # Initialize model
    cloud_model = UNET(in_channels=4, out_channels=2)
    
    # Load the model weights - update the path to your model file
    model_path = "/home/sbn/souradip/geo-servers/unet_cloud_segmentation.pth"
    try:
        cloud_model.load_state_dict(torch.load(model_path))
        cloud_model.eval()
        print(f"✅ Cloud segmentation model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading cloud model: {str(e)}")

# Helper function to preprocess imagery for the model
def preprocess_for_cloud_model(raster_data):
    """
    Preprocess satellite imagery for the cloud segmentation model.
    
    Args:
        raster_data: List of numpy arrays containing band data
    
    Returns:
        torch.Tensor: Tensor ready for model input
    """
    # Extract and normalize the required bands
    bands = []
    
    # Process up to 4 bands if available
    for i in range(min(4, len(raster_data))):
        band = raster_data[i].astype(np.float32)
        
        # Check if band is empty before normalizing
        if band.size > 0:
            # Normalize band values to 0-1 range
            band_min = band.min()
            band_max = band.max()
            
            if band_max > band_min:
                band = (band - band_min) / (band_max - band_min)
            else:
                # If band has constant values, set to zeros
                band = np.zeros_like(band)
        else:
            # Handle empty band by creating a zero array with expected dimensions
            print(f"Warning: Empty band encountered at index {i}")
            if bands:
                band = np.zeros_like(bands[0])
            else:
                raise ValueError(f"Band at index {i} is empty and no previous bands exist to determine shape")
        
        bands.append(band)
    
    # Pad with zeros if we have fewer than 4 bands
    while len(bands) < 4:
        if bands:
            bands.append(np.zeros_like(bands[0]))
        else:
            # If we have no bands at all, we can't proceed
            raise ValueError("No valid bands found in the input data")
    
    # Stack bands into tensor with shape [4, H, W]
    tensor = torch.tensor(np.stack(bands), dtype=torch.float32)
    
    return tensor.unsqueeze(0)  # Add batch dimension [1, 4, H, W]

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Upsampling part with size matching
        upconv3 = self.upconv3(conv3)
        
        # Ensure upconv3 has the same spatial dimensions as conv2
        if upconv3.size()[2:] != conv2.size()[2:]:
            upconv3 = F.interpolate(
                upconv3, 
                size=conv2.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        
        # Ensure upconv2 has the same spatial dimensions as conv1
        if upconv2.size()[2:] != conv1.size()[2:]:
            upconv2 = F.interpolate(
                upconv2, 
                size=conv1.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
        return expand
    
# Global variable for cloud model
cloud_model = None

@app.before_first_request
def load_cloud_model():
    global cloud_model
    
    # Initialize model
    cloud_model = UNET(in_channels=4, out_channels=2)
    
    # Load the model weights
    model_path = "/home/sbn/souradip/geo-servers/unet_cloud_segmentation.pth"
    try:
        cloud_model.load_state_dict(torch.load(model_path, weights_only=True))
        cloud_model.eval()
        print(f"✅ Cloud segmentation model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading cloud model: {str(e)}")
@app.route("/cloud-mask", methods=["POST"])
@swag_from({
    'tags': ['Cloud Detection'],
    'summary': 'Generate cloud segmentation mask overlaid on original satellite imagery',
    'description': 'Creates a cloud segmentation mask using a U-Net deep learning model and overlays it on the original image',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'SatelliteId': {'type': 'string', 'example': '3R'},
                    'startDateTime': {'type': 'string', 'format': 'date-time', 
                                     'example': '2025-03-22T00:00:00'},
                    'endDateTime': {'type': 'string', 'format': 'date-time',
                                   'example': '2025-03-22T00:05:45'},
                    'processingLevel': {'type': 'string', 'example': 'L1C'},
                    'productType': {'type': 'string', 'example': 'ASIA_MER'},
                    'bbox': {
                        'type': 'object',
                        'properties': {
                            'minx': {'type': 'number', 'example': -1000000},
                            'miny': {'type': 'number', 'example': -500000},
                            'maxx': {'type': 'number', 'example': 1000000},
                            'maxy': {'type': 'number', 'example': 1000000}
                        }
                    },
                    'format': {'type': 'string', 'enum': ['png', 'tiff', 'json'], 'default': 'png'},
                    'confidenceThreshold': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.5,
                                          'description': 'Threshold for cloud confidence (0-1)'},
                    'maskColor': {
                        'type': 'array',
                        'items': {'type': 'integer', 'minimum': 0, 'maximum': 255},
                        'minItems': 3,
                        'maxItems': 3,
                        'example': [255, 0, 0],
                        'description': 'RGB color for cloud mask overlay [R, G, B]'
                    },
                    'opacity': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1,
                        'default': 0.5,
                        'description': 'Opacity of cloud mask overlay (0-1)'
                    }
                },
                'required': ['SatelliteId', 'startDateTime', 'endDateTime', 
                            'processingLevel', 'productType']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Cloud segmentation mask overlaid on original imagery'
        },
        '400': {
            'description': 'Bad request parameters'
        },
        '500': {
            'description': 'Server error during processing'
        }
    }
})
def generate_cloud_mask():
    """Generate cloud mask using the loaded U-Net model and overlay it on the original image."""
    data = request.get_json(force=True)
    output_format = data.get('format', 'png').lower()
    confidence_threshold = float(data.get('confidenceThreshold', 0.5))
    mask_color = data.get('maskColor', [255, 0, 0])  # Default: red overlay
    opacity = float(data.get('opacity', 0.5))  # Default: 50% opacity
    
    # Add default bandName if not provided
    if 'bandName' not in data:
        data['bandName'] = ['VIS', 'SWIR', 'TIR1', 'TIR2']  # Default bands for cloud detection
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"cloud_mask_overlay.{output_format}")
    
    try:
        # Get selected COG using the existing filtering function
        selected_cogs = get_filtered_cogs(data)
        
        if not selected_cogs:
            return jsonify({"error": "No COGs found matching criteria"}), 404
            
        # Use the most recent COG
        cog = selected_cogs[-1]
        cog_path = os.path.join(cog['filepath'], cog['filename'])
        
        # Process the imagery
        with rasterio.open(cog_path) as src:
            transform = src.transform
            crs = src.crs
            
            # Read bands for the model
            bands_data = []
            band_ids = []
            
            # Look for relevant bands
            target_bands = ['VIS', 'RED', 'GREEN', 'BLUE', 'NIR', 'SWIR', 'TIR1', 'TIR2']
            found_bands = set()
            
            for band in cog["bands"]:
                band_desc = band["description"]
                for target in target_bands:
                    if target in band_desc and target not in found_bands:
                        band_ids.append(band['bandId'])
                        found_bands.add(target)
                        break
            
            # If we didn't find 4 bands, just take the first 4 available
            if len(band_ids) < 4:
                band_ids = [b['bandId'] for b in cog["bands"][:4]]
            
            # Only use the first 4 bands for the model
            model_band_ids = band_ids[:4]
            
            # Handle bounding box if provided
            if 'bbox' in data:
                bbox = data['bbox']
                geo_bounds = (bbox['minx'], bbox['miny'], bbox['maxx'], bbox['maxy'])
                print(f"Requested geo bounds: {geo_bounds}")
                
                # Transform coordinates if needed
                if (crs.is_projected and 
                    bbox['minx'] >= -180 and bbox['maxx'] <= 180 and 
                    bbox['miny'] >= -90 and bbox['maxy'] <= 90):
                    
                    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
                    minx, miny = transformer.transform(geo_bounds[0], geo_bounds[1])
                    maxx, maxy = transformer.transform(geo_bounds[2], geo_bounds[3])
                    proj_bounds = (minx, miny, maxx, maxy)
                    
                    window = rasterio.windows.from_bounds(*proj_bounds, transform=transform)
                else:
                    window = rasterio.windows.from_bounds(*geo_bounds, transform=transform)
                
                print(f"Window: {window}")
                
                # Validate window size
                if window.width < 1 or window.height < 1:
                    return jsonify({
                        "error": "Requested bounding box is too small or outside image extent"
                    }), 400
                
                # Read data for model
                for idx in model_band_ids:
                    band_data = src.read(idx, window=window)
                    print(f"Band {idx} shape: {band_data.shape}, size: {band_data.size}")
                    if band_data.size == 0:
                        return jsonify({"error": f"Empty data read for band {idx}"}), 400
                    bands_data.append(band_data)
                
                # Read RGB bands for visualization (use first 3 bands if RGB not available)
                rgb_ids = []
                for color in ['RED', 'GREEN', 'BLUE']:
                    for band in cog["bands"]:
                        if color in band["description"]:
                            rgb_ids.append(band['bandId'])
                            break
                
                # If we don't have RGB bands explicitly, use the first 3 bands
                if len(rgb_ids) < 3:
                    rgb_ids = band_ids[:3] if len(band_ids) >= 3 else [1, 1, 1]
                
                # Read RGB data
                rgb_data = np.zeros((3, bands_data[0].shape[0], bands_data[0].shape[1]), dtype=np.float32)
                for i, idx in enumerate(rgb_ids[:3]):
                    rgb_data[i] = src.read(idx, window=window)
            else:
                # Read data for full image
                for idx in model_band_ids:
                    band_data = src.read(idx)
                    print(f"Band {idx} shape: {band_data.shape}, size: {band_data.size}")
                    bands_data.append(band_data)
                
                # Read RGB bands for visualization
                rgb_ids = []
                for color in ['RED', 'GREEN', 'BLUE']:
                    for band in cog["bands"]:
                        if color in band["description"]:
                            rgb_ids.append(band['bandId'])
                            break
                
                # If we don't have RGB bands explicitly, use the first 3 bands
                if len(rgb_ids) < 3:
                    rgb_ids = band_ids[:3] if len(band_ids) >= 3 else [1, 1, 1]
                
                # Read RGB data
                rgb_data = np.zeros((3, bands_data[0].shape[0], bands_data[0].shape[1]), dtype=np.float32)
                for i, idx in enumerate(rgb_ids[:3]):
                    rgb_data[i] = src.read(idx)
            
            # Create RGB visualization image
            rgb_image = np.zeros((rgb_data.shape[1], rgb_data.shape[2], 3), dtype=np.uint8)
            
            # Normalize and convert to 8-bit for each channel
            for i in range(3):
                channel = rgb_data[i]
                if channel.min() != channel.max():
                    channel_norm = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
                else:
                    channel_norm = np.zeros_like(channel, dtype=np.uint8)
                rgb_image[:, :, i] = channel_norm
            
            # Preprocess the raster data for model input
            input_tensor = preprocess_for_cloud_model(bands_data)
            
            # Run inference with the model
            with torch.no_grad():
                output = cloud_model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                cloud_prob = probabilities[0, 1].cpu().numpy()
                
                # Debug output to diagnose sizing issues
                print(f"Cloud probability shape: {cloud_prob.shape}")
                print(f"RGB image shape: {rgb_image.shape[:2]}")
                
                # Explicitly resize cloud probability to match RGB image dimensions
                from skimage.transform import resize
                cloud_prob_resized = resize(
                    cloud_prob, 
                    rgb_image.shape[:2],
                    mode='constant',
                    anti_aliasing=True,
                    preserve_range=True
                )
                
                # Apply threshold to get binary mask
                cloud_mask = (cloud_prob_resized >= confidence_threshold).astype(np.uint8)
                
                print(f"Resized cloud mask shape: {cloud_mask.shape}")
            
            # Create cloud mask overlay
            # Create a colored mask based on the cloud mask
            colored_mask = np.zeros_like(rgb_image)
            colored_mask[cloud_mask == 1] = mask_color
            
            # Create PIL images for blending
            original_img = Image.fromarray(rgb_image)
            mask_img = Image.fromarray(colored_mask)
            
            # Blend images with specified opacity
            overlay_img = Image.blend(original_img, mask_img, opacity)
            
            # Save the result
            if output_format == 'tiff':
                # Convert to numpy and save with georeference
                overlay_array = np.array(overlay_img)
                # Transpose to rasterio format (channels, height, width)
                overlay_array = overlay_array.transpose(2, 0, 1)
                
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=overlay_array.shape[1],
                    width=overlay_array.shape[2],
                    count=3,
                    dtype=rasterio.uint8,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(overlay_array)
                
                return send_file(output_path, mimetype="image/tiff", 
                                download_name="cloud_mask_overlay.tiff")
            
            elif output_format == 'png':
                # Save as PNG
                overlay_img.save(output_path)
                
                return send_file(output_path, mimetype="image/png", 
                                download_name="cloud_mask_overlay.png")
            
            elif output_format == 'json':
                # Calculate cloud statistics
                total_pixels = cloud_mask.size
                cloud_pixels = np.sum(cloud_mask)
                cloud_percentage = (cloud_pixels / total_pixels) * 100
                
                # Calculate connected components for cloud count
                from skimage import measure
                labels, num_clouds = measure.label(cloud_mask, return_num=True)
                
                return jsonify({
                    "cloudPercentage": round(cloud_percentage, 2),
                    "cloudCount": num_clouds,
                    "timestamp": convert_epoch_to_datetime(cog["aquisition_datetime"]).isoformat(),
                    "filename": cog["filename"]
                })
                
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
