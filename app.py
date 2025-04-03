import os
import json
import tempfile
import shutil
from urllib.parse import urlparse, parse_qs, unquote
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
from rasterio.plot import reshape_as_image
from flask import Flask, request, jsonify, send_file
from shapely.geometry import shape, box
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re

app = Flask(__name__)

@app.route('/stack-layers', methods=['POST'])
def stack_layers():
    """
    Stack multiple layers based on z-index using only directURL (TiTiler).
    Expected input is a list of layers with valid 'directURL'.
    """
    try:
        layers = request.json
        if not layers or not isinstance(layers, list):
            return jsonify({"error": "Invalid input. Expected a list of layers."}), 400

        output_format = request.args.get('format', 'tiff').lower()
        temp_dir = tempfile.mkdtemp()

        file_ext = 'png' if output_format == 'png' else 'tiff'
        output_path = os.path.join(temp_dir, f"stacked_output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "stacked_output_temp.tiff")

        # Sort by zIndex (lower on bottom, higher on top)
        sorted_layers = sorted(layers, key=lambda x: x.get('zIndex', 0))

        processed_layers = []
        reference_transform = None
        reference_crs = None
        reference_width = None
        reference_height = None

        for layer in sorted_layers:
            try:
                layer_id = layer.get('id', 'unknown')
                direct_url = layer.get('directURL')
                transparency = float(layer.get('transparency', 1.0))

                if not direct_url or '/cog/bbox/' not in direct_url:
                    raise ValueError(f"Invalid or missing directURL for layer '{layer_id}'")

                # Output file for this layer
                layer_file = os.path.join(temp_dir, f"layer_{layer_id}.tiff")

                # Download and save the raster
                download_from_titiler(direct_url, layer_file)

                with rasterio.open(layer_file) as src:
                    if reference_transform is None:
                        reference_transform = src.transform
                        reference_crs = src.crs
                        reference_width = src.width
                        reference_height = src.height

                processed_layers.append({
                    'file': layer_file,
                    'transparency': transparency
                })

                print(f"Processed {layer_id} from directURL")

            except Exception as e:
                print(f"Error processing layer {layer.get('id', 'unknown')}: {e}")
                continue

        if not processed_layers:
            return jsonify({"error": "No layers could be processed."}), 400

        stack_result = stack_layers_with_transparency(
            processed_layers,
            temp_tiff_path,
            reference_transform,
            reference_crs,
            reference_width,
            reference_height
        )

        if not stack_result:
            return jsonify({"error": "Failed to stack layers."}), 500

        if output_format == 'png':
            convert_tiff_to_png(temp_tiff_path, output_path)
        else:
            shutil.copy(temp_tiff_path, output_path)

        return send_file(output_path, as_attachment=True, download_name=f"stacked_layers.{file_ext}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

def extract_bbox_from_url(url):
    """Extract bounding box coordinates from a TiTiler URL."""
    if not url:
        return None
    
    # Try to extract bbox from the URL path
    bbox_pattern = r'/bbox/([^.]+)'
    match = re.search(bbox_pattern, url)
    
    if match:
        bbox_str = match.group(1)
        try:
            coords = [float(coord) for coord in bbox_str.split(',')]
            if len(coords) == 4:
                return coords  # [minx, miny, maxx, maxy]
        except:
            pass
    
    # Try to extract from query parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    if 'bbox' in query_params:
        try:
            bbox_str = query_params['bbox'][0]
            coords = [float(coord) for coord in bbox_str.split(',')]
            if len(coords) == 4:
                return coords  # [minx, miny, maxx, maxy]
        except:
            pass
    
    return None

def process_layer(file_url, direct_url, aoi, output_path, band_ids=None, min_max_values=None):
    """Process a layer based on its URL and direct URL."""
    try:
        # If we have a directURL with a bounding box, use that instead of processing the full file
        if direct_url and ('/cog/bbox/' in direct_url):
            # Extract bbox from directURL
            bbox_match = re.search(r'/bbox/([^.]+)', direct_url)
            if bbox_match:
                bbox_str = bbox_match.group(1)
                bbox = [float(coord) for coord in bbox_str.split(',')]
                
                # Download from TiTiler with the correct bbox
                download_from_titiler(direct_url, output_path)
                return
        
        # Fallback to local file processing if no directURL or bbox
        is_local_file = os.path.exists(file_url) or (
            urlparse(file_url).scheme == 'file' or 
            urlparse(file_url).scheme == ''
        )
        
        if is_local_file:
            local_path = file_url
            if urlparse(file_url).scheme == 'file':
                local_path = urlparse(file_url).path
                if os.name == 'nt' and local_path.startswith('/'):
                    local_path = local_path[1:]
            
            process_local_file(local_path, output_path, aoi, band_ids, min_max_values)
        else:
            temp_file = output_path + ".temp"
            download_file(file_url, temp_file)
            process_local_file(temp_file, output_path, aoi, band_ids, min_max_values)
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"Error processing layer: {str(e)}")
        raise

def process_local_file(file_path, output_path, aoi=None, band_ids=None, min_max_values=None):
    """Process a local file with the given parameters."""
    with rasterio.open(file_path) as src:
        # Determine which bands to use
        if band_ids and all(band_id.isdigit() for band_id in band_ids):
            bands_to_read = [int(band_id) for band_id in band_ids]
            # Ensure bands are within range
            bands_to_read = [b for b in bands_to_read if b <= src.count and b > 0]
        else:
            # Default to first 3 bands for RGB or first band for single band
            bands_to_read = list(range(1, min(4, src.count + 1)))
        
        # If no valid bands, use the first band
        if not bands_to_read:
            bands_to_read = [1]
        
        # Process AOI if provided
        if aoi and len(aoi) == 4:
            try:
                # Get window from bounds
                window = from_bounds(aoi[0], aoi[1], aoi[2], aoi[3], src.transform)
                
                # Read data for each band
                bands_data = []
                for i, band in enumerate(bands_to_read):
                    data = src.read(band, window=window)
                    
                    # Apply min/max rescaling if provided
                    if min_max_values and i < len(min_max_values):
                        min_val = min_max_values[i].get('min', 0)
                        max_val = min_max_values[i].get('max', 1000)
                        
                        # Rescale
                        data = np.clip(data, min_val, max_val)
                        data = (data - min_val) / (max_val - min_val) * 255
                        data = data.astype(np.uint8)
                    
                    bands_data.append(data)
                
                # Update metadata for output
                out_meta = src.meta.copy()
                out_meta.update({
                    'count': len(bands_data),
                    'width': bands_data[0].shape[1],
                    'height': bands_data[0].shape[0],
                    'transform': rasterio.windows.transform(window, src.transform)
                })
                
            except Exception as e:
                print(f"Error processing AOI: {str(e)}. Using full extent.")
                # Fallback to reading the full image
                bands_data = []
                for i, band in enumerate(bands_to_read):
                    data = src.read(band)
                    
                    # Apply min/max rescaling if provided
                    if min_max_values and i < len(min_max_values):
                        min_val = min_max_values[i].get('min', 0)
                        max_val = min_max_values[i].get('max', 1000)
                        
                        # Rescale
                        data = np.clip(data, min_val, max_val)
                        data = (data - min_val) / (max_val - min_val) * 255
                        data = data.astype(np.uint8)
                    
                    bands_data.append(data)
                
                # Use original metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    'count': len(bands_data),
                })
        else:
            # No AOI, read full image
            bands_data = []
            for i, band in enumerate(bands_to_read):
                data = src.read(band)
                
                # Apply min/max rescaling if provided
                if min_max_values and i < len(min_max_values):
                    min_val = min_max_values[i].get('min', 0)
                    max_val = min_max_values[i].get('max', 1000)
                    
                    # Rescale
                    data = np.clip(data, min_val, max_val)
                    data = (data - min_val) / (max_val - min_val) * 255
                    data = data.astype(np.uint8)
                
                bands_data.append(data)
            
            # Use original metadata
            out_meta = src.meta.copy()
            out_meta.update({
                'count': len(bands_data),
            })
        
        # Write the processed data to output file
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            for i, data in enumerate(bands_data, 1):
                dst.write(data, i)

def download_from_titiler(titiler_url, output_path):
    """Download a COG from TiTiler."""
    # Handle URL-encoded parameters
    decoded_url = unquote(titiler_url)
    
    # Download the image
    response = requests.get(decoded_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download from TiTiler: {response.status_code} - {response.text}")
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_file(url, output_path):
    """Download a file from a URL."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code} - {response.text}")
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def stack_layers_with_transparency(layers, output_path, transform, crs, width, height):
    """Stack multiple layers with transparency using vectorized operations."""
    # Create a blank canvas with alpha channel (RGBA)
    stacked_data = np.zeros((height, width, 4), dtype=np.uint8)
    
    try:
        # Process layers from bottom to top
        for layer_info in layers:
            layer_file = layer_info['file']
            transparency = layer_info['transparency']
            
            with rasterio.open(layer_file) as src:
                # Reproject if needed
                if src.transform != transform or src.crs != crs or src.width != width or src.height != height:
                    # Reproject to match reference
                    with rasterio.open(layer_file) as src:
                        # Read source data
                        if src.count == 1:
                            data = src.read(1)
                            rgb_data = np.stack([data, data, data])
                        elif src.count >= 3:
                            rgb_data = src.read([1, 2, 3])
                        else:
                            data = src.read(1)
                            rgb_data = np.stack([data, data, data])
                        
                        # Prepare destination array
                        dest_data = np.zeros((3, height, width), dtype=rgb_data.dtype)
                        
                        # Reproject
                        reproject(
                            rgb_data, dest_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=crs,
                            resampling=Resampling.bilinear
                        )
                        
                        # Convert to image format (H, W, C)
                        layer_rgb = reshape_as_image(dest_data)
                else:
                    # No reprojection needed
                    if src.count == 1:
                        data = src.read(1)
                        layer_rgb = np.stack([data, data, data], axis=2)
                    elif src.count >= 3:
                        rgb = src.read([1, 2, 3])
                        layer_rgb = reshape_as_image(rgb)
                    else:
                        data = src.read(1)
                        layer_rgb = np.stack([data, data, data], axis=2)
                
                # Convert to uint8 if not already
                if layer_rgb.dtype != np.uint8:
                    if layer_rgb.max() > 0:
                        layer_rgb = (layer_rgb / layer_rgb.max() * 255).astype(np.uint8)
                    else:
                        layer_rgb = layer_rgb.astype(np.uint8)
                
                # Create alpha channel (fully opaque)
                alpha = np.ones((height, width), dtype=np.uint8) * 255
                alpha = (alpha * transparency).astype(np.uint8)
                
                # Vectorized alpha blending
                # Convert alpha to float for calculations
                src_alpha = alpha.astype(float) / 255.0
                dst_alpha = stacked_data[..., 3].astype(float) / 255.0
                
                # Calculate new alpha
                out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
                
                # Create mask for pixels that need blending
                mask = out_alpha > 0
                
                # Apply blending only where needed
                for c in range(3):  # RGB channels
                    stacked_data[..., c] = np.where(
                        mask,
                        (layer_rgb[..., c] * src_alpha + 
                         stacked_data[..., c] * dst_alpha * (1 - src_alpha)) / out_alpha,
                        stacked_data[..., c]
                    ).astype(np.uint8)
                
                # Update alpha channel
                stacked_data[..., 3] = (out_alpha * 255).astype(np.uint8)
        
        # Write the stacked image to a GeoTIFF
        rgb_data = stacked_data[..., :3].transpose(2, 0, 1)
        
        with rasterio.open(
            output_path, 
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=rgb_data.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(rgb_data)
        
        return True
    
    except Exception as e:
        print(f"Error stacking layers: {str(e)}")
        return False

def convert_tiff_to_png(tiff_path, png_path):
    """Convert a GeoTIFF to PNG format."""
    with rasterio.open(tiff_path) as src:
        # Read data
        if src.count >= 3:
            # Read as RGB
            rgb = src.read([1, 2, 3])
            # Convert to image shape (height, width, channels)
            rgb = reshape_as_image(rgb)
        else:
            # Single band, read as grayscale
            data = src.read(1)
            # No need for reshaping for single band
        
        # Save as PNG
        if src.count >= 3:
            plt.imsave(png_path, rgb)
        else:
            plt.imsave(png_path, data, cmap='gray')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 