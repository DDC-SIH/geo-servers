import os
import json
import tempfile
import shutil
import logging
from urllib.parse import urlparse, parse_qs, unquote, urlunparse
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
import zipfile
from io import BytesIO
import imageio
from PIL import Image
import traceback
import concurrent.futures
from threading import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class TitilerUrlData:
    """Class for storing extracted data from TiTiler URLs"""
    valid_url: str = None
    local_file_path: str = None
    band_indices: list = None
    rescale_values: list = None
    bbox: list = None
    error: str = None

def extract_and_validate_titiler_url(direct_url):
    """
    Extracts and validates the TiTiler URL, ensuring it has the correct parameters.
    Also extracts file path and parameters for direct processing.
    
    Returns:
        tuple: (TitilerUrlData object, error_message)
    """
    try:
        result = TitilerUrlData()
        
        # Handle URL-encoded parameters
        decoded_url = unquote(direct_url)
        
        # Parse the URL
        parsed_url = urlparse(decoded_url)
        query_params = parse_qs(parsed_url.query)
        
        # Check for the source URL parameter
        if 'url' not in query_params:
            return None, "Missing 'url' parameter in TiTiler URL"
        
        # For debugging, show all parameters
        logger.debug(f"TiTiler URL parameters: {query_params}")
        
        # Extract file path and other parameters for direct processing
        result.local_file_path = unquote(query_params['url'][0])
        result.band_indices = [int(b) for b in query_params.get('bidx', [])] if 'bidx' in query_params else None
        result.rescale_values = query_params.get('rescale', [])
        
        # Extract bbox from URL path
        bbox_pattern = r'/bbox/([^.]+)'
        match = re.search(bbox_pattern, decoded_url)
        if match:
            bbox_str = match.group(1)
            try:
                result.bbox = [float(coord) for coord in bbox_str.split(',')]
            except:
                result.bbox = None
        else:
            result.bbox = None
        
        # Ensure bidx parameters are present if needed
        if 'bidx' not in query_params:
            logger.warning("No 'bidx' parameters in TiTiler URL")
        
        # Construct a valid URL if needed
        # This ensures we're getting a valid TiFF output
        if not decoded_url.endswith('.tif'):
            base_url = direct_url.split('?')[0]
            if not base_url.endswith('.tif'):
                base_url += '.tif'
            
            # Reconstruct URL with query parameters
            query_string = '&'.join([f"{k}={v}" for k, v in query_params.items() for v in query_params[k]])
            valid_url = f"{base_url}?{query_string}"
            logger.debug(f"Reconstructed URL: {valid_url}")
            result.valid_url = valid_url
            return result, None
        
        result.valid_url = decoded_url
        return result, None
        
    except Exception as e:
        return None, f"Error validating TiTiler URL: {str(e)}"

def read_full_image(src, bands_to_read, rescale_values=None):
    """Helper function to read full image data."""
    bands_data = []
    for i, band in enumerate(bands_to_read):
        data = src.read(band)
        
        # Apply rescaling if provided
        if rescale_values and i < len(rescale_values):
            try:
                min_val, max_val = map(float, rescale_values[i].split(','))
                data = np.clip(data, min_val, max_val)
                data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            except Exception as e:
                logger.warning(f"Error applying rescale for band {band}: {str(e)}")
        
        bands_data.append(data)
    
    # Use original metadata with updated count
    out_meta = src.meta.copy()
    out_meta.update({
        'count': len(bands_data),
    })
    
    return bands_data, out_meta

def process_local_file(file_path, output_path, bbox=None, band_indices=None, rescale_values=None):
    """Process local file directly when TiTiler is unavailable."""
    logger.info(f"Processing local file directly: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Local file not found: {file_path}")
            return False
            
        with rasterio.open(file_path) as src:
            # Determine which bands to use
            if band_indices and all(1 <= b <= src.count for b in band_indices):
                bands_to_read = band_indices
            else:
                # Default to first 3 bands for RGB or first band for grayscale
                bands_to_read = list(range(1, min(4, src.count + 1)))
                
            logger.debug(f"Reading bands: {bands_to_read}")
            
            # Process AOI if provided
            if bbox and len(bbox) == 4:
                try:
                    # Get window from bounds
                    window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
                    
                    # Read data for each band
                    bands_data = []
                    for i, band in enumerate(bands_to_read):
                        data = src.read(band, window=window)
                        
                        # Apply rescaling if provided
                        if rescale_values and i < len(rescale_values):
                            try:
                                min_val, max_val = map(float, rescale_values[i].split(','))
                                data = np.clip(data, min_val, max_val)
                                data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                            except Exception as e:
                                logger.warning(f"Error applying rescale for band {band}: {str(e)}")
                        
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
                    logger.warning(f"Error processing AOI: {str(e)}. Using full extent.")
                    # Fallback to reading the full image
                    bands_data, out_meta = read_full_image(src, bands_to_read, rescale_values)
            else:
                # No AOI, read full image
                bands_data, out_meta = read_full_image(src, bands_to_read, rescale_values)
            
            # Write the processed data to output file
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                for i, data in enumerate(bands_data, 1):
                    dst.write(data, i)
                    
            logger.info(f"Successfully processed local file to {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing local file: {str(e)}", exc_info=True)
        return False

def download_from_titiler(titiler_url_data, output_path):
    """Download a COG from TiTiler."""
    try:
        if not titiler_url_data or not titiler_url_data.valid_url:
            logger.error("Invalid TiTiler URL data")
            return False
            
        # Handle URL-encoded parameters
        decoded_url = titiler_url_data.valid_url
        logger.debug(f"Downloading from URL: {decoded_url}")
        
        try:
            # Try to download from TiTiler
            response = requests.get(decoded_url, stream=True, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Failed to download from TiTiler: {response.status_code} - {response.text}")
                # Fall back to direct file processing
                if titiler_url_data.local_file_path:
                    logger.info(f"Falling back to direct file processing")
                    return process_local_file(
                        titiler_url_data.local_file_path, 
                        output_path, 
                        bbox=titiler_url_data.bbox,
                        band_indices=titiler_url_data.band_indices,
                        rescale_values=titiler_url_data.rescale_values
                    )
                return False
            
            # Write response to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.debug(f"Successfully downloaded from TiTiler to {output_path}")
                return True
            else:
                logger.warning(f"Download seemed successful but file is empty: {output_path}")
                # Fall back to direct file processing
                if titiler_url_data.local_file_path:
                    logger.info(f"Falling back to direct file processing")
                    return process_local_file(
                        titiler_url_data.local_file_path, 
                        output_path, 
                        bbox=titiler_url_data.bbox,
                        band_indices=titiler_url_data.band_indices,
                        rescale_values=titiler_url_data.rescale_values
                    )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"TiTiler connection failed: {str(e)}")
            
            # Fall back to direct file processing
            if titiler_url_data.local_file_path:
                logger.info(f"Falling back to direct file processing")
                return process_local_file(
                    titiler_url_data.local_file_path, 
                    output_path, 
                    bbox=titiler_url_data.bbox,
                    band_indices=titiler_url_data.band_indices,
                    rescale_values=titiler_url_data.rescale_values
                )
            else:
                logger.error("No local file path available for fallback processing")
                return False
        
    except Exception as e:
        logger.error(f"Error in download_from_titiler: {str(e)}", exc_info=True)
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
        
        # Handle floating point values appropriately
        if src.count >= 3:
            # Normalize RGB data if it's floating point
            if np.issubdtype(rgb.dtype, np.floating):
                # Check if data has negative values
                if np.min(rgb) < 0:
                    # Apply offset and normalization
                    min_val = np.min(rgb)
                    rgb = rgb - min_val  # Shift to positive range
                    max_val = np.max(rgb)
                    if max_val > 0:
                        rgb = rgb / max_val  # Normalize to 0-1
                else:
                    # Just normalize positive floating point values
                    max_val = np.max(rgb)
                    if max_val > 0:
                        rgb = rgb / max_val
            elif rgb.dtype != np.uint8:
                # Convert other integer types to uint8
                rgb = (rgb / np.iinfo(rgb.dtype).max * 255).astype(np.uint8)
            
            plt.imsave(png_path, rgb)
        else:
            # Handle single band data
            if np.issubdtype(data.dtype, np.floating):
                # For floating point data, normalize
                if np.min(data) < 0:
                    min_val = np.min(data)
                    data = data - min_val
                    max_val = np.max(data)
                    if max_val > 0:
                        data = data / max_val
                else:
                    max_val = np.max(data)
                    if max_val > 0:
                        data = data / max_val
            elif data.dtype != np.uint8:
                # Convert other integer types to 0-1 range
                data = data.astype(float) / np.iinfo(data.dtype).max
                
            plt.imsave(png_path, data, cmap='gray')

def process_animation_frame(frame_config, temp_dir, frame_index):
    """
    Process a single animation frame with the given configuration.
    
    Args:
        frame_config: Layer configuration for this frame
        temp_dir: Temporary directory for processing
        frame_index: Index of this frame
        
    Returns:
        dict: Frame information including path and metadata
    """
    try:
        # Extract frame properties
        frame_id = frame_config.get('id', f"frame_{frame_index}")
        direct_url = frame_config.get('directURL', '')
        
        if not direct_url:
            logger.warning(f"Skipping frame {frame_index}: Missing directURL")
            return None
        
        logger.debug(f"Processing frame {frame_index} from URL: {direct_url}")
        
        # Get band indices if provided
        band_indices = frame_config.get('band_indices')
        if band_indices and isinstance(band_indices, list):
            direct_url = modify_url_with_band_indices(direct_url, band_indices)
            logger.debug(f"Modified URL with band indices {band_indices}: {direct_url}")
        
        # Download and process the frame
        frame_tiff = os.path.join(temp_dir, f"frame_{frame_index}.tiff")
        frame_png = os.path.join(temp_dir, f"frame_{frame_index}.png")
        
        # Validate and extract TiTiler URL
        url_data, error = extract_and_validate_titiler_url(direct_url)
        if error:
            logger.error(f"Invalid TiTiler URL for frame {frame_index}: {error}")
            return None
        
        # Download the frame
        download_result = download_from_titiler(url_data, frame_tiff)
        if not download_result:
            logger.error(f"Failed to download frame {frame_index}")
            return None
        
        # Convert to PNG for animation
        convert_tiff_to_png(frame_tiff, frame_png)
        
        logger.debug(f"Successfully processed frame {frame_index}")
        
        return {
            'file': frame_png,
            'id': frame_id,
            'index': frame_index,
            'metadata': {
                'date': frame_config.get('date'),
                'time': frame_config.get('time')
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing frame {frame_index}: {str(e)}", exc_info=True)
        return None

def create_animation_from_layers(layers, output_path, temp_dir):
    """
    Create an animated GIF from a sequence of layers using multithreading.
    
    Args:
        layers: List of layer objects with directURL properties
        output_path: Path to save the final GIF
        temp_dir: Temporary directory for processing
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.debug(f"Creating animation from {len(layers)} layers")
        
        # Use multithreading to process frames in parallel
        max_workers = min(len(layers), os.cpu_count() or 4)
        logger.debug(f"Processing animation frames using {max_workers} threads")
        
        frame_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frame processing tasks
            future_to_frame = {
                executor.submit(process_animation_frame, layer, temp_dir, i): i
                for i, layer in enumerate(layers)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_index = future_to_frame[future]
                frame_info = future.result()
                
                if frame_info:
                    frame_results.append(frame_info)
                    logger.debug(f"Frame {frame_index} processed successfully")
                else:
                    logger.warning(f"Frame {frame_index} processing failed")
        
        if not frame_results:
            logger.error("No frames could be processed for animation")
            return False
        
        # Sort frames by their index to maintain proper sequence
        frame_results.sort(key=lambda x: x['index'])
        frame_files = [frame['file'] for frame in frame_results]
        
        # Create the animated GIF
        logger.debug(f"Creating GIF from {len(frame_files)} frames")
        
        # Use imageio to create the GIF with 1 second interval
        with imageio.get_writer(output_path, mode='I', duration=1, loop=0) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        logger.debug(f"Successfully created GIF: {output_path} ({os.path.getsize(output_path)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Error creating animation: {str(e)}", exc_info=True)
        return False

@app.route('/stack-layers', methods=['POST'])
def stack_layers():
    """
    Stack multiple layers based on z-index with transparency.
    
    JSON input format:
    {
      "directURL": "http://127.0.0.1:8000/cog/bbox/72.02,15.75,100.76,34.22.tif?url=C:/repos/data/3RIMG_{DATE}_{TIME}_L1C_ASIA_MER_V01R00.cog.tif&rescale=0,1000&rescale=0,1000&rescale=0,1000",
      "date_range": ["2025-03-22", "2025-03-24"],
      "time_range": ["09:00", "15:00"],
      "transparency": [0.3, 0.5, 0.8],
      "zIndex": [1000, 999, 998],
      "band_indices": [
        [1, 2, 4],
        [1, 3, 4],
        [1, 2, 3]
      ]
    }
    """
    try:
        # Get the config from the request with better error handling
        try:
            config = request.json
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raw_data = request.get_data(as_text=True)
            logger.debug(f"Raw request data (first 200 chars): {raw_data[:200]}")
            return jsonify({
                "error": "Invalid JSON in request", 
                "details": str(e),
                "help": "Please validate your JSON input. Common issues include missing commas, unquoted property names, or trailing commas."
            }), 400
        
        if not config or not isinstance(config, dict):
            return jsonify({"error": "Invalid input. Expected a configuration object."}), 400
        
        # Required fields
        if 'directURL' not in config:
            return jsonify({"error": "Missing 'directURL' in configuration"}), 400
        if 'date_range' not in config or not isinstance(config['date_range'], list) or len(config['date_range']) != 2:
            return jsonify({"error": "Missing or invalid 'date_range'. Expected format: ['YYYY-MM-DD', 'YYYY-MM-DD']"}), 400
        if 'time_range' not in config or not isinstance(config['time_range'], list) or len(config['time_range']) != 2:
            return jsonify({"error": "Missing or invalid 'time_range'. Expected format: ['HH:MM', 'HH:MM']"}), 400
        
        # Expand the configuration into individual layers
        try:
            layers = expand_config_to_layers(config)
            logger.debug(f"Expanded configuration to {len(layers)} layers")
        except Exception as e:
            logger.error(f"Error expanding configuration: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error expanding configuration: {str(e)}"}), 400
        
        # Now continue with the standard processing using the expanded layers list
        output_format = request.args.get('format', 'tiff').lower()
        create_zip = request.args.get('zip', 'no').lower() == 'yes'
        create_animation = request.args.get('animation', 'no').lower() == 'yes'
        
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")
        
        if create_animation:
            file_ext = 'gif'
            output_format = 'gif'
        elif output_format == 'png':
            file_ext = 'png'
        else:
            file_ext = 'tiff'
            output_format = 'tiff'
            
        output_path = os.path.join(temp_dir, f"stacked_output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "stacked_output_temp.tiff")
        
        if create_animation:
            animation_result = create_animation_from_layers(layers, output_path, temp_dir)
            if not animation_result:
                logger.error("Failed to create animation.")
                return jsonify({"error": "Failed to create animation."}), 500
            
            logger.debug(f"Successfully created animation: {output_path} ({os.path.getsize(output_path)} bytes)")
            return send_file(output_path, mimetype='image/gif',
                           as_attachment=True, download_name="animation.gif")
        
        sorted_layers = sorted(layers, key=lambda x: x.get('zIndex', 0))
        logger.debug(f"Processing {len(sorted_layers)} layers")
        
        processed_layers = []
        reference_lock = Lock()
        reference_data = {
            'transform': None,
            'crs': None,
            'width': None,
            'height': None,
            'bbox': None
        }
        
        def process_layer(layer_index, layer):
            try:
                layer_id = layer.get('id', f"layer_{layer_index}")
                direct_url = layer.get('directURL', '')
                transparency = float(layer.get('transparency', 1.0))
                
                if not direct_url:
                    logger.warning(f"Skipping layer {layer_id}: Missing directURL.")
                    return None
                
                logger.debug(f"Processing layer {layer_id} with transparency {transparency} and zIndex {layer.get('zIndex')}")
                
                # If band indices are provided in the layer, modify the URL
                band_indices = layer.get('band_indices')
                if band_indices and isinstance(band_indices, list):
                    direct_url = modify_url_with_band_indices(direct_url, band_indices)
                    logger.debug(f"Modified URL with band indices {band_indices}: {direct_url}")
                
                url_data, error = extract_and_validate_titiler_url(direct_url)
                if error:
                    logger.error(f"Invalid TiTiler URL for layer {layer_id}: {error}")
                    return None
                
                layer_file = os.path.join(temp_dir, f"layer_{layer_id}.tiff")
                download_result = download_from_titiler(url_data, layer_file)
                if not download_result:
                    logger.error(f"Failed to download/process layer {layer_id}")
                    return None
                
                if not os.path.exists(layer_file) or os.path.getsize(layer_file) < 100:
                    logger.error(f"Layer file is missing or too small: {layer_file}")
                    return None
                
                try:
                    with rasterio.open(layer_file) as src:
                        with reference_lock:
                            if reference_data['transform'] is None:
                                reference_data['transform'] = src.transform
                                reference_data['crs'] = src.crs
                                reference_data['width'] = src.width
                                reference_data['height'] = src.height
                                reference_data['bbox'] = src.bounds
                                logger.debug(f"Reference layer set: {src.width}x{src.height}, CRS: {src.crs}")
                        
                        return {
                            'file': layer_file,
                            'transparency': transparency,
                            'id': layer_id
                        }
                        
                except rasterio.errors.RasterioIOError as e:
                    logger.error(f"Failed to open raster file for layer {layer_id}: {str(e)}")
                    return None
                
            except Exception as e:
                logger.error(f"Error processing layer {layer.get('id', f'layer_{layer_index}')}: {str(e)}", exc_info=True)
                return None
        
        max_workers = min(len(sorted_layers), os.cpu_count() or 4)
        logger.debug(f"Processing layers using {max_workers} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_layer = {
                executor.submit(process_layer, i, layer): (i, layer) 
                for i, layer in enumerate(sorted_layers)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_layer):
                layer_info = future.result()
                if layer_info:
                    processed_layers.append(layer_info)
        
        if not processed_layers:
            logger.error("No layers could be processed.")
            return jsonify({"error": "No layers could be processed."}), 400
        
        logger.debug(f"Successfully processed {len(processed_layers)} layers out of {len(sorted_layers)}")
        
        if reference_data['transform'] is None and processed_layers:
            with rasterio.open(processed_layers[0]['file']) as src:
                reference_data['transform'] = src.transform
                reference_data['crs'] = src.crs
                reference_data['width'] = src.width
                reference_data['height'] = src.height
                reference_data['bbox'] = src.bounds
        
        stack_result = stack_layers_with_transparency(
            processed_layers,
            temp_tiff_path,
            reference_data['transform'],
            reference_data['crs'],
            reference_data['width'],
            reference_data['height']
        )
        
        if not stack_result:
            logger.error("Failed to stack layers.")
            return jsonify({"error": "Failed to stack layers."}), 500
        
        logger.debug(f"Successfully stacked layers to {temp_tiff_path} ({os.path.getsize(temp_tiff_path)} bytes)")
        
        if output_format == 'png':
            convert_tiff_to_png(temp_tiff_path, output_path)
            logger.debug(f"Converted to PNG: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            shutil.copy(temp_tiff_path, output_path)
            logger.debug(f"Copied to output: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        if create_zip:
            logger.debug("Creating ZIP file with all layers and result")
            zip_path = os.path.join(temp_dir, "stacked_layers_package.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(output_path, os.path.basename(output_path))
                
                for layer in processed_layers:
                    layer_filename = os.path.basename(layer['file'])
                    zipf.write(layer['file'], f"raw_layers/{layer['id']}_{layer_filename}")
            
            logger.debug(f"Created ZIP file: {zip_path} ({os.path.getsize(zip_path)} bytes)")
            return send_file(zip_path, mimetype='application/zip',
                           as_attachment=True, download_name="stacked_layers_package.zip")
        else:
            return send_file(output_path, mimetype=f'image/{file_ext}', 
                            as_attachment=True, download_name=f"stacked_layers.{file_ext}")
    
    except Exception as e:
        logger.error(f"General error in stack_layers: {str(e)}", exc_info=True)
        error_details = {
            "error": str(e),
            "type": type(e).__name__
        }
        if app.debug:
            error_details["traceback"] = traceback.format_exc()
        return jsonify(error_details), 500
    
    finally:
        if 'temp_dir' in locals():
            logger.debug(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

def expand_config_to_layers(config):
    """
    Expands the configuration with date/time ranges into individual layer configurations.
    
    Args:
        config: Dictionary with directURL, date_range, time_range and optional parameters
        
    Returns:
        list: List of expanded layer configurations
    """
    # Extract configuration parameters
    url_template = config['directURL']
    date_range = config['date_range']
    time_range = config['time_range']
    
    # Get optional parameters with defaults
    transparencies = config.get('transparency', [1.0])
    if not isinstance(transparencies, list):
        transparencies = [transparencies]
    
    z_indices = config.get('zIndex', [1000])
    if not isinstance(z_indices, list):
        z_indices = [z_indices]
    
    band_indices_list = config.get('band_indices', None)
    if band_indices_list:
        if not isinstance(band_indices_list, list):
            band_indices_list = [band_indices_list]
        elif band_indices_list and not isinstance(band_indices_list[0], list):
            band_indices_list = [band_indices_list]  # Wrap single band config in a list
    
    # Generate date sequence
    try:
        start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
        
        date_sequence = []
        current_date = start_date
        while current_date <= end_date:
            # Format date as DDMMMYYYY (e.g. 22MAR2025)
            date_formatted = current_date.strftime('%d%b%Y').upper()
            date_sequence.append((current_date.strftime('%Y-%m-%d'), date_formatted))
            current_date += timedelta(days=1)
        
        logger.debug(f"Generated date sequence: {date_sequence}")
    except Exception as e:
        logger.error(f"Error generating date sequence: {str(e)}")
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {date_range}")
    
    # Generate time sequence (hourly increments)
    try:
        start_time = datetime.strptime(time_range[0], '%H:%M')
        end_time = datetime.strptime(time_range[1], '%H:%M')
        
        time_sequence = []
        current_time = start_time
        while current_time <= end_time:
            # Format time as HHMM (e.g. 0915)
            time_formatted = current_time.strftime('%H%M')
            time_sequence.append((current_time.strftime('%H:%M'), time_formatted))
            current_time += timedelta(hours=1)
        
        logger.debug(f"Generated time sequence: {time_sequence}")
    except Exception as e:
        logger.error(f"Error generating time sequence: {str(e)}")
        raise ValueError(f"Invalid time format. Expected HH:MM, got: {time_range}")
    
    # Generate all combinations
    layers = []
    parameter_index = 0
    
    for (date_original, date_formatted), (time_original, time_formatted) in itertools.product(date_sequence, time_sequence):
        # Replace placeholders in URL with the properly formatted values
        direct_url = url_template.replace('{DATE}', date_formatted).replace('{TIME}', time_formatted)
        
        # Get parameters for this layer (cycling through available values)
        transparency = transparencies[parameter_index % len(transparencies)]
        z_index = z_indices[parameter_index % len(z_indices)]
        
        # Create the layer configuration
        layer = {
            'id': f"layer_{date_original}_{time_original.replace(':', '')}",
            'directURL': direct_url,
            'transparency': transparency,
            'zIndex': z_index,
            'date': date_original,
            'time': time_original
        }
        
        # Add band indices if provided
        if band_indices_list:
            band_index_set = band_indices_list[parameter_index % len(band_indices_list)]
            layer['band_indices'] = band_index_set
        
        layers.append(layer)
        parameter_index += 1
    
    logger.debug(f"Expanded to {len(layers)} layer configurations")
    return layers

def modify_url_with_band_indices(url, band_indices):
    """
    Modifies a TiTiler URL to include the specified band indices.
    
    Args:
        url: Original TiTiler URL
        band_indices: List of band indices to use [1, 2, 3]
        
    Returns:
        str: Modified URL with band indices
    """
    # Parse the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Remove any existing bidx parameters
    if 'bidx' in query_params:
        del query_params['bidx']
    
    # Add new bidx parameters
    bidx_params = []
    for idx in band_indices:
        bidx_params.append(('bidx', str(idx)))
    
    # Reconstruct query string
    query_items = []
    for key, values in query_params.items():
        for value in values:
            query_items.append(f"{key}={value}")
    
    # Add band indices
    for key, value in bidx_params:
        query_items.append(f"{key}={value}")
    
    # Reconstruct URL
    query_string = "&".join(query_items)
    url_parts = list(parsed_url)
    url_parts[4] = query_string
    
    return urlunparse(url_parts)

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
        logger.error(f"Error stacking layers: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)