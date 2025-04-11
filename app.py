import os
import json
import tempfile
import shutil
import logging
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
import zipfile
from io import BytesIO
import imageio
from PIL import Image
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add middleware to preprocess raw JSON before Flask parses it
@app.before_request
def preprocess_json():
    """
    Preprocesses incoming JSON data to fix common formatting errors before Flask's JSON parser handles it.
    This helps users who might be unfamiliar with strict JSON syntax requirements.
    """
    if request.method == 'POST' and request.content_type and 'application/json' in request.content_type:
        try:
            # Get raw data as text
            raw_data = request.get_data(as_text=True)
            if not raw_data:
                return
            
            # Common JSON syntax fixes
            fixed_data = raw_data
            
            # Fix unquoted 'yes'/'no' values to proper JSON booleans
            # Match "key": yes or "key":yes patterns
            fixed_data = re.sub(r'("[^"]+"\s*:\s*)yes([,\s\}])', r'\1true\2', fixed_data)
            fixed_data = re.sub(r'("[^"]+"\s*:\s*)no([,\s\}])', r'\1false\2', fixed_data)
            
            # Fix missing commas between key-value pairs
            # This pattern looks for end of a value (not a comma) followed by a key
            fixed_data = re.sub(r'(true|false|null|"[^"]*"|\d+)(\s*\n?\s*)("[^"]+"\s*:)', r'\1,\2\3', fixed_data)
            
            # Fix single quotes used instead of double quotes for keys or string values
            fixed_data = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', fixed_data)  # Fix keys
            fixed_data = re.sub(r':\s*\'([^\']+)\'([,\s\}])', r': "\1"\2', fixed_data)  # Fix values
            
            # Fix trailing commas in objects and arrays
            fixed_data = re.sub(r',(\s*[\]}])', r'\1', fixed_data)
            
            # Log the changes if any were made
            if fixed_data != raw_data:
                logger.debug("JSON automatically fixed by preprocessor")
                if app.debug:
                    # In debug mode, show the difference
                    logger.debug(f"Original JSON (first 200 chars): {raw_data[:200]}")
                    logger.debug(f"Fixed JSON (first 200 chars): {fixed_data[:200]}")
                
                # Store the fixed data for Flask to use
                request.data = fixed_data.encode('utf-8')
                
        except Exception as e:
            logger.warning(f"Error in JSON preprocessor: {str(e)}")
            # Continue with original data, let Flask handle any remaining errors

# Add a helper function to provide detailed JSON validation
def get_json_with_detailed_error(request_obj):
    """Helper function to get JSON with detailed error messages"""
    try:
        return request_obj.json, None
    except Exception as e:
        raw_data = request_obj.get_data(as_text=True)
        error_details = str(e)
        
        # Extract line and column information
        line_num = col_num = None
        if "line" in error_details and "column" in error_details:
            line_match = re.search(r'line (\d+)', error_details)
            col_match = re.search(r'column (\d+)', error_details)
            if line_match and col_match:
                line_num = int(line_match.group(1))
                col_num = int(col_match.group(1))
        
        # Format the error location
        error_location = ""
        if line_num is not None and col_num is not None:
            lines = raw_data.split('\n')
            if 0 < line_num <= len(lines):
                problem_line = lines[line_num-1]
                pointer = ' ' * (col_num-1) + '^'
                error_location = f"\nError at line {line_num}, column {col_num}:\n{problem_line}\n{pointer}"
        
        # Try to identify common issues
        suggestions = []
        if '"' in error_details or "quote" in error_details.lower():
            suggestions.append("Check for unmatched quotes or missing quotes around string values")
        if "Expecting ',' delimiter" in error_details:
            suggestions.append("Check for missing commas between JSON objects or array items")
        if "Expecting property name" in error_details:
            suggestions.append("Check for missing or improperly formatted property names")
        if "value" in error_details.lower():
            suggestions.append("Check for invalid values (boolean values must be true/false, not yes/no)")
        
        # Create a detailed error message
        detailed_error = {
            "error": "Invalid JSON",
            "details": error_details,
            "location": error_location,
            "suggestions": suggestions
        }
        
        return None, detailed_error

# Add a route to validate curl commands for the API
@app.route('/validate-curl', methods=['POST'])
def validate_curl():
    """
    Validates a curl command and returns a corrected version if needed.
    
    Expected input:
    {
        "curl_command": "curl --location 'http://localhost:5000/stack-layers' ..."
    }
    """
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request must contain JSON with a 'curl_command' field"}), 400
            
        curl_command = data.get('curl_command', '')
        if not curl_command:
            return jsonify({"error": "Missing 'curl_command' field"}), 400
            
        # Extract the JSON payload from the curl command
        json_match = re.search(r"--data\s+'(.*?)'\s*($|\\|\n)", curl_command, re.DOTALL)
        if not json_match:
            return jsonify({"error": "Could not find JSON payload in curl command"}), 400
            
        json_payload = json_match.group(1)
        
        # Fix common JSON errors
        fixed_data = json_payload
        
        # Fix unquoted yes/no
        fixed_data = re.sub(r'("[^"]+"\s*:\s*)yes([,\s\}])', r'\1true\2', fixed_data)
        fixed_data = re.sub(r'("[^"]+"\s*:\s*)no([,\s\}])', r'\1false\2', fixed_data)
        
        # Fix missing commas
        fixed_data = re.sub(r'(true|false|null|"[^"]*"|\d+)(\s*\n?\s*)("[^"]+"\s*:)', r'\1,\2\3', fixed_data)
        
        # Fix single quotes
        fixed_data = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', fixed_data)
        fixed_data = re.sub(r':\s*\'([^\']+)\'([,\s\}])', r': "\1"\2', fixed_data)
        
        # Fix trailing commas
        fixed_data = re.sub(r',(\s*[\]}])', r'\1', fixed_data)
        
        # Check if the JSON is valid now
        try:
            json.loads(fixed_data)
            is_valid = True
        except json.JSONDecodeError as e:
            is_valid = False
            error_message = str(e)
        
        # Create the response
        result = {
            "original_curl": curl_command,
            "is_valid": is_valid
        }
        
        if fixed_data != json_payload:
            # Replace the JSON in the curl command
            fixed_curl = curl_command.replace(json_payload, fixed_data)
            result["fixed_curl"] = fixed_curl
            result["fixed_json"] = fixed_data
            
        if not is_valid:
            result["error"] = error_message
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in validate_curl: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/stack-layers', methods=['POST'])
def stack_layers():
    """
    Stack multiple layers based on z-index with transparency.
    
    Simplified JSON input:
    [
        {
            "transparency": 1.0,
            "zIndex": 1000,
            "directURL": "http://titiler_url/cog/bbox/minx,miny,maxx,maxy.tif?parameters"
        },
        ...
    ]
    """
    try:
        # Get the layers from the request with better error handling
        try:
            # First try to get the JSON directly
            layers = request.json
        except Exception as e:
            # If that fails, try to manually parse the raw data
            logger.error(f"JSON parsing error: {str(e)}")
            
            # Get the raw request data and try to diagnose the issue
            raw_data = request.get_data(as_text=True)
            logger.debug(f"Raw request data (first 200 chars): {raw_data[:200]}")
            
            # Try to find obvious JSON errors
            error_details = str(e)
            if "line" in error_details and "column" in error_details:
                # Try to extract line and column from error
                try:
                    line_match = re.search(r'line (\d+)', error_details)
                    col_match = re.search(r'column (\d+)', error_details)
                    
                    if line_match and col_match:
                        line_num = int(line_match.group(1))
                        col_num = int(col_match.group(1))
                        
                        # Get the problematic line
                        lines = raw_data.split('\n')
                        if 0 < line_num <= len(lines):
                            problem_line = lines[line_num-1]
                            pointer = ' ' * (col_num-1) + '^'
                            logger.error(f"JSON error near:\n{problem_line}\n{pointer}")
                except Exception:
                    pass
            
            return jsonify({
                "error": "Invalid JSON in request", 
                "details": str(e),
                "help": "Please validate your JSON input. Common issues include missing commas, unquoted property names, or trailing commas."
            }), 400
        
        if not layers or not isinstance(layers, list):
            return jsonify({"error": "Invalid input. Expected a list of layers."}), 400
        
        # Optional query parameters
        output_format = request.args.get('format', 'tiff').lower()
        create_zip = request.args.get('zip', 'no').lower() == 'yes'
        
        # Check if animation is requested - either in query params or any layer has animation: true
        create_animation = request.args.get('animation', 'no').lower() == 'yes'
        
        if not create_animation:
            for layer in layers:
                if layer.get('animation', '').lower() == 'yes' or layer.get('animation', False) is True:
                    create_animation = True
                    break
        
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")
        
        # Choose file extension based on format
        if create_animation:
            file_ext = 'gif'
            output_format = 'gif'
        elif output_format == 'png':
            file_ext = 'png'
        else:
            file_ext = 'tiff'
            output_format = 'tiff'  # Default to tiff if not png
            
        output_path = os.path.join(temp_dir, f"stacked_output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "stacked_output_temp.tiff")
        
        # For animation, we need to handle frame generation differently
        if create_animation:
            animation_result = create_animation_from_layers(layers, output_path, temp_dir)
            if not animation_result:
                logger.error("Failed to create animation.")
                return jsonify({"error": "Failed to create animation."}), 500
            
            # Return the animation file
            logger.debug(f"Successfully created animation: {output_path} ({os.path.getsize(output_path)} bytes)")
            return send_file(output_path, mimetype='image/gif',
                           as_attachment=True, download_name="animation.gif")
        
        # Sort layers by zIndex, highest zIndex on top
        sorted_layers = sorted(layers, key=lambda x: x.get('zIndex', 0))
        logger.debug(f"Processing {len(sorted_layers)} layers")
        
        # Process each layer and prepare for stacking
        processed_layers = []
        reference_transform = None
        reference_crs = None
        reference_width = None
        reference_height = None
        bounding_box = None
        
        for i, layer in enumerate(sorted_layers):
            try:
                # Extract layer properties
                layer_id = layer.get('id', f"layer_{i}")
                direct_url = layer.get('directURL', '')
                transparency = float(layer.get('transparency', 1.0))
                
                # Ensure directURL is provided
                if not direct_url:
                    logger.warning(f"Skipping layer {layer_id}: Missing directURL.")
                    continue
                
                logger.debug(f"Processing layer {layer_id} with transparency {transparency} and zIndex {layer.get('zIndex')}")
                
                # Validate and extract TiTiler URL
                validated_url, error = extract_and_validate_titiler_url(direct_url)
                if error:
                    logger.error(f"Invalid TiTiler URL for layer {layer_id}: {error}")
                    continue
                
                # Process the layer using directURL
                layer_file = os.path.join(temp_dir, f"layer_{layer_id}.tiff")
                
                # Download from TiTiler or process locally
                download_result = download_from_titiler(validated_url, layer_file)
                if not download_result:
                    logger.error(f"Failed to download/process layer {layer_id}")
                    continue
                
                # Verify the file exists
                if not os.path.exists(layer_file) or os.path.getsize(layer_file) < 100:
                    logger.error(f"Layer file is missing or too small: {layer_file}")
                    continue
                
                # Read the processed layer
                try:
                    with rasterio.open(layer_file) as src:
                        # If this is the first layer, use it as reference
                        if reference_transform is None:
                            reference_transform = src.transform
                            reference_crs = src.crs
                            reference_width = src.width
                            reference_height = src.height
                            bounding_box = src.bounds
                            logger.debug(f"Reference layer set: {reference_width}x{reference_height}, CRS: {reference_crs}")
                        
                        # Add to processed layers with transparency
                        processed_layers.append({
                            'file': layer_file,
                            'transparency': transparency,
                            'id': layer_id
                        })
                        
                    logger.debug(f"Successfully processed layer {layer_id}")
                except rasterio.errors.RasterioIOError as e:
                    logger.error(f"Failed to open raster file for layer {layer_id}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing layer {layer.get('id', f'layer_{i}')}: {str(e)}", exc_info=True)
                continue
        
        if not processed_layers:
            logger.error("No layers could be processed.")
            return jsonify({"error": "No layers could be processed."}), 400
        
        logger.debug(f"Stacking {len(processed_layers)} layers")
        
        # Stack the layers
        stack_result = stack_layers_with_transparency(
            processed_layers,
            temp_tiff_path,
            reference_transform,
            reference_crs,
            reference_width,
            reference_height
        )
        
        if not stack_result:
            logger.error("Failed to stack layers.")
            return jsonify({"error": "Failed to stack layers."}), 500
        
        logger.debug(f"Successfully stacked layers to {temp_tiff_path} ({os.path.getsize(temp_tiff_path)} bytes)")
        
        # Convert to PNG if requested
        if output_format == 'png':
            convert_tiff_to_png(temp_tiff_path, output_path)
            logger.debug(f"Converted to PNG: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            # Just copy the temporary TIFF to the output path
            shutil.copy(temp_tiff_path, output_path)
            logger.debug(f"Copied to output: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        # If zip is requested, create a zip with all layers and result
        if create_zip:
            logger.debug("Creating ZIP file with all layers and result")
            zip_path = os.path.join(temp_dir, "stacked_layers_package.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add the stacked result
                zipf.write(output_path, os.path.basename(output_path))
                
                # Add all the individual layers
                for layer in processed_layers:
                    layer_filename = os.path.basename(layer['file'])
                    zipf.write(layer['file'], f"raw_layers/{layer['id']}_{layer_filename}")
            
            logger.debug(f"Created ZIP file: {zip_path} ({os.path.getsize(zip_path)} bytes)")
            return send_file(zip_path, mimetype='application/zip',
                           as_attachment=True, download_name="stacked_layers_package.zip")
        else:
            # Return just the stacked file to the user
            return send_file(output_path, mimetype=f'image/{file_ext}', 
                            as_attachment=True, download_name=f"stacked_layers.{file_ext}")
    
    except Exception as e:
        logger.error(f"General error in stack_layers: {str(e)}", exc_info=True)
        # Include stack trace in debug mode
        error_details = {
            "error": str(e),
            "type": type(e).__name__
        }
        if app.debug:
            error_details["traceback"] = traceback.format_exc()
        return jsonify(error_details), 500
    
    finally:
        # Clean up temporary files
        if 'temp_dir' in locals():
            logger.debug(f"Cleaning up temporary directory: {temp_dir}")
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

def extract_and_validate_titiler_url(direct_url):
    """
    Extracts and validates the TiTiler URL, ensuring it has the correct parameters.
    Also extracts file path and parameters for direct processing.
    
    Returns:
        tuple: (valid_url, error_message)
    """
    try:
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
        local_file_path = unquote(query_params['url'][0])
        band_indices = [int(b) for b in query_params.get('bidx', [])] if 'bidx' in query_params else None
        rescale_values = query_params.get('rescale', [])
        
        # Store these in flask.g for later use if TiTiler is unavailable
        from flask import g
        g.local_file_path = local_file_path
        g.band_indices = band_indices
        g.rescale_values = rescale_values
        
        # Extract bbox from URL path
        bbox_pattern = r'/bbox/([^.]+)'
        match = re.search(bbox_pattern, decoded_url)
        if match:
            bbox_str = match.group(1)
            try:
                g.bbox = [float(coord) for coord in bbox_str.split(',')]
            except:
                g.bbox = None
        else:
            g.bbox = None
        
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
            return valid_url, None
        
        return decoded_url, None
        
    except Exception as e:
        return None, f"Error validating TiTiler URL: {str(e)}"

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

def download_from_titiler(titiler_url, output_path):
    """Download a COG from TiTiler."""
    try:
        # Handle URL-encoded parameters
        decoded_url = unquote(titiler_url)
        logger.debug(f"Downloading from URL: {decoded_url}")
        
        try:
            # Try to download from TiTiler
            response = requests.get(decoded_url, stream=True, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Failed to download from TiTiler: {response.status_code} - {response.text}")
                # Fall back to direct file processing
                from flask import g
                if hasattr(g, 'local_file_path'):
                    logger.info(f"Falling back to direct file processing")
                    return process_local_file(
                        g.local_file_path, 
                        output_path, 
                        bbox=getattr(g, 'bbox', None),
                        band_indices=getattr(g, 'band_indices', None),
                        rescale_values=getattr(g, 'rescale_values', None)
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
                from flask import g
                if hasattr(g, 'local_file_path'):
                    logger.info(f"Falling back to direct file processing")
                    return process_local_file(
                        g.local_file_path, 
                        output_path, 
                        bbox=getattr(g, 'bbox', None),
                        band_indices=getattr(g, 'band_indices', None),
                        rescale_values=getattr(g, 'rescale_values', None)
                    )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"TiTiler connection failed: {str(e)}")
            
            # Fall back to direct file processing
            from flask import g
            if hasattr(g, 'local_file_path'):
                logger.info(f"Falling back to direct file processing")
                return process_local_file(
                    g.local_file_path, 
                    output_path, 
                    bbox=getattr(g, 'bbox', None),
                    band_indices=getattr(g, 'band_indices', None),
                    rescale_values=getattr(g, 'rescale_values', None)
                )
            else:
                logger.error("No local file path available for fallback processing")
                return False
        
    except Exception as e:
        logger.error(f"Error in download_from_titiler: {str(e)}", exc_info=True)
        return False

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

def create_animation_from_layers(layers, output_path, temp_dir):
    """
    Create an animated GIF from a sequence of layers.
    
    Args:
        layers: List of layer objects with directURL properties
        output_path: Path to save the final GIF
        temp_dir: Temporary directory for processing
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.debug("Creating animation from layers")
        
        # Process each frame
        frame_files = []
        
        for i, layer in enumerate(layers):
            try:
                # Extract layer properties
                layer_id = layer.get('id', f"frame_{i}")
                direct_url = layer.get('directURL', '')
                
                if not direct_url:
                    logger.warning(f"Skipping frame {i}: Missing directURL")
                    continue
                
                # Process the frame
                logger.debug(f"Processing frame {i} from URL: {direct_url}")
                
                # Download and process the frame
                frame_tiff = os.path.join(temp_dir, f"frame_{i}.tiff")
                frame_png = os.path.join(temp_dir, f"frame_{i}.png")
                
                # Validate and extract TiTiler URL
                validated_url, error = extract_and_validate_titiler_url(direct_url)
                if error:
                    logger.error(f"Invalid TiTiler URL for frame {i}: {error}")
                    continue
                
                # Download the frame
                download_result = download_from_titiler(validated_url, frame_tiff)
                if not download_result:
                    logger.error(f"Failed to download frame {i}")
                    continue
                
                # Convert to PNG for animation
                convert_tiff_to_png(frame_tiff, frame_png)
                frame_files.append(frame_png)
                
                logger.debug(f"Successfully processed frame {i}")
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}", exc_info=True)
                continue
        
        if not frame_files:
            logger.error("No frames could be processed for animation")
            return False
            
        # Create the animated GIF
        logger.debug(f"Creating GIF from {len(frame_files)} frames")
        
        # Use imageio to create the GIF with 1 second interval
        with imageio.get_writer(output_path, mode='I', duration=1, loop=0) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        logger.debug(f"Successfully created GIF: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating animation: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)