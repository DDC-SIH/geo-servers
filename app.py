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
import concurrent.futures
from threading import Lock

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
            layers = request.json
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raw_data = request.get_data(as_text=True)
            logger.debug(f"Raw request data (first 200 chars): {raw_data[:200]}")
            return jsonify({
                "error": "Invalid JSON in request", 
                "details": str(e),
                "help": "Please validate your JSON input. Common issues include missing commas, unquoted property names, or trailing commas."
            }), 400
        
        if not layers or not isinstance(layers, list):
            return jsonify({"error": "Invalid input. Expected a list of layers."}), 400
        
        output_format = request.args.get('format', 'tiff').lower()
        create_zip = request.args.get('zip', 'no').lower() == 'yes'
        create_animation = request.args.get('animation', 'no').lower() == 'yes'
        
        if not create_animation:
            for layer in layers:
                if layer.get('animation', '').lower() == 'yes' or layer.get('animation', False) is True:
                    create_animation = True
                    break
        
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
                
                validated_url, error = extract_and_validate_titiler_url(direct_url)
                if error:
                    logger.error(f"Invalid TiTiler URL for layer {layer_id}: {error}")
                    return None
                
                layer_file = os.path.join(temp_dir, f"layer_{layer_id}.tiff")
                download_result = download_from_titiler(validated_url, layer_file)
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
            future_to_layer = {
                executor.submit(process_layer, i, layer): (i, layer) 
                for i, layer in enumerate(sorted_layers)
            }
            
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

# Other functions remain unchanged...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)