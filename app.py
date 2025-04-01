import os
import json
import tempfile
import shutil
from urllib.parse import urlparse, parse_qs
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.plot import reshape_as_image
from flask import Flask, request, jsonify, send_file
from shapely.geometry import shape, box
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

app = Flask(__name__)

@app.route('/download-cog', methods=['POST'])
def download_cog():
    data = request.json
    
    # Required parameters
    if not all(key in data for key in ['aoi', 'url']):
        return jsonify({"error": "Missing required parameters. 'aoi' and 'url' are required."}), 400
    
    aoi = data['aoi']  # GeoJSON geometry
    file_url = data['url']  # Titiler URL, file:// URL, or direct file path
    
    # Optional parameters
    output_format = data.get('format', 'tiff').lower()
    
    try:
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        
        # Choose file extension based on format
        if output_format == 'png':
            file_ext = 'png'
        else:
            file_ext = 'tiff'
            output_format = 'tiff'  # Default to tiff if not png
            
        output_path = os.path.join(temp_dir, f"output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "temp_output.tiff")
        
        # Check if the URL is a local file URL or direct path
        parsed_url = urlparse(file_url)
        is_local_file = parsed_url.scheme == 'file' or parsed_url.scheme == '' or os.path.exists(file_url)
        
        # Parse URL parameters if it's not a local file
        if not is_local_file:
            query_params = parse_qs(parsed_url.query)
        else:
            # For local files, we don't have query parameters
            query_params = {}
        
        # Parse the AOI
        if isinstance(aoi, dict):
            geom = shape(aoi)
            bounds = geom.bounds
        else:
            # Assume it's a list of [minx, miny, maxx, maxy]
            bounds = aoi
        
        # Extract min/max values from URL if available
        min_values = []
        max_values = []
        
        # Check if there are r, g, b parameters in the URL
        is_multiband = False
        bands = []
        
        if not is_local_file:
            is_multiband = 'r' in query_params and 'g' in query_params and 'b' in query_params
            if is_multiband:
                # Multi-band request
                for band_param in ['r', 'g', 'b']:
                    if band_param in query_params:
                        bands.append(query_params[band_param][0])
                        
                # Extract min/max if available
                for band in bands:
                    min_key = f'rescale_range.{band}.0'
                    max_key = f'rescale_range.{band}.1'
                    if min_key in query_params and max_key in query_params:
                        min_values.append(float(query_params[min_key][0]))
                        max_values.append(float(query_params[max_key][0]))
        
        if is_local_file:
            # Direct processing of local file
            local_path = file_url
            if parsed_url.scheme == 'file':
                local_path = parsed_url.path
                if os.name == 'nt' and local_path.startswith('/'):
                    # Fix Windows file paths from file:///C:/path to C:/path
                    local_path = local_path[1:]
            
            # Verify the file exists
            if not os.path.exists(local_path):
                return jsonify({"error": f"Local file not found: {local_path}"}), 404
            
            with rasterio.open(local_path) as src:
                # Check if it's multiband
                is_multiband = src.count >= 3
                
                # Get a window for the bounds
                window = from_bounds(*bounds, src.transform)
                
                if is_multiband:
                    # For multiband local files, read the first 3 bands
                    band_files = []
                    for i in range(1, min(4, src.count + 1)):  # Read up to 3 bands (RGB)
                        band_data = src.read(i, window=window)
                        band_file = os.path.join(temp_dir, f"band_{i}.tiff")
                        
                        # Save band to temporary file
                        band_profile = src.profile.copy()
                        band_profile.update({
                            'count': 1,
                            'width': band_data.shape[1],
                            'height': band_data.shape[0],
                            'transform': rasterio.windows.transform(window, src.transform)
                        })
                        
                        with rasterio.open(band_file, 'w', **band_profile) as dst:
                            dst.write(band_data, 1)
                        
                        band_files.append(band_file)
                    
                    # Stack the bands
                    stack_bands(band_files, temp_tiff_path, min_values, max_values)
                else:
                    # Single band local file
                    band_data = src.read(1, window=window)
                    
                    # Save to temporary file
                    band_profile = src.profile.copy()
                    band_profile.update({
                        'count': 1,
                        'width': band_data.shape[1],
                        'height': band_data.shape[0],
                        'transform': rasterio.windows.transform(window, src.transform)
                    })
                    
                    with rasterio.open(temp_tiff_path, 'w', **band_profile) as dst:
                        dst.write(band_data, 1)
        else:
            # TiTiler URL processing
            if is_multiband:
                # Multi-band request from TiTiler
                band_files = []
                for i, band in enumerate(bands):
                    band_url = file_url
                    # Create a TiTiler URL for each band
                    band_file = os.path.join(temp_dir, f"band_{band}.tiff")
                    download_cog_from_titiler(band_url, band_file, bounds)
                    band_files.append(band_file)
                
                # Stack the bands to a temporary TIFF file first
                stack_bands(band_files, temp_tiff_path, min_values, max_values)
            else:
                # Single band from TiTiler
                # Extract min/max if available
                if 'rescale' in query_params:
                    rescale_parts = query_params['rescale'][0].split(',')
                    if len(rescale_parts) == 2:
                        min_values = [float(rescale_parts[0])]
                        max_values = [float(rescale_parts[1])]
                
                # Download the single band to a temporary TIFF file
                download_cog_from_titiler(file_url, temp_tiff_path, bounds)
        
        # Convert to PNG if requested
        if output_format == 'png':
            convert_tiff_to_png(temp_tiff_path, output_path, is_multiband, min_values, max_values)
        else:
            # Just copy the temporary TIFF to the output path
            shutil.copy(temp_tiff_path, output_path)
        
        # Return the file to the user
        return send_file(output_path, as_attachment=True, download_name=f"cog_data.{file_ext}")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary files
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/stack-cogs', methods=['POST'])
def stack_cogs_endpoint():
    data = request.json
    
    # Required parameters
    if not all(key in data for key in ['urls', 'aoi']):
        return jsonify({"error": "Missing required parameters. 'urls' and 'aoi' are required."}), 400
    
    urls = data['urls']  # List of Titiler URLs
    aoi = data['aoi']  # GeoJSON geometry
    
    # Optional parameters
    output_format = data.get('format', 'tiff').lower()
    
    try:
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        
        # Choose file extension based on format
        if output_format == 'png':
            file_ext = 'png'
        else:
            file_ext = 'tiff'
            output_format = 'tiff'  # Default to tiff if not png
            
        output_path = os.path.join(temp_dir, f"output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "temp_output.tiff")
        
        # Parse the AOI
        if isinstance(aoi, dict):
            geom = shape(aoi)
            bounds = geom.bounds
        else:
            # Assume it's a list of [minx, miny, maxx, maxy]
            bounds = aoi
        
        # Download each URL and get the file paths
        band_files = []
        min_values = []
        max_values = []
        
        for url in urls:
            # Parse URL to get min/max values
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # Extract min/max if available
            if 'rescale' in query_params:
                rescale_parts = query_params['rescale'][0].split(',')
                if len(rescale_parts) == 2:
                    min_values.append(float(rescale_parts[0]))
                    max_values.append(float(rescale_parts[1]))
            
            # Download the band
            band_file = os.path.join(temp_dir, f"band_{len(band_files)}.tiff")
            download_cog_from_titiler(url, band_file, bounds)
            band_files.append(band_file)
        
        # Stack all the bands to a temporary TIFF file
        stack_bands(band_files, temp_tiff_path, min_values, max_values)
        
        # Convert to PNG if requested
        if output_format == 'png':
            is_multiband = len(band_files) >= 3
            convert_tiff_to_png(temp_tiff_path, output_path, is_multiband, min_values, max_values)
        else:
            # Just copy the temporary TIFF to the output path
            shutil.copy(temp_tiff_path, output_path)
        
        # Return the file to the user
        return send_file(output_path, as_attachment=True, download_name=f"stacked_cog.{file_ext}")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary files
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

def download_cog_from_titiler(titiler_url, output_path, bounds):
    """Download a COG from TiTiler based on bounds."""
    # Create a TiTiler crop request with the bounds
    crop_url = f"{titiler_url.split('?')[0]}/crop/{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
    
    # Keep the query parameters from the original URL
    if '?' in titiler_url:
        crop_url += f"?{titiler_url.split('?')[1]}"
    
    # Download the cropped COG
    response = requests.get(crop_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download COG: {response.status_code} - {response.text}")
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path

def stack_bands(band_files, output_path, min_values=None, max_values=None):
    """Stack multiple single-band GeoTIFFs into a multi-band GeoTIFF."""
    # Get metadata from the first file
    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()
    
    # Update the metadata for a multi-band raster
    meta.update({
        'count': len(band_files),
        'driver': 'GTiff'
    })
    
    # Create the output file
    with rasterio.open(output_path, 'w', **meta) as dst:
        # For each band file, read and write the data
        for i, band_file in enumerate(band_files, 1):
            with rasterio.open(band_file) as src:
                data = src.read(1)
                
                # Apply min/max rescaling if provided
                if min_values and max_values and i <= len(min_values) and i <= len(max_values):
                    min_val = min_values[i-1]
                    max_val = max_values[i-1]
                    
                    # Rescale data
                    data = np.clip(data, min_val, max_val)
                    data = (data - min_val) / (max_val - min_val) * 255
                    data = data.astype(np.uint8)
                
                dst.write(data, i)

def convert_tiff_to_png(tiff_path, png_path, is_multiband=False, min_values=None, max_values=None):
    """Convert a GeoTIFF to PNG format."""
    with rasterio.open(tiff_path) as src:
        # Read data
        if is_multiband or src.count >= 3:
            # Read as RGB
            rgb = src.read([1, 2, 3])
            # Convert to image shape (height, width, channels)
            rgb = reshape_as_image(rgb)
            
            # Apply min/max normalization if provided
            if min_values and max_values and len(min_values) >= 3 and len(max_values) >= 3:
                # Normalize bands individually
                for i in range(3):
                    min_val = min_values[i]
                    max_val = max_values[i]
                    # Clip values and rescale to 0-1
                    rgb[:, :, i] = np.clip(rgb[:, :, i], min_val, max_val)
                    rgb[:, :, i] = (rgb[:, :, i] - min_val) / (max_val - min_val)
            
            # Save as PNG
            plt.imsave(png_path, rgb)
        else:
            # Single band, save as grayscale
            data = src.read(1)
            
            # Apply min/max normalization if provided
            if min_values and max_values and len(min_values) > 0 and len(max_values) > 0:
                min_val = min_values[0]
                max_val = max_values[0]
                
                # Create a normalized colormap
                norm = Normalize(vmin=min_val, vmax=max_val)
                data = norm(data)
            
            # Save as PNG
            plt.imsave(png_path, data, cmap='gray')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 