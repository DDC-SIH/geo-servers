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

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
        # Get the layers from the request
        layers = request.json
        
        if not layers or not isinstance(layers, list):
            return jsonify({"error": "Invalid input. Expected a list of layers."}), 400
        
        # Optional query parameters
        output_format = request.args.get('format', 'tiff').lower()
        create_zip = request.args.get('zip', 'no').lower() == 'yes'
        
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")
        
        # Choose file extension based on format
        if output_format == 'png':
            file_ext = 'png'
        else:
            file_ext = 'tiff'
            output_format = 'tiff'  # Default to tiff if not png
            
        output_path = os.path.join(temp_dir, f"stacked_output.{file_ext}")
        temp_tiff_path = os.path.join(temp_dir, "stacked_output_temp.tiff")
        
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
        return jsonify({"error": str(e)}), 500
    
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
        
        # Save as PNG
        if src.count >= 3:
            plt.imsave(png_path, rgb)
        else:
            plt.imsave(png_path, data, cmap='gray')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)