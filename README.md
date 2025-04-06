# COG Download and Processing API

A Flask server for downloading and processing Cloud Optimized GeoTIFFs (COGs) based on an area of interest.

## Features

- Download single-band COGs from TiTiler
- Download multi-band COGs with RGB bands
- Stack multiple COGs into a single multi-band image
- Stack multiple layers with transparency and z-index control
- Apply min/max rescaling based on URL parameters
- Crop COGs to a specific area of interest (AOI)
- Output in TIFF or PNG format
- Create animated GIFs from sequence of layers
- Package outputs and raw files in ZIP format

## Installation

### Using Conda (Recommended)

1. Clone this repository
2. Create and activate the conda environment:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate geo-cog-server
```

### Using pip

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will run on port 5000 by default.

## API Endpoints

### 1. Download COG

`POST /download-cog`

Downloads a COG from TiTiler based on an Area of Interest (AOI).

#### Request Body

```json
{
  "url": "https://titiler.example.com/cog/viewer?url=https://example.com/cog.tif",
  "aoi": {
    "type": "Polygon",
    "coordinates": [[[x1, y1], [x2, y2], [x3, y3], [x1, y1]]]
  },
  "format": "tiff"  // Optional, defaults to "tiff". Can also be "png"
}
```

Or using bounding box coordinates:

```json
{
  "url": "https://titiler.example.com/cog/viewer?url=https://example.com/cog.tif",
  "aoi": [minx, miny, maxx, maxy],
  "format": "tiff"  // Optional, defaults to "tiff"
}
```

#### Response

The endpoint returns the processed COG file as an attachment.

### 2. Stack Multiple COGs

`POST /stack-cogs`

Downloads multiple COGs and stacks them into a single multi-band GeoTIFF.

#### Request Body

```json
{
  "urls": [
    "https://titiler.example.com/cog/viewer?url=https://example.com/cog1.tif&rescale=0,100",
    "https://titiler.example.com/cog/viewer?url=https://example.com/cog2.tif&rescale=0,100",
    "https://titiler.example.com/cog/viewer?url=https://example.com/cog3.tif&rescale=0,100"
  ],
  "aoi": {
    "type": "Polygon",
    "coordinates": [[[x1, y1], [x2, y2], [x3, y3], [x1, y1]]]
  },
  "format": "tiff"  // Optional, defaults to "tiff". Can also be "png"
}
```

#### Response

The endpoint returns the stacked COG file as an attachment.

### 3. Stack Layers with Transparency

`POST /stack-layers`

Stack multiple layers with transparency control based on z-index.

#### Request Body

```json
[
  {
    "transparency": 0.5,
    "zIndex": 1000,
    "directURL": "http://127.0.0.1:8000/cog/bbox/minx,miny,maxx,maxy.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=3&rescale=0,1000"
  },
  {
    "transparency": 0.5,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/minx,miny,maxx,maxy.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=3&rescale=0,1000"
  }
]
```

Each layer object in the array must contain:

- `transparency`: Float value from 0.0 (fully transparent) to 1.0 (fully opaque)
- `zIndex`: Integer for determining layer order (higher number = on top)
- `directURL`: TiTiler URL for the layer, including bbox coordinates and band/rescale parameters

#### Optional Query Parameters

- `format`: Output format (`tiff` or `png`). Defaults to `tiff` if not specified.
- `zip`: When set to `yes`, returns a zip file containing both the raw source files and the stacked result.
- `animation`: When set to `yes`, creates an animated GIF from the provided layers in sequence.

#### Optional Layer Properties

- `animation`: Set to `yes` to enable animation mode (alternative to query parameter)
- `id`: Custom identifier for the layer (used in filename when creating ZIP packages)

#### Response

The endpoint returns one of the following depending on the request parameters:

- The stacked layers as a TIFF/PNG file attachment
- A ZIP file containing both the raw source files and the stacked result when `zip=yes`
- An animated GIF when `animation=yes` is specified


### 4. Stack Layers with Transparency

```bash
curl --location 'http://localhost:5000/stack-layers' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '[
  {
    "transparency": 0.5,
    "zIndex": 1000,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=C%3A%5Crepos%5CPoint-prober%5Cdata%5C3RIMG_22MAR2025_0915_L1C_ASIA_MER_V01R00.cog.tif&bidx=1&bidx=3&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  },
  {
    "transparency": 0.5,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=C%3A%5Crepos%5CPoint-prober%5Cdata%5C3RIMG_22MAR2025_0915_L1C_ASIA_MER_V01R00.cog.tif&bidx=1&bidx=2&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  }
]' --output test_output.tif
```

### 5. Stack Layers and Output as PNG

```bash
curl --location 'http://localhost:5000/stack-layers?format=png' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '[
  {
    "transparency": 0.5,
    "zIndex": 1000,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=3&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  },
  {
    "transparency": 0.5,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  }
]' --output test_output.png
```

### 6. Stack Layers and Download Raw Files in ZIP

```bash
curl --location 'http://localhost:5000/stack-layers?zip=yes' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '[
  {
    "transparency": 0.5,
    "zIndex": 1000,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=3&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  },
  {
    "transparency": 0.5,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  }
]' --output all_files.zip
```

### 7. Create an Animated GIF from Layers

```bash
curl --location 'http://localhost:5000/stack-layers?animation=yes' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '[
  {
    "transparency": 1.0,
    "zIndex": 1000,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=3&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  },
  {
    "transparency": 1.0,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  }
]' --output animation.gif
```

### 8. Alternative Method to Create Animation Using Layer Property

```bash
curl --location 'http://localhost:5000/stack-layers' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '[
  {
    "transparency": 1.0,
    "zIndex": 1000,
    "animation": "yes",
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=3&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  },
  {
    "transparency": 1.0,
    "zIndex": 999,
    "directURL": "http://127.0.0.1:8000/cog/bbox/72.0254,15.7501,100.7698,34.2257.tif?url=path/to/cog.tif&bidx=1&bidx=2&bidx=4&rescale=0%2C1000&rescale=0%2C1000&rescale=0%2C1000"
  }
]' --output animation.gif
```

## Additional Information

### Local File Processing

The server can process COGs directly from local files if TiTiler is unavailable or cannot access the files. This serves as a fallback mechanism and requires:

1. TiTiler to be running locally
2. The URL parameter in the TiTiler URL to point to a local file
3. Proper permissions for the Flask application to access the local file

### Error Handling

The API includes robust error handling for common issues:

- Invalid input JSON
- Missing required parameters
- Failed downloads from TiTiler
- Problems with file processing
- Permission errors

Detailed error messages are returned as JSON responses with appropriate HTTP status codes.

### Default Values

- Default output format: TIFF
- Default animation frame rate: 1 second per frame
- Default transparency: 1.0 (fully opaque)
- Default zIndex: 0

## Dependencies

- Flask
- NumPy
- Rasterio
- Requests
- Matplotlib
- Imageio (for animations)
- PIL/Pillow
- Shapely
