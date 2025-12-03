#!/usr/bin/env python3
"""
Core Google Maps Satellite Imagery Functions
=============================================
Low-level tile download and coordinate conversion utilities.
Used by both sequential and MPI-parallel implementations.
"""

import os
import math
import time
import requests
import hmac
import hashlib
import base64
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Dict, Optional


def latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Convert lat/lon to pixel coordinates in Web Mercator projection."""
    world_px = 256 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * world_px
    siny = math.sin(math.radians(lat))
    siny = max(-0.9999, min(0.9999, siny))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_px
    return x, y


def pixel_to_latlon(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Convert pixel coordinates to lat/lon in Web Mercator projection."""
    world_px = 256 * (2 ** zoom)
    lon = x / world_px * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / world_px
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def calculate_tile_grid(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int, tile_size_px: int
) -> Tuple[List[Dict], int, int, Dict]:
    """
    Calculate tile grid for a bounding box.
    
    Returns:
        (tile_requests, num_rows, num_cols, metadata)
    """
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    meters_per_pixel = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    meters_per_tile = meters_per_pixel * tile_size_px
    
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_meters = lat_range * 111000
    lon_meters = lon_range * 111000 * math.cos(math.radians(center_lat))
    
    num_rows = max(1, int(math.ceil(lat_meters / meters_per_tile)))
    num_cols = max(1, int(math.ceil(lon_meters / meters_per_tile)))
    
    # Get center in pixel coordinates
    cx, cy = latlon_to_pixel(center_lat, center_lon, zoom)
    step_px = tile_size_px
    
    tile_requests = []
    for i in range(num_rows):
        for j in range(num_cols):
            dx_px = (j - (num_cols - 1) / 2.0) * step_px
            dy_px = (i - (num_rows - 1) / 2.0) * step_px
            x = cx + dx_px
            y = cy + dy_px
            lat, lon = pixel_to_latlon(x, y, zoom)
            tile_requests.append({
                'lat': lat,
                'lon': lon,
                'row': i,
                'col': j,
                'index': i * num_cols + j
            })
    
    metadata = {
        'center_lat': center_lat,
        'center_lon': center_lon,
        'meters_per_pixel': meters_per_pixel,
        'meters_per_tile': meters_per_tile,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'total_tiles': num_rows * num_cols,
        'zoom': zoom,
        'tile_size_px': tile_size_px
    }
    
    return tile_requests, num_rows, num_cols, metadata


def download_single_tile(
    lat: float, lon: float,
    zoom: int, tile_size_px: int, scale: int,
    api_key: str, secret: str = None,
    crop_bottom: int = 40,
    max_retries: int = 3
) -> Optional[Image.Image]:
    """
    Download a single satellite tile from Google Maps API.
    
    Returns:
        Cropped PIL Image or None if failed
    """
    params = {
        'center': f'{lat:.10f},{lon:.10f}',
        'zoom': zoom,
        'size': f'{tile_size_px}x{tile_size_px}',
        'scale': scale,
        'maptype': 'satellite',
        'format': 'jpg',
        'key': api_key
    }
    
    if secret:
        path = "/maps/api/staticmap"
        query = urlencode(params, doseq=True)
        resource = f"{path}?{query}"
        decoded_secret = base64.urlsafe_b64decode(secret)
        signature = hmac.new(decoded_secret, resource.encode('utf-8'), hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode('utf-8')
        full_url = f"https://maps.googleapis.com{resource}&signature={encoded_signature}"
    else:
        full_url = "https://maps.googleapis.com/maps/api/staticmap?" + urlencode(params)
    
    backoff = 0.5
    for attempt in range(max_retries):
        try:
            response = requests.get(full_url, timeout=15)
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('image'):
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                if crop_bottom > 0:
                    img = img.crop((0, 0, width, height - crop_bottom))
                return img
            return None
        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                return None
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
    return None


def stitch_mosaic(
    tiles: List[Dict],
    num_rows: int, num_cols: int,
    tile_size_px: int, scale: int,
    crop_bottom: int = 40
) -> Image.Image:
    """Stitch downloaded tiles into a mosaic."""
    actual_tile_size = tile_size_px * scale
    cropped_tile_height = actual_tile_size - crop_bottom
    cropped_tile_width = actual_tile_size
    
    mosaic_width = num_cols * cropped_tile_width
    mosaic_height = num_rows * cropped_tile_height
    
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color=(0, 0, 0))
    
    for tile in tiles:
        if tile.get('image') is not None:
            x_px = tile['col'] * cropped_tile_width
            y_px = tile['row'] * cropped_tile_height
            mosaic.paste(tile['image'], (x_px, y_px))
    
    return mosaic

