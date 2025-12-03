#!/usr/bin/env python3
"""
Ultra-Fast Parallel Google Maps Downloader
===========================================
Maximum throughput with aggressive parallelism.
Relies on API's built-in rate limiting (429) with smart retries.
"""

import os
import sys
import time
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Dict, List
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hmac
import hashlib
import base64
from urllib.parse import urlencode
from io import BytesIO
import threading

sys.path.insert(0, os.path.dirname(__file__))
from maps_core import calculate_tile_grid, stitch_mosaic, stitch_mosaic_streaming


class FastSessionPool:
    """Thread-local session pool optimized for high throughput."""
    def __init__(self, pool_size: int = 100):
        self._local = threading.local()
        self._pool_size = pool_size
    
    def get(self):
        if not hasattr(self._local, 'session'):
            session = requests.Session()
            # No automatic retries - we handle 429 manually
            adapter = HTTPAdapter(
                pool_connections=self._pool_size,
                pool_maxsize=self._pool_size,
                max_retries=0  # We handle retries manually for speed
            )
            session.mount('https://', adapter)
            self._local.session = session
        return self._local.session


_pool = FastSessionPool(100)


def build_signed_url(lat, lon, zoom, tile_size_px, scale, api_key, secret):
    """Build signed URL for Google Maps Static API."""
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
        return f"https://maps.googleapis.com{resource}&signature={encoded_signature}"
    else:
        return "https://maps.googleapis.com/maps/api/staticmap?" + urlencode(params)


def download_tile_aggressive(
    lat: float, lon: float,
    zoom: int, tile_size_px: int, scale: int,
    api_key: str, secret: str,
    crop_bottom: int,
    output_path: str = None,
    max_retries: int = 5
) -> Tuple[bool, Optional[Image.Image]]:
    """
    Download tile with aggressive retry strategy.
    Returns (success, image_or_none).
    """
    url = build_signed_url(lat, lon, zoom, tile_size_px, scale, api_key, secret)
    session = _pool.get()
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                if response.headers.get('content-type', '').startswith('image'):
                    img = Image.open(BytesIO(response.content))
                    w, h = img.size
                    if crop_bottom > 0:
                        img = img.crop((0, 0, w, h - crop_bottom))
                    
                    if output_path:
                        img.save(output_path, 'JPEG', quality=80)  # Fast compression
                        img.close()
                        return True, None
                    return True, img
                return False, None
            
            elif response.status_code == 429:
                # Rate limited - short sleep and retry
                time.sleep(0.1 * (attempt + 1))
                continue
            
            elif response.status_code >= 500:
                # Server error - retry
                time.sleep(0.05 * (attempt + 1))
                continue
            
            else:
                # Client error - don't retry
                return False, None
                
        except requests.exceptions.Timeout:
            time.sleep(0.1)
            continue
        except Exception:
            time.sleep(0.05)
            continue
    
    return False, None


def worker_disk(args):
    """Worker for disk-based downloads."""
    req, zoom, tile_size, scale, api_key, secret, crop_bottom, temp_dir = args
    
    output_path = os.path.join(temp_dir, f"tile_{req['row']:03d}_{req['col']:03d}.jpg")
    success, _ = download_tile_aggressive(
        req['lat'], req['lon'],
        zoom, tile_size, scale,
        api_key, secret, crop_bottom,
        output_path
    )
    
    return {
        'row': req['row'],
        'col': req['col'],
        'index': req['index'],
        'file': output_path if success else None,
        'success': success
    }


def worker_memory(args):
    """Worker for in-memory downloads."""
    req, zoom, tile_size, scale, api_key, secret, crop_bottom = args
    
    success, img = download_tile_aggressive(
        req['lat'], req['lon'],
        zoom, tile_size, scale,
        api_key, secret, crop_bottom
    )
    
    return {
        'row': req['row'],
        'col': req['col'],
        'index': req['index'],
        'image': img,
        'success': success
    }


def download_satellite_map_fast(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int = 19,
    tile_size_px: int = 640,
    scale: int = 2,
    crop_bottom: int = 40,
    api_key: str = None,
    secret: str = None,
    max_workers: int = 50,
    verbose: bool = True,
    output_path: str = None,
    use_disk: bool = None
) -> Tuple[Optional[Image.Image], Optional[Dict]]:
    """
    Download satellite mosaic with maximum parallelism.
    
    Args:
        max_workers: Concurrent threads (50-100 recommended with URL signing)
        output_path: Save mosaic directly to this path
        use_disk: Force disk-based (True) or memory-based (False)
    """
    if api_key is None:
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY') or os.environ.get('GMAPS_KEY')
    if secret is None:
        secret = os.environ.get('GOOGLE_MAPS_SECRET')
    
    if not api_key:
        print("[Fast] ERROR: No API key!")
        return None, None
    
    tile_requests, num_rows, num_cols, metadata = calculate_tile_grid(
        lat_min, lat_max, lon_min, lon_max, zoom, tile_size_px
    )
    total_tiles = len(tile_requests)
    
    if use_disk is None:
        use_disk = total_tiles > 150
    
    if verbose:
        print(f"[Fast] Downloading {total_tiles} tiles ({num_rows}x{num_cols})")
        print(f"[Fast]   Workers: {max_workers}, Mode: {'disk' if use_disk else 'memory'}")
        print(f"[Fast]   URL signing: {'YES' if secret else 'NO'}")
    
    start_time = time.time()
    temp_dir = None
    
    try:
        if use_disk:
            temp_dir = tempfile.mkdtemp(prefix='gmaps_')
            
            work_items = [
                (req, zoom, tile_size_px, scale, api_key, secret, crop_bottom, temp_dir)
                for req in tile_requests
            ]
            
            results = []
            completed = 0
            last_report = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(worker_disk, item): item for item in work_items}
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if verbose and (completed - last_report >= 50 or completed == total_tiles):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"[Fast]   Progress: {completed}/{total_tiles} ({rate:.1f} t/s)")
                        last_report = completed
            
            download_time = time.time() - start_time
            results.sort(key=lambda x: x['index'])
            success_count = sum(1 for r in results if r['success'])
            
            if verbose:
                print(f"[Fast] Downloaded {success_count}/{total_tiles} in {download_time:.2f}s")
                print(f"[Fast]   Throughput: {success_count/download_time:.1f} tiles/sec")
            
            if verbose:
                print(f"[Fast] Stitching...")
            
            stitch_start = time.time()
            tile_files = {(r['row'], r['col']): r['file'] for r in results if r['file']}
            
            mosaic_path = output_path if output_path else os.path.join(temp_dir, 'mosaic.jpg')
            
            mosaic_size = stitch_mosaic_streaming(
                tile_files, num_rows, num_cols,
                tile_size_px, scale, crop_bottom,
                mosaic_path, quality=85
            )
            
            stitch_time = time.time() - stitch_start
            
            if verbose:
                print(f"[Fast] Mosaic: {mosaic_size[0]}x{mosaic_size[1]} px ({stitch_time:.1f}s)")
            
            mosaic = None if output_path else Image.open(mosaic_path)
            
        else:
            work_items = [
                (req, zoom, tile_size_px, scale, api_key, secret, crop_bottom)
                for req in tile_requests
            ]
            
            results = []
            completed = 0
            last_report = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(worker_memory, item): item for item in work_items}
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if verbose and (completed - last_report >= 50 or completed == total_tiles):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"[Fast]   Progress: {completed}/{total_tiles} ({rate:.1f} t/s)")
                        last_report = completed
            
            download_time = time.time() - start_time
            results.sort(key=lambda x: x['index'])
            success_count = sum(1 for r in results if r['success'])
            
            if verbose:
                print(f"[Fast] Downloaded {success_count}/{total_tiles} in {download_time:.2f}s")
                print(f"[Fast]   Throughput: {success_count/download_time:.1f} tiles/sec")
            
            tiles_for_stitch = [{'row': r['row'], 'col': r['col'], 'image': r['image']} for r in results]
            mosaic = stitch_mosaic(tiles_for_stitch, num_rows, num_cols, tile_size_px, scale, crop_bottom)
            
            if output_path:
                mosaic.save(output_path, 'JPEG', quality=85)
            
            if verbose:
                print(f"[Fast] Mosaic: {mosaic.size[0]}x{mosaic.size[1]} px")
        
        elapsed = time.time() - start_time
        
        metadata['download_method'] = 'fast_parallel'
        metadata['max_workers'] = max_workers
        metadata['tiles_success'] = success_count
        metadata['tiles_total'] = total_tiles
        metadata['download_time'] = download_time
        metadata['total_time'] = elapsed
        metadata['throughput'] = success_count / download_time if download_time > 0 else 0
        metadata['use_disk'] = use_disk
        metadata['url_signing'] = secret is not None
        
        return mosaic, metadata
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
