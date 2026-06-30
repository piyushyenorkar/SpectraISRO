"""
SPECTRA — Landsat 9 Data Download via Google Earth Engine

Downloads Landsat 9 Level-2 bands (B2, B3, B4, B10) for configured Indian regions.
Exports as GeoTIFF files to Google Drive.

Prerequisites:
    1. Google Earth Engine account: https://earthengine.google.com
    2. Install: pip install earthengine-api
    3. Authenticate: earthengine authenticate

Usage:
    python download_landsat.py                    # Download all configured regions
    python download_landsat.py --region mumbai    # Download specific region
    python download_landsat.py --list             # List available regions

Output:
    Exported to Google Drive folder: SPECTRA_Landsat9/
    Files per region:
        {region_name}_B2.tif   (Blue band, 30m)
        {region_name}_B3.tif   (Green band, 30m)
        {region_name}_B4.tif   (Red band, 30m)
        {region_name}_B10.tif  (Thermal IR band, 100m native)
"""

import argparse
import sys

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed.")
    print("Run: pip install earthengine-api")
    print("Then: earthengine authenticate")
    sys.exit(1)

import config


def initialize_gee(project_id=None):
    """Initialize Google Earth Engine with a Cloud Project."""
    if project_id:
        try:
            ee.Initialize(project=project_id)
            print(f"✅ Google Earth Engine initialized (project: {project_id})")
            return
        except Exception as e:
            print(f"ERROR: Could not initialize with project '{project_id}': {e}")
            sys.exit(1)
    
    # Try without explicit project (works if user has a default project)
    try:
        ee.Initialize()
        print("✅ Google Earth Engine initialized successfully")
        return
    except Exception:
        pass
    
    print("ERROR: Google Earth Engine requires a Cloud Project ID.")
    print()
    print("Follow these steps to create one (FREE, takes 2 minutes):")
    print("  1. Go to: https://console.cloud.google.com/projectcreate")
    print("  2. Project name: 'spectra-isro' (or anything you like)")
    print("  3. Click 'Create'")
    print("  4. Then go to: https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
    print("  5. Click 'Enable' to enable the Earth Engine API")
    print("  6. Then register it at: https://code.earthengine.google.com/register")
    print()
    print("Once done, run:")
    print("  python ml/download_landsat.py --project YOUR_PROJECT_ID")
    print()
    print("Example:")
    print("  python ml/download_landsat.py --project spectra-isro")
    sys.exit(1)


def get_region_geometry(center_lat, center_lon, size_km):
    """Create a square region of interest centered on given coordinates."""
    # Convert km to degrees (approximate at Indian latitudes)
    half_size_deg = (size_km / 2) / 111.0  # ~111 km per degree
    return ee.Geometry.Rectangle([
        center_lon - half_size_deg,  # west
        center_lat - half_size_deg,  # south
        center_lon + half_size_deg,  # east
        center_lat + half_size_deg,  # north
    ])


def get_best_scene(region_geometry):
    """
    Find the least cloudy Landsat 9 L2 scene for the given region.
    
    Uses LANDSAT/LC09/C02/T1_L2 (Collection 2, Level 2 — surface reflectance).
    """
    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region_geometry)
        .filterDate(config.GEE_DATE_START, config.GEE_DATE_END)
        .filter(ee.Filter.lt("CLOUD_COVER", config.GEE_CLOUD_COVER_MAX))
        .sort("CLOUD_COVER")
    )

    count = collection.size().getInfo()
    if count == 0:
        return None, 0

    best = collection.first()
    return best, count


def apply_scale_factors(image):
    """
    Apply Landsat 9 Collection 2 Level-2 scale factors.
    
    Optical bands (B2-B7): multiply by 0.0000275, add -0.2
    Thermal band (B10): multiply by 0.00341802, add 149.0 (converts to Kelvin)
    
    We normalize optical bands to 0-1 range and thermal to a usable range.
    """
    # Scale optical bands to surface reflectance [0, 1]
    optical = image.select(['SR_B2', 'SR_B3', 'SR_B4']).multiply(0.0000275).add(-0.2).clamp(0, 1)
    
    # Scale thermal band to brightness temperature in Kelvin
    thermal = image.select(['ST_B10']).multiply(0.00341802).add(149.0)
    
    return optical.addBands(thermal)


def export_band(image, band_name, region_name, region_geometry, export_band_name):
    """
    Export a single band as a GeoTIFF to Google Drive.
    
    Returns the export task (must be started manually in GEE Task Manager
    or programmatically via task.start()).
    """
    band_image = image.select(band_name)
    
    task = ee.batch.Export.image.toDrive(
        image=band_image,
        description=f"{region_name}_{export_band_name}",
        folder=config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"{region_name}_{export_band_name}",
        region=region_geometry,
        scale=config.GEE_EXPORT_SCALE,
        crs='EPSG:4326',
        maxPixels=1e9,
        fileFormat='GeoTIFF',
    )
    
    return task


def download_region(region_name, center_lat, center_lon, size_km):
    """Download all 4 bands for a single region."""
    print(f"\n{'─'*50}")
    print(f"📍 Region: {region_name}")
    print(f"   Center: {center_lat}°N, {center_lon}°E")
    print(f"   Size: {size_km}km × {size_km}km")
    
    # Create region geometry
    roi = get_region_geometry(center_lat, center_lon, size_km)
    
    # Find best (least cloudy) scene
    scene, total_scenes = get_best_scene(roi)
    
    if scene is None:
        print(f"   ❌ No scenes found with <{config.GEE_CLOUD_COVER_MAX}% cloud cover")
        print(f"   Try increasing GEE_CLOUD_COVER_MAX in config.py")
        return False
    
    # Get scene metadata
    scene_info = scene.getInfo()
    scene_id = scene_info['properties'].get('LANDSAT_PRODUCT_ID', 'unknown')
    cloud_cover = scene_info['properties'].get('CLOUD_COVER', 'unknown')
    date_acquired = scene_info['properties'].get('DATE_ACQUIRED', 'unknown')
    
    print(f"   Found {total_scenes} scenes, using best:")
    print(f"   Scene ID: {scene_id}")
    print(f"   Date: {date_acquired}")
    print(f"   Cloud Cover: {cloud_cover}%")
    
    # Apply scale factors
    scaled = apply_scale_factors(scene)
    
    # Export each band
    bands_to_export = [
        ('SR_B2', 'B2'),    # Blue
        ('SR_B3', 'B3'),    # Green
        ('SR_B4', 'B4'),    # Red
        ('ST_B10', 'B10'),  # Thermal IR
    ]
    
    tasks = []
    for gee_band, export_name in bands_to_export:
        task = export_band(scaled, gee_band, region_name, roi, export_name)
        task.start()
        tasks.append((export_name, task))
        print(f"   📤 Exporting {export_name}... (task started)")
    
    print(f"   ✅ All 4 export tasks started for {region_name}")
    print(f"   → Check Google Drive folder: {config.GEE_DRIVE_FOLDER}/")
    
    return True


def list_regions():
    """Print all configured download regions."""
    print("\n📍 Configured Download Regions:")
    print(f"{'─'*50}")
    for name, lat, lon, size in config.GEE_REGIONS:
        print(f"   {name:25s} → {lat:.2f}°N, {lon:.2f}°E ({size}km × {size}km)")
    print(f"\nDate range: {config.GEE_DATE_START} to {config.GEE_DATE_END}")
    print(f"Max cloud cover: {config.GEE_CLOUD_COVER_MAX}%")
    print(f"Export folder: Google Drive/{config.GEE_DRIVE_FOLDER}/")


def main():
    parser = argparse.ArgumentParser(
        description="SPECTRA — Download Landsat 9 data via Google Earth Engine"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Google Cloud Project ID (required for GEE). Create one at console.cloud.google.com"
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Download a specific region by name (partial match OK)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all configured regions and exit"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("SPECTRA — Landsat 9 Data Downloader")
    print("=" * 50)

    if args.list:
        list_regions()
        return

    # Initialize GEE
    initialize_gee(project_id=args.project)

    # Filter regions if --region specified
    regions = config.GEE_REGIONS
    if args.region:
        regions = [r for r in regions if args.region.lower() in r[0].lower()]
        if not regions:
            print(f"ERROR: No region matching '{args.region}'")
            list_regions()
            return

    print(f"\n🛰️  Downloading {len(regions)} region(s)...")
    print(f"   Export to: Google Drive/{config.GEE_DRIVE_FOLDER}/")
    
    success_count = 0
    for name, lat, lon, size in regions:
        if download_region(name, lat, lon, size):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"✅ Done! {success_count}/{len(regions)} regions exported.")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Go to: https://code.earthengine.google.com")
    print(f"   2. Click 'Tasks' tab on the right panel")
    print(f"   3. Wait for all tasks to complete (5-15 min each)")
    print(f"   4. Go to Google Drive → {config.GEE_DRIVE_FOLDER}/")
    print(f"   5. Download all .tif files to: ml/data/raw/")
    print(f"   6. Run: python preprocess_landsat.py --input data/raw --output data")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
