import requests
import base64
import exifread
from datetime import datetime

def _convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def convert_to_degrees(gps_data):
    if not gps_data:
        return None
    lat = gps_data.get('GPSLatitude')
    lat_ref = gps_data.get('GPSLatitudeRef')
    lon = gps_data.get('GPSLongitude')
    lon_ref = gps_data.get('GPSLongitudeRef')
    
    if lat and lon and lat_ref and lon_ref:
        latitude = _convert_to_degrees(lat)
        if lat_ref != 'N':
            latitude = -latitude
        
        longitude = _convert_to_degrees(lon)
        if lon_ref != 'E':
            longitude = -longitude
        
        location = (latitude, longitude)
    else:
        location = None
    
    return location

def gps_to_location_detailed(latitude, longitude, api_key):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'latlng': f"{latitude},{longitude}",
        'key': api_key,
        # 'language': 'zh-CN'  # Set language to Simplified Chinese
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            # Extract formatted address and address components from the response
            result = data['results'][0]
            formatted_address = result['formatted_address']
            address_components = result['address_components']
            return {
                'formatted_address': formatted_address,
                'address_components': address_components
            }
        else:
            return {"error": f"Error: {data['status']}"}

    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_photo_datetime_exif(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if 'EXIF DateTimeOriginal' in tags:
                date_str = str(tags['EXIF DateTimeOriginal'])
                timestamp = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                return timestamp.isoformat()
    except Exception as e:
        print(f"错误: {str(e)}")
    return None