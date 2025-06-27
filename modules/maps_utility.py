from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time
import requests
import re
from datetime import datetime, timedelta
from math import ceil

instan_siang =  [25, 25, 25, 30, 30, 35, 35, 40, 40, 45, 50, 55, 55, 55, 55, 55, 55, 65, 65, 75]
express_siang = [20, 20, 20, 25, 25, 25, 30, 30, 30, 35, 45, 45, 45, 45, 50, 50, 50, 50, 50, 60]

instan_malam =  [20, 20, 25, 30, 30, 30, 35, 35, 35, 45, 45, 45, 50, 50, 50, 55, 55, 60, 60, 70]
express_malam = [15, 15, 15, 25, 25, 25, 25, 25, 25, 40, 40, 40, 40, 45, 50, 50, 50, 50, 50, 60]

def address_to_latlng(address, api_key):
    """
    Mengubah alamat menjadi koordinat latitude dan longitude menggunakan Google Maps Geocoding API.
    
    Params:
    - address: str, alamat seperti "Jl. Sudirman, Jakarta"
    - api_key: str, API key Google Maps

    Returns:
    - (lat, lng): tuple of float, atau (None, None) jika gagal
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] != "OK":
        return None, None

    location = data["results"][0]["geometry"]["location"]
    lat = location["lat"]
    lng = location["lng"]

    return lat, lng


# def resolve_maps_shortlink(shortlink, api_key):
#     # Step 1: Buka browser headless dan resolve shortlink
#     options = Options()
#     options.add_argument("--headless=new")  # Headless modern
#     options.add_argument("--no-sandbox")  # Wajib untuk VPS
#     options.add_argument("--disable-dev-shm-usage")  # Hindari crash karena RAM VPS kecil
#     options.add_argument("--disable-gpu")
#     options.add_argument("--disable-software-rasterizer")
#     options.add_argument("--remote-debugging-port=9222")  # ← Fix penting!
#     options.add_argument("--user-data-dir=/tmp/selenium")  # Biar tidak konflik profile

#     driver = webdriver.Chrome(options=options)

#     driver.get(shortlink)
#     time.sleep(5)  # Tunggu JS redirect
#     final_url = driver.current_url
#     driver.quit()

#     # Step 2: Ekstrak koordinat dari URL hasil redirect
#     match = re.search(r'/@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
#     if not match:
#         return None, None
#     lat, lng = map(float, match.groups())

#     # Step 3: Ambil alamat dari koordinat via Geocoding API
#     endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
#     response = requests.get(endpoint)
#     data = response.json()

#     if data["status"] != "OK":
#         return None, (lat, lng)
    
#     address = data["results"][0]["formatted_address"]
#     return address, (lat, lng)

def resolve_maps_shortlink(shortlink, api_key):
    # Step 1: Buka browser headless
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--user-data-dir=/tmp/selenium")

    driver = webdriver.Chrome(options=options)
    driver.get(shortlink)
    time.sleep(5)
    final_url = driver.current_url
    driver.quit()

    # Step 2: Koordinat
    match = re.search(r'/@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
    if not match:
        return None, None, None, None, None, None
    lat, lng = map(float, match.groups())

    # Step 3: Geocoding API
    endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    response = requests.get(endpoint)
    data = response.json()

    if data["status"] != "OK":
        return None, (lat, lng), None, None, None, None

    results = data["results"][0]
    address = results["formatted_address"]
    components = results["address_components"]

    kelurahan = kecamatan = kota = provinsi = None

    for component in components:
        types = component["types"]
        if "administrative_area_level_4" in types:
            kelurahan = component["long_name"]
        elif "administrative_area_level_3" in types:
            kecamatan = component["long_name"]
        elif "administrative_area_level_2" in types:
            kota = component["long_name"]
        elif "administrative_area_level_1" in types:
            provinsi = component["long_name"]

    return address, (lat, lng), kelurahan, kecamatan, kota, provinsi

def get_travel_distance(origin, destination, api_key, mode="driving"):
    """
    origin: tuple (lat1, lng1)
    destination: tuple (lat2, lng2)
    mode: driving, walking, bicycling, transit
    """
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    params = {
        "origins": f"{origin[0]},{origin[1]}",
        "destinations": f"{destination[0]},{destination[1]}",
        "mode": mode,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] != "OK":
        return None

    row = data["rows"][0]["elements"][0]
    if row["status"] != "OK":
        return None

    distance_text = row["distance"]["text"]
    distance_meters = row["distance"]["value"]
    duration_text = row["duration"]["text"]
    duration_seconds = row["duration"]["value"]

    return {
        "distance_text": distance_text,
        "distance_meters": distance_meters,
        "duration_text": duration_text,
        "duration_seconds": duration_seconds
    }

def distance_cost_rule(dist: float, is_free: bool = False) -> str:
    if is_free and dist >= 9.5:
        return "Subsidi Ongkir 10K"

    if dist < 9.5:
        return "Gratis Ongkir"
        
    if dist >= 9.5 and dist <= 14.4:
        return "Ongkir 10K"
    
    elif dist > 14.4 and dist <= 19.4:
        return "Ongkir 15K"
    
    elif dist > 19.4 and dist <= 24.4:
        return "Ongkir 20K"
    
    elif dist > 24.4 and dist <= 29.4:
        return "Ongkir 25K"
    
    elif dist > 29.4 and dist <= 34.4:
        return "Ongkir 30K"
    
    elif dist > 34.4 and dist <= 39.4:
        return "Ongkir 35K"
    
    elif dist > 39.4 and dist <= 44.4:
        return "Ongkir 40K"
    
    elif dist > 44.4 and dist <= 45:
        return "Ongkir 45K"
    
    else:
        return "Ongkir 45K"

def is_free_delivery(address, free_areas):
    address = address.lower()
    free_areas = [area.lower() for area in free_areas]
    """
    Mengecek apakah alamat mengandung salah satu kata dari daftar area gratis ongkir.

    Params:
    - address: str, alamat lengkap
    - free_areas: list of str, daftar area dengan ongkir gratis

    Returns:
    - bool, True jika alamat mengandung salah satu area, False jika tidak
    - area_matched: str atau None
    """
    address_lower = address.lower()
    for area in free_areas:
        if area.lower() in address_lower:
            return True, area
    return False, None

def waktu_siang(dt: datetime):
    return dt.hour >= 11 and dt.hour < 19

def waktu_malam(dt: datetime):
    return dt.hour >= 20 or dt.hour < 4

def estimasi_tiba(jarak_km: float, tipe: str, waktu_mulai: datetime) -> datetime:
    tipe = tipe.upper()
    km_index = ceil(jarak_km) - 1  # index 0 = km 1

    # Tentukan mode waktu
    if waktu_siang(waktu_mulai):
        instan = instan_siang
        express = express_siang
    elif waktu_malam(waktu_mulai):
        instan = instan_malam
        express = express_malam
    else:
        raise ValueError("Jam operasional hanya 11:00–19:00 (siang) atau 20:00–04:00 (malam)")

    # Hitung waktu tambahan
    if tipe == "FD":
        waktu_tambah = timedelta(minutes=35 * ceil(jarak_km))
    elif tipe == "I":
        if km_index >= len(instan):
            raise ValueError("Jarak terlalu jauh untuk pengiriman Instan (maks 20 km)")
        waktu_tambah = timedelta(minutes=instan[km_index])
    elif tipe == "EX":
        if km_index >= len(express):
            raise ValueError("Jarak terlalu jauh untuk pengiriman Express (maks 20 km)")
        waktu_tambah = timedelta(minutes=express[km_index])
    else:
        raise ValueError("Tipe harus 'FD', 'I', atau 'EX'")

    return waktu_mulai + waktu_tambah
