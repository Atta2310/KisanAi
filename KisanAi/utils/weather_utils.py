import requests

API_KEY = "cfb4cdf28f32906b67d0e60ca283e88e"

PAKISTAN_REGIONS = [
    "Karachi", "Lahore", "Islamabad", "Quetta", "Peshawar",
    "Multan", "Faisalabad", "Hyderabad", "Sukkur", "Rawalpindi",
    "Sargodha", "Gujranwala", "Bahawalpur", "Mardan", "Mingora"
]

CROP_CONDITIONS = {
    "wheat": {"temp_min": 10, "temp_max": 25, "humidity_min": 40, "humidity_max": 70},
    "rice": {"temp_min": 20, "temp_max": 35, "humidity_min": 60, "humidity_max": 90},
    "sugarcane": {"temp_min": 20, "temp_max": 35, "humidity_min": 50, "humidity_max": 85},
}

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},PK&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("main"):
        return {
            "temp": response["main"]["temp"],
            "humidity": response["main"]["humidity"]
        }
    return None

def best_regions_for_crop(crop_name):
    crop_name = crop_name.lower()
    if crop_name not in CROP_CONDITIONS:
        return [], f"❌ Sorry, {crop_name} is not in the supported crop list."

    ideal = CROP_CONDITIONS[crop_name]
    suitable_regions = []

    for region in PAKISTAN_REGIONS:
        weather = get_weather(region)
        if weather:
            if (ideal["temp_min"] <= weather["temp"] <= ideal["temp_max"] and
                ideal["humidity_min"] <= weather["humidity"] <= ideal["humidity_max"]):
                suitable_regions.append(f"{region} ✅ (Temp: {weather['temp']}°C, Humidity: {weather['humidity']}%)")

    if not suitable_regions:
        return [], "⚠ No perfectly suitable regions right now. Try irrigation/greenhouse methods."
    return suitable_regions, "✅ Found best regions!"