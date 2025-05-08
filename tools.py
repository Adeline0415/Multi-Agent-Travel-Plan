# tools.py
import json
import logging
import requests
from typing import Tuple, Annotated
from openai import OpenAI
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

# Standalone weather functions
def get_lat_long(api_key: str, location_name: str) -> Tuple[float, float]:
    """Get latitude and longitude from a location name."""
    API_URL = 'https://api.opencagedata.com/geocode/v1/json'   
    
    params = {
        'q': location_name,
        'key': api_key
    }
    
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    else:
        raise Exception("Geocoding API request failed")

def get_weather_forecast(api_key: str, location_name: str, start_date: str, end_date: str) -> str:
    """Get the weather forecast for a given location and date range."""
    try:
        latitude, longitude = get_lat_long(api_key, location_name)
        logger.info(f"latitude:{latitude}, longitude:{longitude}")
        
        base_url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_min,temperature_2m_max,rain_sum,weather_code',
            'timezone': 'GMT'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            forecast_data = response.json()
            
            forecast = forecast_data['daily']
            summary = f"Weather forecast for {location_name} from {start_date} to {end_date}:"
            for date, temp_min, temp_max, rain, weather_code in zip(
                forecast['time'], 
                forecast['temperature_2m_min'], 
                forecast['temperature_2m_max'], 
                forecast['rain_sum'], 
                forecast['weather_code']
            ):
                summary += (
                    f"\nDate: {date}, Min Temp: {temp_min}°C, Max Temp: {temp_max}°C, "
                    f"Rain: {rain}mm, Weather Code: {weather_code}"
                )
            return summary
        else:
            raise Exception("Open-Meteo API request failed")
    except Exception as e:
        return str(e)

# Semantic Kernel plugins
class WeatherPlugin:
    """Plugin for weather-related functions"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    @kernel_function(
        description="Get weather forecast for a location",
        name="get_weather_forecast"
    )
    def get_weather_forecast(self, 
                          location_name: Annotated[str, "The name of the location"],
                          start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
                          end_date: Annotated[str, "End date in YYYY-MM-DD format"]) -> str:
        """Get the weather forecast for a given location and date range."""
        return get_weather_forecast(self.api_key, location_name, start_date, end_date)


class DALLEPlugin:
    """Plugin for generating images with DALLE"""
    def __init__(self, dalle_client):
        self.dalle_client = dalle_client
        self.definitions = [self.generate_image_definition()]  # 添加這行來提供定義
        
    def generate_image_definition(self):  # 添加此方法以創建定義
        """Create definition for the generate_image function"""
        return {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on the given prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The query for generating the image"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
        
    @kernel_function(
        description="Generate an image based on the given prompt",
        name="generate_image"
    )
    def generate_image(self, prompt: Annotated[str, "The query for generating the image"]) -> str:
        """Generate an image based on the given prompt."""
        logging.info(f"DALLE prompt: {prompt}")
                
        try:
            result = self.dalle_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                quality="standard",
                size="1024x1024",
                n=1
            )
                        
            result_json = json.loads(result.model_dump_json())
            image_url = result_json['data'][0]["url"]
                        
            logging.info(f"Generated image URL: {image_url}")
            return image_url
        except Exception as e:
            logging.error(f"DALLE image generation error: {e}")
            return f"Error generating image: {str(e)}"