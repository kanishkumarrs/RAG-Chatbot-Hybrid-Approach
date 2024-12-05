import os
from typing import Optional
import requests
from groq import Groq
import json

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

from traffic import get_traffic

# Function to get weather data for London
def get_weather(city="London"):
    url = f" "
    response = requests.get(url)
    data = response.json()
    
    if data.get("cod") != 200:
        return "Couldn't fetch the weather information right now."
    
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    
    return f"The current weather in {city} is {temp}*C "


def prompt_screening(prompt: str):
    screening_prompt = f"""You are a JSON-only response analyzer. Your sole purpose is to output a JSON string matching this exact schema, analyzing if the input requires weather, traffic, or crime database download. If the user doesn't mention a location, assume it to be "london".

    OUTPUT_SCHEMA = {{
        "weather_api": {{
            "required": boolean,
            "locations": string[]  // Default to ["london"] if no locations are specified
        }},
        "traffic_api": {{
            "required": boolean,
            "locations": string[]  // Default to ["london"] if no locations are specified
        }},
        "crime_database_download": {{
            "required": boolean
        }}
    }}

    Rules:
    1. ONLY output valid JSON nothing else (don't include explanations or additional text)
    2. NEVER include explanations or additional text
    3. Empty locations must be [] unless "london" is assumed due to unspecified location
    4. All fields are required
    5. Use lowercase for city names
    6. Set required: false for unrelated queries
    7. If in the user query they do not ask regarding the above three things then put required: false for all fields.

    Input: {prompt}
    """

    requirements = ask_groq(screening_prompt)
    print(requirements)
    parsed_req = json.loads(requirements)
    response = ""
    
    # Handle weather requests
    if parsed_req["weather_api"]["required"]:
        for city in parsed_req["weather_api"]["locations"]:
            weather_data = get_weather(city)
            response += f"Weather for {city}: {str(weather_data)}\n\n"
    
    # Handle traffic requests        
    if parsed_req["traffic_api"]["required"]:
        for city in parsed_req["traffic_api"]["locations"]:
            traffic_data = get_traffic()
            response += f"Traffic for {city}: {str(traffic_data)}\n\n"
    
    # Handle crime data download
    if parsed_req["crime_database_download"]["required"]:
        crime_data = download_crime_data()
        response += f"Crime data: {str(crime_data)}\n\n"
    
    return response



def ask_groq(prompt: str, api_key: Optional[str] = None) -> str:
    client = Groq(
        api_key=api_key or os.environ.get("GROQ_API_KEY")
    )
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a JSON-only response generator. You must:
                1. ONLY output valid JSON
                2. NEVER include explanations or additional text
                3. Follow the exact schema provided
                4. Return error message in JSON if input is invalid"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.1,  # Lower temperature for more consistent JSON output
    )
    
    return chat_completion.choices[0].message.content


# Function to download Borough region crime data
def download_crime_data():
    chrome_options = Options()
    download_dir = "C:/path/to/your/download/folder"  # Specify your download directory
    prefs = {
        "download.default_directory": download_dir,  # Set download directory
        "download.prompt_for_download": False,  # Disable download prompt
        "directory_upgrade": True,
        "safebrowsing.enabled": True  # Allow downloads without prompt
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("start-maximized")

    # Initialize the Chrome driver with options
    driver = webdriver.Chrome(options=chrome_options)
    url = "https://data.london.gov.uk/dataset/recorded_crime_summary"
    driver.get(url)
    time.sleep(5)  # Wait for page load

    try:
        # Find the download link and extract the href attribute
        download_button = driver.find_element(By.CSS_SELECTOR, "a.dp-resource__button")
        download_url = download_button.get_attribute("href")
        if download_url:
            driver.get(download_url)
            print("Navigated directly to the download URL for crime data.")
        else:
            print("Download URL not found.")
    except Exception as e:
        print("Error finding or clicking the download link:", e)
    finally:
        time.sleep(20)  # Adjust for download completion time
        driver.quit()
