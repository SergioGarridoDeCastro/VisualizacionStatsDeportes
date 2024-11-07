from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import requests

# Ruta del ChromeDriver 
CHROME_DRIVER_PATH = ''

# Columnas para el fichero csv 
columnas = ["Name", "Age", "Height", "Weight", "Position", "Player Nationality", "Market Value", "Club", "League", "League Country", "Transfer Type", "Fee", "Foot", "Matches", "Goals", "Assists", "Yellow Cards", "2nd Yellow Cards", "Red Cards", "Minutes", "MinutesxGoal", "Expected Goals (xG)", "Expected Assists (xA)", "Tackles per Game", "Interceptions per Game", "Dribbles per Game", "Key Passes per Game", "Pass Accuracy (%)", "Aerial Duels Won", "Penalties Scored", "Penalties Missed", "Distance Covered per Game", "Contract Expiry Date", "Injury History", "Previous Clubs", "Agent"]

# Inicializar el navegador Chrome
service = Service(CHROME_DRIVER_PATH)
options = webdriver.ChromeOptions()
# options.add_argument("--headless") #Para ejecutar el navegador en modo headless
options.headless = False  # Cambiar a True si se quiere ejecutar en modo headless 
options.add_argument('--disable-dev-shm-usage')  # Necessario para algunos entornos Linux
options.add_argument('--no-sandbox')  # Necessario para algunos entornos Linux
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)

def obtener_datos_transfermarkt(url):
    

    driver.get(url)
    time.sleep(5)

    jugadores = []

    return df_jugadores

def get_datos_jugadores(fila):
    try:
        detalles = fila.text.split('\n')
        contenidos = {}
        return contenidos
    except Exception as e:
        print(f"Error al processar la fila: {str(e)}")
        return {}