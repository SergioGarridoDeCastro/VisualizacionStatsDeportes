from f1.visulization_f1 import positions_change_race, strategies, team_pace, gear_changes_track, driver_lap_speed, drives_laptime, track_with_corners, animate_classification
import streamlit as st
import pandas as pd
from fastf1 import core
from controller.auxiliary import get_f1_championship_classification
import fastf1
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime




def overview(season, gp):
    st.subheader("Resumen")
    event_schedule = fastf1.get_event_schedule(season)
    event = fastf1.get_event(season, gp)
    
    qualy = event.get_qualifying()
    qualy.load()
    qualy_results = qualy.results
    columnas_qualy = ["DriverNumber", "FullName", "Abbreviation", "TeamName", "CountryCode"]
    qualy_results_filtered = qualy_results[columnas_qualy].copy()
    st.table(qualy_results_filtered)
    show_weather_data(qualy)

    race = event.get_race()
    race.load()

    show_weather_data(race)
    
    st.write(f"Numero de vueltas: {race.total_laps}")
    #Layout del circuito con curvas numeradas
    try:
        st.pyplot(track_with_corners(season, gp, "R"))
    except:
        print("Error: No hay informacion sobre el layout del circuito") #Incluir codigo de error?

    #Animacion clasificacion del mundial antes de la carrera
    before_race, after_race = get_f1_championship_classification(season, gp)
    animation = animate_classification(before_race, after_race)
    animation.save("classification_animation.gif", writer='pillow')

    # Mostrar la animación en Streamlit
    st.image("classification_animation.gif")

    st.write(f"Ganador: {race.results['FullName'][0]}")
    st.write(f"Segundo puesto: {race.results['FullName'][1]}")
    st.write(f"Tercer puesto: {race.results['FullName'][2]}")
    #Animacion clasificacion del mundial despues de la carrera

def show_weather_data(session):
    # Cargar la sesión del GP
    session.load()

    # Obtener los datos meteorológicos
    weather_data = session.weather_data

    # Mostrar los datos meteorológicos en una tabla
    st.subheader(f"Tiempo en la sesión: {session.name}")
    st.write(f"Evento: {session.event['EventName']}")
    st.write(f"Fecha: {session.date}")

    st.write(f"Temperatura del aire: {weather_data['AirTemp'].mean():.2f}°C")
    st.write(f"Temperatura de la pista: {weather_data['TrackTemp'].mean():.2f}°C")
    st.write(f"Humedad: {weather_data['Humidity'].mean():.2f}%")
    st.write(f"Velocidad del viento: {weather_data['WindSpeed'].mean():.2f} km/h")
    st.write(f"Dirección del viento: {weather_data['WindDirection'].mean():.2f}°")
    st.write("---")
    


def change_pos_race(season, gp):
    st.subheader('Cambios de Posición en un carrera')
    
    st.pyplot(positions_change_race(season, gp,  session_type= "R", telemetry = False, weather= True))
    return

def strategy(season, gp):
    st.subheader("Estrategias")

    st.pyplot(strategies(season, gp, "R"))
    return

def circuit_speed(season, gp):
    st.subheader("Velocidad en el circuito")
    st.pyplot(driver_lap_speed(season, gp, "R"))
    return

def gear_changes(season, gp):
    st.subheader("Cambios de marcha")
    st.pyplot(gear_changes_track(season, gp, "R"))
    return

def tpace(season, gp):
    st.subheader("Team Pace")
    st.pyplot(team_pace(season, gp, "R"))
    return

def time_driver(season, gp):
    st.subheader("Distribuccion de tiempos de vuelta por piloto")
    st.pyplot(drives_laptime(season, gp, "R"))
    return


def corner_speed(season, gp):
    return

def telemetry(season, gp):
    return

def circuit(season, gp):
    return



function_dict = {
    "Resumen": overview, 
    "Cambios de Posición en un carrera": change_pos_race, 
    "Estrategias": strategy, 
    "Velocidad en el circuito": circuit_speed, 
    "Cambios de marcha": gear_changes,
    'Ritmo de equipo': tpace, 
    'Tiempos por piloto': time_driver, 
    "Circuito": circuit, 
    "Velocidad en curva": corner_speed, 
    'Telemetria': telemetry
}

def create_plot(selected_chart, season, gp):
    current_fun = function_dict[selected_chart]
    current_fun(season, gp)