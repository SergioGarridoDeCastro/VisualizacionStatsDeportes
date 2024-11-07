# app_main.py

import streamlit as st
from controller.auxiliary import match_dict, matches
import controller.auxiliary as auxiliary
import football.change_chart_football as change_chart_football
import basket.change_chart_basket as change_chart_basket
import f1.change_chart_f1 as change_chart_f1
import pandas as pd
import os
import sys
import basket.connect_api_nba

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
        page_title="Analysis and Prediction Tool"
        #page_icon=
    )

# Título de la aplicación

st.title("Visualización de estadisticas y predicción de resultados deportivos")
st.markdown("---")

# Descripción de la aplicación
st.write("""
### Bienvenido a la aplicación de predicción de resultados deportivos.
Esta aplicación permite visualizar estadísticas y predecir resultados de deportes como Fórmula 1, fútbol y baloncesto.
""")

# Sección para seleccionar el deporte
st.sidebar.header("Selecciona un deporte")
deporte = st.sidebar.selectbox(
    "Deportes",
    ("Fútbol", "Fórmula 1", "Baloncesto", "NFL", "Tenis", "Rugby")
)

if deporte == "Fútbol":
    # Dropdown para seleccionar la competición
    st.sidebar.title("Selecciona Competición")

    # Obtener lista de competiciones
    competiciones = auxiliary.get_competitions()  
    selected_competition = st.sidebar.selectbox("Competición:", competiciones.keys())

    # Obtener partidos en función de la competición seleccionada
    competition_id, season_id = competiciones[selected_competition]
    partidos_competicion, partidos_df = auxiliary.get_matches_by_competition(competition_id, season_id)

    # dropdown for choosing the match
    st.sidebar.title("Selecciona Partido")
    selected_match = st.sidebar.selectbox("Partido:", partidos_competicion.keys())
    

    # Obtener el match_id
    match_id = partidos_competicion[selected_match]
    home_team, away_team = selected_match.split(' - ')

    competition_stage = partidos_df[partidos_df['match_id']==match_id].iloc[0]['competition_stage']


    # choose chart type
    selected_chart = st.sidebar.radio("Selecciona grafica",
                                    ["Resumen", "Passing Network", "Passing Sonars", "Individual Pass Map", "Team Pass Map",
                                        'Progressive Passes', 'xG Flow', "Shot Map", 'Individual Convex Hull', "Team Convex Hull",
                                        "Voronoi Diagram", "Team Expected Threat", "Pressure Heatmap", 'Momentum xT', 'Pass Heatmap']
                                        )
    match_data = partidos_df[partidos_df['match_id']==match_id].iloc[0]
    st.write(f"### {home_team} {match_data['home_score']}:{match_data['away_score']} {away_team}")
    st.markdown('---')
    
    change_chart_football.create_plot(selected_chart, match_data, match_id, home_team, away_team, competition_stage)

    st.markdown('---')
elif deporte == 'Baloncesto':
    st.sidebar.title("Selecciona Equipo")

    # Obtener lista de competiciones
    teams_nba = auxiliary.get_basket_teams() 
    selected_team = st.sidebar.selectbox("Equipo:", teams_nba.keys())

    team_id = teams_nba[selected_team]
    season = auxiliary.get_season_nba()    

    st.sidebar.title("Selecciona Temporada")
    selected_season = st.sidebar.selectbox("Temporada:", season)

    matches_by_team = auxiliary.get_matches_by_nba_team(team_id, selected_season)
    st.sidebar.title("Selecciona Partido")
    selected_match = st.sidebar.selectbox("Partido:", matches_by_team.keys())

    game_id = matches_by_team[selected_match]
    match_without_date, date = selected_match.split(' - ')

    if " vs." in match_without_date:
        home_team, away_team = match_without_date.split(" vs. ")  
    elif " @ " in match_without_date:
        home_team, away_team = match_without_date.split(' @ ')
    else:
        raise ValueError("Formato de partido no reconocido")

    # choose chart type
    selected_chart = st.sidebar.radio("Selecciona grafica",
                                    ["Resumen", "Play By Play", "Average Possessions", "Individual Pass Map", "Team Pass Map",
                                        'Offensive Rating', '3P Shooting', "Shot Map",
                                        "Voronoi Diagram", 'Momentum']
                                        )
    
    change_chart_basket.create_plot(selected_chart, game_id, home_team, away_team, selected_season)
elif deporte == "Fórmula 1":
    st.sidebar.title("Selecciona Temporada")
    season = auxiliary.get_available_seasons()
    selected_season = st.sidebar.selectbox("Temporada:", season)

    st.sidebar.title("Selecciona Gran Premio")
    grandprix = auxiliary.get_f1_grandprix(selected_season)
    selected_grandprix = st.sidebar.selectbox("Gran Premio:", grandprix)

    selected_chart = st.sidebar.radio("Selecciona grafica",
                                    ["Resumen", "Cambios de Posición en un carrera", "Estrategias", "Velocidad en el circuito", "Cambios de marcha",
                                        'Ritmo de equipo', 'Tiempos por piloto', "Circuito", 
                                        "Velocidad en curva", 'Telemetria']
                                        )
    change_chart_f1.create_plot(selected_chart, selected_season, selected_grandprix)

else:
    # En el caso de otros deportes, puedes seguir el mismo esquema.
    st.sidebar.title(f"{deporte}  (en desarrollo)")
    # Aquí puedes agregar lógica para los otros deportes si lo necesitas.
st.markdown('---')
#st.image('sb_icon.png', caption='App made by Sergio Garrido de Castro. Data powered by StatsBomb', use_column_width=True)

# signature
st.sidebar.markdown('---')

