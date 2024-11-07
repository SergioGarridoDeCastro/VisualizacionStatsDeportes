import streamlit as st
import pandas as pd
from basket.visulization_basket import avg_possessions_over_time, offensive_ratings, shooting_chart, get_player_shotchartdetail
import controller.auxiliary as auxiliary
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.static import teams
import matplotlib.pyplot as plt



def resumen(game_id, home_team, away_team, season):
    st.subheader(f"Resumen {home_team} vs. {away_team}")
    
    nba_teams = teams.get_teams()
    home_team_name = [team for team in nba_teams if team["abbreviation"] == home_team][0]['full_name']
    away_team_name = [team for team in nba_teams if team["abbreviation"] == away_team][0]['full_name']

    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    data_game = boxscore.get_data_frames()[0]

    starters = data_game.dropna()

    columnas_resumen = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'START_POSITION', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'TO', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FT_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'PLUS_MINUS']
    resumen = starters[columnas_resumen].copy()

    def format_minutes(minutos_str):
        # El formato típico para los minutos es M:SS, entonces eliminamos los decimales de minutos
        if isinstance(minutos_str, str) and ':' in minutos_str:
            # Asegurarse de que los minutos sean correctos
            minutos, segundos = minutos_str.split(':')
            minutos = int(float(minutos))  # Convertimos los minutos a enteros
            segundos = int(float(segundos))  # Convertimos los segundos a enteros
            return f"{minutos}:{str(segundos).zfill(2)}"
        else:
            return minutos_str
        
    
    # Aplicar redondeo y formato de minutos
    resumen[['PTS', 'REB', 'AST', 'STL', 'TO', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FT_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'PLUS_MINUS']] = resumen[['PTS', 'REB', 'AST', 'STL', 'TO', 'BLK', 'FGM', 'FGA', 'FG_PCT', 'FT_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'PLUS_MINUS']].apply(pd.to_numeric, errors='coerce').round(2).applymap(lambda x: f'{x:.2f}' if pd.notna(x) else x)
    resumen['MIN'] = resumen['MIN'].apply(lambda x: format_minutes(x) if pd.notna(x) else x)

    # Dividir por equipo
    equipo_local = resumen[resumen['TEAM_ABBREVIATION'] == resumen['TEAM_ABBREVIATION'].iloc[0]]
    equipo_visitante = resumen[resumen['TEAM_ABBREVIATION'] != resumen['TEAM_ABBREVIATION'].iloc[0]]

    # Función para aplicar color al quinteto inicial
    def highlight_starters(row):
        return ['background-color: lightgreen' if row['START_POSITION'] else '' for _ in row]
    
    # Aplicar el estilo con pandas Styler
    styled_local = equipo_local.style.apply(highlight_starters, axis=1)
    styled_visitante = equipo_visitante.style.apply(highlight_starters, axis=1)
    st.write(f"{home_team_name}")
    st.table(styled_local)
    st.write(f"{away_team_name}") 
    st.table(styled_visitante)

    
    return

def pbp(game_id, home_team, away_team, season):
    return

def avg_possession(game_id, home_team, away_team, season):
    st.subheader("Posesion media")
    st.pyplot(avg_possessions_over_time(game_id, home_team, away_team, season))
    return

def pass_map(game_id, home_team, away_team, season):
    return

def tpass_map(game_id, home_team, away_team, season):
    return

def off_ranting(game_id, home_team, away_team, season):
    return

def threep_shooting(game_id, home_team, away_team, season):
    return

def shot_map(game_id, home_team, away_team, season):
    st.subheader("Mapa de tiros")
    season_type = st.selectbox(
        "Selecciona el tipo de temporada",
        ["Regular Season", "Playoffs", "All-Star"]
    )
    home_roster = auxiliary.get_team_roster(home_team, season)
    player_names_home = home_roster['PLAYER'].tolist()  # Nombres de los jugadores
    player_ids = home_roster['PLAYER_ID'].tolist()  # IDs de los jugadores
    selected_player_name = st.selectbox(f"Selecciona el jugador de {home_team}", player_names_home)
    selected_player_id = player_ids[player_names_home.index(selected_player_name)]
    home_team_id = auxiliary.get_team_id(home_team)
     
    title = selected_player_name + " Shot Chart " + season

    print(title)

    player_shotchart_df, league_avg = get_player_shotchartdetail(selected_player_id, season)

    # Draw Court and plot Shot Chart
    shooting_chart(player_shotchart_df, title=title)
    plt.rcParams['figure.figsize'] = (12, 11)
    st.pyplot(plt.show())

    #st.pyplot(shooting_chart(selected_player_id, season))

    away_roster = auxiliary.get_team_roster(away_team, season)
    player_names_away = away_roster['PLAYER'].tolist()  # Nombres de los jugadores
    player_ids = away_roster['PLAYER_ID'].tolist()  # IDs de los jugadores
    selected_player_name_away = st.selectbox(f"Selecciona el jugadorde {away_team}", player_names_away)
    selected_player_id = player_ids[player_names_away.index(selected_player_name_away)]
    away_team_id = auxiliary.get_team_id(away_team)

    title = selected_player_name_away + " Shot Chart " + season

    print(title)

    player_shotchart_df, league_avg = get_player_shotchartdetail(selected_player_id, season)

    # Draw Court and plot Shot Chart
    shooting_chart(player_shotchart_df, title=title)
    plt.rcParams['figure.figsize'] = (12, 11)
    st.pyplot(plt.show())
    
    #st.pyplot(shooting_chart(game_id, selected_player_id, away_team_id, season))
    return

def voronoi(game_id, home_team, away_team, season):

    return

def momentum(game_id, home_team, away_team, season):
    return

function_dict = {
    "Resumen": resumen,
    "Play By Play": pbp, 
    "Average Possessions": avg_possession, 
    "Individual Pass Map": pass_map, 
    "Team Pass Map": tpass_map,
    'Offensive Rating': off_ranting, 
    '3P Shooting': threep_shooting, 
    "Shot Map": shot_map,
    "Voronoi Diagram": voronoi, 
    'Momentum': momentum
}

def create_plot(selected_chart, game_id, home_team, away_team, season):
    current_fun = function_dict[selected_chart]
    current_fun(game_id, home_team, away_team, season)
     
