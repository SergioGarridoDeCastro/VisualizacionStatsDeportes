from json import JSONDecodeError
import fastf1.events
from statsbombpy import sb
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import leaguegamefinder, leagueleaders, commonteamroster
import pandas as pd
import numpy as np
import fastf1
import streamlit as st
from nba_api.stats.library.http import NBAStatsHTTP

# ----------------     FUTBOL     ----------------
matches = sb.matches(competition_id=43, season_id=106)
match_dict = {home+' - '+away: match_id
                 for match_id, home, away
                 in zip(matches['match_id'], matches['home_team'], matches['away_team'])}

def get_starting_XI(match_id, team):
    events = sb.events(match_id=match_id)
    events = events[events["team"]==team]
    players = events[pd.isna(events["player"])==False]["player"].unique()
    eleven = players[:11] # first eleven

    lineups = sb.lineups(match_id)
    lineup = lineups[team][lineups[team]['player_name'].isin(list(set(eleven)))][['player_name', 'jersey_number']].sort_values('jersey_number')
    lineup.columns = ['Player', 'Number']
    lineup.index = lineup['Number']
    return lineup['Player']

def get_competitions():
    competiciones_df = sb.competitions()
    # Crear un diccionario con las competiciones donde el nombre es la clave y el competition_id es el valor
    competiciones_dict = {f"{row['competition_name']} {row['season_name']}": (row['competition_id'], row['season_id']) for _, row in competiciones_df.iterrows()}
    return competiciones_dict

def get_matches_by_competition(competition_id, season_id):
    partidos_df = sb.matches(competition_id= competition_id, season_id=season_id)
    # Imprime los valores de 'home_team' y 'away_team' para entender su estructura
    print(partidos_df.columns)
    partidos_dict = {f"{row['home_team']} - {row['away_team']}": row['match_id'] for _, row in partidos_df.iterrows()}
    return partidos_dict, partidos_df
    
def get_competition_stage(match_id):
    sb.events(match_id=match_id)

# ----------------     BALONCESTO     ----------------

def get_basket_teams():
    teams_df = teams.get_teams()
    teams_dict = {team['full_name']: team['id'] for team in teams_df}
    return teams_dict

def get_team_id(team_name):
    teams_dict = get_basket_teams()
    for team in teams_dict:
        if team_name in team['full_name']:
            return team['id']
    return None

def get_season_nba():
    temporadas_dict = {str(year) + '-' + str(year+1)[-2:]: str(year) + '-' + str(year+1)[-2:] for year in range(2000, 2024)}  # De 2000 a 2023
    return temporadas_dict

def get_season_id(season):
    seasons_dict = get_season_nba()
    for season in seasons_dict:
        if season in season[0]:
            return season[1]
    return None


def get_matches_by_nba_team(team_id, season):
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
    partidos = gamefinder.get_data_frames()[0]
    # Crear un diccionario con los partidos en formato "opponent - fecha"
    partidos['MATCHUP_DATE'] = partidos['MATCHUP'] + " - " + partidos['GAME_DATE']
    partidos_dict = {row['MATCHUP_DATE']: row['GAME_ID'] for _, row in partidos.iterrows()}
    return partidos_dict

def get_basket_players():
    return players.get_players()

# Se considera buzzer beater a cualquier canasta encestada cuando el tiempo es 00.0. Para simplificar se aumenta el limite a 0.02 segundos
def detect_buzzer_beater(pbp_data):
    buzzer_beaters = []
    for index, row in pbp_data.interrows():
        if row['EVENTMSGTYPR']== 1 and (row['PCTIMESTRING'] == '00:00.0' or row['PCTIMESTRING'].endswith('00.1') or row['PCTIMESTRING'].endswith('00.2')):
            buzzer_beaters.append(row)
    return buzzer_beaters

# Se considera clutch una jugada cuando los últimos 2 minutos del último cuarto y/o las prórrogas la diferencia en 
# el marcador entre ambos equipos es menor o igual que 5
def detect_clutch_plays(pbp_data):
    clutch_plays=[]
    fourth_quarter_plays = pbp_data[pbp_data['PERIOD'] >= 4] #Incluye el ultimo cuarto y las prorrogas

    for index, row in fourth_quarter_plays.interrows():
        if row['PCTIMESTRING'].startswith('01:') or row['PCTIMESTRING'].startswith('00:'):
            home_score = row['SCORE'].split('-')[0]
            away_score = row['SCORE'].split('-')[1]
            score_diff = abs(int(home_score) - int(away_score))

            if score_diff <= 5:
                clutch_plays.append(row)
    return clutch_plays

def get_team_id(team_name):
    nba_teams = teams.get_teams()  # Obtiene todos los equipos de la NBA con sus nombres e IDs
    for team in nba_teams:
        if team['abbreviation'].lower() == team_name.lower():
            return team['id']
    return None  # Devuelve None si no se encuentra el equipo

def get_team_roster(team_name, season):
    team_id = get_team_id(team_name)
    
    if team_id is None:
        raise ValueError(f"No se encontró un equipo con el nombre '{team_name}'")
    
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        return roster.get_data_frames()[0]
    except NBAStatsHTTP as e:
        st.write(f"Error en la solicitud a la API de la NBA: {e}")
        return None
    except JSONDecodeError as e:
        st.write(f"Error decodificando la respuesta JSON: {e}")
        return None
    except Exception as e:
        st.write(f"Ocurrió un error inesperado: {e}")
        return None
    

# ----------------     FORMULA 1     ----------------

# Se considera, inicialmente, año de inicio 2018 para evitar errores debido a que fast_f1 recoge datos desde
# 2018 y Ergast recoge datos desde 1950 pero no coinciden todos los datos recogidos.
def get_available_seasons(start_year=2018, end_year=2024):
    available_seasons = {}
    for year in range(start_year, end_year + 1):
        try:
            # Intentar obtener el calendario de eventos para ese año
            fastf1.get_event_schedule(year)
            available_seasons[year] = year  # El diccionario usa el año como clave y valor
        except Exception as e:
            # Si ocurre algún error, simplemente no agregamos el año al diccionario
            pass
    return available_seasons

def get_f1_grandprix(year):
    grandprix_df = fastf1.get_event_schedule(year)
    grandprix_dict = dict(zip(grandprix_df['EventName'], grandprix_df['RoundNumber']))
    return grandprix_dict


def get_f1_championship_classification(season, gp):
# Obtener todos los eventos de la temporada
    event_schedule = fastf1.get_event_schedule(season)
    
    # Filtrar los eventos que han ocurrido antes del GP actual
    current_event = fastf1.get_event(season, gp)
    past_events = event_schedule[event_schedule['EventDate'] < current_event['EventDate']]
    
    # Crear un diccionario para almacenar los puntos de cada piloto
    championship_points = {}

    # Iterar sobre los eventos anteriores y acumular los puntos
    for event in past_events.itertuples():
        race = fastf1.get_session(season, event.EventName, 'Race')
        race.load()
        
        # Sumar los puntos de cada piloto
        for result in race.results.itertuples():
            driver_id = result.DriverNumber
            points = result.Points
            if driver_id in championship_points:
                championship_points[driver_id] += points
            else:
                championship_points[driver_id] = points
    
    # Guardar la clasificación antes de la carrera actual
    before_race = pd.DataFrame(list(championship_points.items()), columns=['DriverNumber', 'Points'])
    
    # Obtener la carrera actual y sumar los puntos a los pilotos
    current_race = fastf1.get_session(season, gp, 'Race')
    current_race.load()

    # Crear una copia de la clasificación para después de la carrera
    after_race = before_race.copy()

    # Actualizar los puntos con los resultados de la carrera actual
    for result in current_race.results.itertuples():
        driver_id = result.DriverNumber
        points = result.Points
        if driver_id in after_race['DriverNumber'].values:
            after_race.loc[after_race['DriverNumber'] == driver_id, 'Points'] += points
        else:
            after_race = after_race.append({'DriverNumber': driver_id, 'Points': points}, ignore_index=True)

    return before_race, after_race