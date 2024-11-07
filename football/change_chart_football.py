from football.visulization_football import passing_sonnars, passes_heatmap, passing_network, pressure_heatmap, team_passes_heatmap, xg_flow, shot_map, xt_momentum, passes_map, single_convex_hull, team_convex_hull, progressive_passes, voronoi_diagram
import streamlit as st
import pandas as pd
from statsbombpy import sb
from controller.auxiliary import matches

def resumen(match_data, match_id, home_team, away_team, competition_stage):
        st.write(f"Competicion: {match_data['competition']}")
        st.write(f"Temporada: {match_data['season']}")
        st.write(f"Fecha: {match_data['match_date']}")
        st.write(f"Eliminatoria: {match_data['competition_stage']}")
        st.write(f"Estadio: {match_data['stadium']}")
        st.write(f"Arbitro: {match_data['referee']}")
        st.write(f"Entrenador de {home_team}: {match_data['home_managers']}")
        st.write(f"Entrenador de {away_team}: {match_data['away_managers']}")
        st.markdown('---')
        # Lineups 
        from controller.auxiliary import get_starting_XI
        home_lineup = get_starting_XI(match_id, home_team)
        away_lineup = get_starting_XI(match_id, away_team)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{home_team} - XI Inicial")
            st.table(home_lineup.head(11))
        with col2:
            st.write(f"{away_team} - XI Inicial")
            st.table(away_lineup.head(11))

def individualpm(match_data, match_id, home_team, away_team, competition_stage):
        players_h = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{home_team}'")["player"]))
        players_a = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{away_team}'")["player"]))
        # First passmap
        selected_player_1 = st.selectbox(f"Player: {home_team}", players_h)
        st.subheader("Passmap")
        st.pyplot(passes_map(match_id, selected_player_1, home_team, away_team, competition_stage))
        # Second passmap
        selected_player_2 = st.selectbox(f"Player: {away_team}", players_a)
        st.subheader("Passmap")
        st.pyplot(passes_map(match_id, selected_player_2, away_team, home_team, competition_stage))


def momentum(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Momentum xT')
      st.pyplot(xt_momentum(match_id, home_team, away_team, competition_stage))

def pass_sonnar(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Passing Sonnars')
      st.pyplot(passing_sonnars(match_id, home_team, away_team, competition_stage))

def xgflow(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('xG Flow')
      st.pyplot(xg_flow(match_id, home_team, away_team, competition_stage))

def pass_network(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader(f"{home_team}: Passing Network")
      st.pyplot(passing_network(match_id, home_team, away_team, competition_stage))
      st.subheader(f"{away_team}: Passing Network")
      st.pyplot(passing_network(match_id, away_team, home_team, competition_stage))


def shotmap(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Shot Map')
      st.pyplot(shot_map(match_id, home_team, away_team, competition_stage))
      

def pass_heatmap(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Pass Heatmap')
      st.pyplot(passes_heatmap(match_id, home_team, away_team, competition_stage))
      
def individual_ch(match_data, match_id, home_team, away_team, competition_stage):
    players_h = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{home_team}'")["player"]))
    players_a = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{away_team}'")["player"]))
    # First convex hull
    selected_player_1 = st.selectbox(f"Player: {home_team}", players_h)
    st.subheader('Individual Convex Hull')
    st.pyplot(single_convex_hull(match_id, selected_player_1, home_team, away_team, competition_stage))

    # Second convex hull
    selected_player_2 = st.selectbox(f"Player: {away_team}", players_a)
    st.subheader('Individual Convex Hull')
    st.pyplot(single_convex_hull(match_id, selected_player_2, home_team, away_team, competition_stage))      

def team_ch(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Team Convex Hull')
      st.pyplot(team_convex_hull(match_id, home_team, away_team, competition_stage)) 
      st.pyplot(team_convex_hull(match_id, away_team,  home_team, competition_stage)) 


def team_pm(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Team Pass')
      #st.pyplot(team_passes_map(match_id, home_team, away_team, competition_stage)) 

def voronoi(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Voronoi Diagram')
      players_h = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["goals"].query(f"team=='{home_team}'")["player"]))
      players_a = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["goals"].query(f"team=='{away_team}'")["player"]))
      # First Voronoi Fiagram
      selected_player_1 = st.selectbox(f"Goal: {home_team}", players_h)
      st.pyplot(voronoi_diagram(match_id, selected_player_1, home_team, home_team, competition_stage))

      # Second Voronoi Fiagram
      selected_player_2 = st.selectbox(f"Goal: {home_team}", players_h)
      st.pyplot(voronoi_diagram(match_id, selected_player_2, away_team, home_team, competition_stage))

def pass_progressive(match_data, match_id, home_team, away_team, competition_stage):
      players_h = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{home_team}'")["player"]))
      players_a = list(set(sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"].query(f"team=='{away_team}'")["player"]))
      # First passmap
      selected_player_1 = st.selectbox(f"Player: {home_team}", players_h)
      st.subheader(f'Progressive Passes: {selected_player_1}')
      st.pyplot(progressive_passes(match_id, selected_player_1, away_team, home_team, away_team, competition_stage))

      selected_player_2 = st.selectbox(f"Player: {away_team}", players_a)
      st.subheader(f'Progressive Passes: {selected_player_2}')
      st.pyplot(progressive_passes(match_id, selected_player_2, away_team, home_team, away_team, competition_stage))

def team_xT(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Team Expected Threat')

def pressure_h(match_data, match_id, home_team, away_team, competition_stage):
      st.subheader('Pressure Heatmap')

function_dict = {
        'Resumen': resumen,
        'Individual Pass Map': individualpm,
        'Passing Network': pass_network,
        'Shot Map': shotmap,
        'xG Flow': xgflow,
        'Passing Sonars': pass_sonnar,
        'Momentum xT': momentum,
        'Pass Heatmap': pass_heatmap,
        'Individual Convex Hull': individual_ch,
        'Team Convex Hull': team_ch,
        'Voronoi Diagram': voronoi,
        'Progressive Passes': pass_progressive,
        'Team Expected Threat': team_xT,
        'Pressure Heatmap': pressure_h,
        'Team Pass Map': team_pm
    }

def create_plot(selected_chart, match_data, match_id, home_team, away_team, competition_stage):
    current_fun = function_dict[selected_chart]
    current_fun(match_data, match_id, home_team, away_team, competition_stage)