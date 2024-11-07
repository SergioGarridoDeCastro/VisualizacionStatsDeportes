# All needed imports
from nba_api.stats.endpoints import leaguedashteamstats, shotchartdetail, scoreboard, playbyplayv2, playercareerstats
from nba_api.stats.static import players, teams
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
import json
import requests
import datetime
import streamlit as st
import matplotlib.patches as patches

def play_by_play(game_id, home_team, away_team):
    pbp_df = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]

    
    return pbp_df

def play_animation():
    return fig

# Cambiarlo entero.
def avg_possessions_over_time(game_id, home_team, away_team, season):
    df = pd.DataFrame()
    for i in range(20):
        # Let's make some attempts to obtain the data, since we might be doing too many requests...
        for attempt in range(5):
            try:
                # Try to get the information for the season, getting only the columns we need
                teams = leaguedashteamstats.LeagueDashTeamStats(
                    season=season, measure_type_detailed_defense="Advanced",
                ).get_data_frames()[0][["TEAM_ID", "TEAM_NAME", "MIN", "POSS"]]
            except:
                # If we get an error we go to sleep for some time
                time.sleep(30)
            else:
                # If everything's OK with the request, we continue
                break

        if len(teams):
            teams["SEASON"] = season
            df = pd.concat([df, teams], axis=0)
        else:
            print("Try again :(")
            break    

    # Get possessions per minute for every row.
    df["POSS_PER_MIN"] = df["POSS"] / df["MIN"]

    # Get the average possessions per minute for the league in each season.
    poss_per_min_series = df.groupby("SEASON")["POSS_PER_MIN"].mean()
    poss_per_min_df = pd.DataFrame(poss_per_min_series)
    # Transform the index

    # Plot
    fig = px.scatter(poss_per_min_df, x=poss_per_min_df.index, y=poss_per_min_df["POSS_PER_MIN"])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.show()
    
    team_colors = pd.read_csv("team_colors.csv")
    team_colors.head()

    df["TEAM_NAME"].nunique(), team_colors["team"].nunique()
    
    df["TEAM_NAME"].unique()

    df.replace({
        "LA Clippers": "Los Angeles Clippers",
        "Charlotte Bobcats": "Charlotte Hornets",
        "New Orleans Hornets": "New Orleans Pelicans",
        "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
        "New Jersey Nets": "Brooklyn Nets",
    }, inplace=True)
    team_colors.replace({"Seattle Supersonics": "Seattle SuperSonics"}, inplace=True)
    print(df["TEAM_NAME"].nunique(), team_colors["team"].nunique())

    df = df.merge(team_colors, left_on="TEAM_NAME", right_on="team")

    points = [
        go.Scatter(
            x=group["SEASON"], y=group["POSS_PER_MIN"], mode="markers", 
            # Getting the marker to look like the team with border color and color.
            marker=dict(opacity=0.9, size=9, line=dict(width=2, color=group["border_color"].iloc[0])),
            marker_color=group["color"].iloc[0], hoverinfo="name+y",
            text=name, name=name, showlegend=True,
        ) for name, group in df.groupby("TEAM_NAME")
    ]
    fig = go.Figure(data=[
        go.Box(x=df["SEASON"], y=df["POSS_PER_MIN"], boxpoints=False, marker_color="lightgrey", line_color="grey", showlegend=False),
    ] + points)
    fig.update_layout(title="Possessions per Minute", height=600)                    
    fig.show()

    return fig

def offensive_ratings(game_id, home_team, away_team):
    return fig

def threee_point_shootings(game_id, player, team):

    return fig

def get_player_shotchartdetail(player_id, season_id):
    nba_players = players.get_players()
    #player_dict = [player for player in nba_players if player['full_name'] == player][0]

    career_stats = playercareerstats.PlayerCareerStats(player_id)
    career_stats_df = career_stats.get_data_frames()[0]

    team_id = career_stats_df[career_stats_df['SEASON_ID'] == season_id]['TEAM_ID']

    shotchartlist = shotchartdetail.ShotChartDetail(team_id=team_id,
                                                    player_id= int(player_id),
                                                    season_type_all_star='Regular Season',
                                                    season_nullable= season_id,
                                                    context_measure_simple="FGA").get_data_frames()
    return shotchartlist[0], shotchartlist[1]

def draw_court(ax=None, color="blue", lw=1, outer_lines=False):

    if ax is None:
        ax = plt.gca()

    # Basketball Hoop
    hoop = Circle((0,0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Backboard
    backboard = Rectangle((-30, -12.5), 60, 0, linewidth=lw, color=color)

    # The paint
    # outer box
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    # inner box
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Free Throw Top Arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)

    # Free Bottom Top Arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)

    # Restricted Zone
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three Point Line
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    # list of court shapes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw, restricted, corner_three_a, corner_three_b, three_arc, center_outer_arc, center_inner_arc]

    #outer_lines=True
    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)


# Crear el mapa de tiros
def shooting_chart(data, title="", color="b", xlim=(-250, 250), ylim=(422.5, -47.5), line_color="blue",
               court_color="white", court_lw=2, outer_lines=False,
               flip_court=False, gridsize=None,
               ax=None, despine=False):

    if ax is None:
        ax = plt.gca()

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    # draws the court using the draw_court()
    draw_court(ax, color=line_color, lw=court_lw, outer_lines=outer_lines)

    # separate color by make or miss
    x_missed = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_X']
    y_missed = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_Y']

    x_made = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_X']
    y_made = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_Y']

    # Plot missed shots
    ax.scatter(x_missed, y_missed, c='r', marker="x", s=300, linewidths=3)
    # Plot made shots
    ax.scatter(x_made, y_made, facecolors='none', edgecolors='g', marker='o', s=100, linewidths=3)

    # Set the spines to match the rest of court lines, makes outer_lines
    # somewhat unnecessary
    for spine in ax.spines:
        ax.spines[spine].set_lw(court_lw)
        ax.spines[spine].set_color(line_color)

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        plt.rcParams['figure.figsize'] = (12, 11)
    return ax

# Función para dibujar media cancha
def draw_half_court(ax=None, color='black', lw=2):
    if ax is None:
        ax = plt.gca()

    # Crear la media cancha
    hoop = patches.Circle((0, 0), radius=0.75, linewidth=lw, color=color, fill=False)
    backboard = patches.Rectangle((-3, -0.75), 6, 0.5, linewidth=lw, color=color)
    paint = patches.Rectangle((-8, -4.75), 16, 19, linewidth=lw, color=color, fill=False)

    # Dibujar las áreas restringidas
    free_throw = patches.Circle((0, 19), radius=6, linewidth=lw, color=color, fill=False)
    free_throw_top = patches.Arc((0, 19), 12, 12, theta1=0, theta2=180, color=color, lw=lw)
    free_throw_bottom = patches.Arc((0, 19), 12, 12, theta1=180, theta2=0, color=color, lw=lw)

    # Dibujar la línea de 3 puntos
    corner_three_left = patches.Rectangle((-22, -4.75), 0, 14, linewidth=lw, color=color)
    corner_three_right = patches.Rectangle((22, -4.75), 0, 14, linewidth=lw, color=color)
    three_arc = patches.Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, color=color, lw=lw)

    # Dibujar la media cancha
    center_circle = patches.Circle((0, 47.5), radius=6, linewidth=lw, color=color, fill=False)

    # Agregar todos los elementos al gráfico
    ax.add_patch(hoop)
    ax.add_patch(backboard)
    ax.add_patch(paint)
    ax.add_patch(free_throw)
    ax.add_patch(free_throw_top)
    ax.add_patch(free_throw_bottom)
    ax.add_patch(corner_three_left)
    ax.add_patch(corner_three_right)
    ax.add_patch(three_arc)
    ax.add_patch(center_circle)

    return ax

def map_period_to_qtr(period):
    quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'OT1', 6: 'OT2', 7: 'OT3', 8: 'OT4'}
    return quarter_map.get(period)

def network_team():
    return fig


def shot_heatmap():

    return fig