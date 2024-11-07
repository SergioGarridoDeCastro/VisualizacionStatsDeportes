from statsbombpy import sb
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as pat

from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np

from mplsoccer import Pitch, VerticalPitch, Sbopen
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

#from auxiliary import country_colors, annotation_fix_dict

def passes_heatmap(matchid, home_team, away_team, competition_stage):
    #open the data
    parser = Sbopen()
    df_match = parser.event(match_id=matchid)
    #get list of games by our team, either home or away
    match_ids = df_match.loc[(df_match["home_team_name"] == home_team) | (df_match["away_team_name"] == away_team)]["match_id"].tolist()
    #calculate number of games
    no_games = len(match_ids)   

    #declare an empty dataframe
    danger_passes = pd.DataFrame()
    for idx in match_ids:
        #open the event data from this game
        df = parser.event(idx)[0]
        for period in [1, 2]:
            #keep only accurate passes by England that were not set pieces in this period
            mask_pass = (df.team_name == home_team) & (df.type_name == "Pass") & (df.outcome_name.isnull()) & (df.period == period) & (df.sub_type_name.isnull())
            #keep only necessary columns
            passes = df.loc[mask_pass, ["x", "y", "end_x", "end_y", "minute", "second", "player_name"]]
            #keep only Shots by England in this period
            mask_shot = (df.team_name == home_team) & (df.type_name == "Shot") & (df.period == period)
            #keep only necessary columns
            shots = df.loc[mask_shot, ["minute", "second"]]
            #convert time to seconds
            shot_times = shots['minute']*60+shots['second']
            shot_window = 15
            #find starts of the window
            shot_start = shot_times - shot_window
            #condition to avoid negative shot starts
            shot_start = shot_start.apply(lambda i: i if i>0 else (period-1)*45)
            #convert to seconds
            pass_times = passes['minute']*60+passes['second']
            #check if pass is in any of the windows for this half
            pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

            #keep only danger passes
            danger_passes_period = passes.loc[pass_to_shot]
            #concatenate dataframe with a previous one to keep danger passes from the whole tournament
            danger_passes = pd.concat([danger_passes, danger_passes_period], ignore_index = True)    

    #plot pitch
    pitch = Pitch(line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #scatter the location on the pitch
    pitch.scatter(danger_passes.x, danger_passes.y, s=100, color='blue', edgecolors='grey', linewidth=1, alpha=0.2, ax=ax["pitch"])
    #uncomment it to plot arrows
    #pitch.arrows(danger_passes.x, danger_passes.y, danger_passes.end_x, danger_passes.end_y, color = "blue", ax=ax['pitch'])
    #add title
    fig.suptitle('Location of danger passes by ' + team, fontsize = 30)
    plt.show()         

    #plot vertical pitch
    pitch = Pitch(line_zorder=2, line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #get the 2D histogram
    bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y, statistic='count', bins=(6, 5), normalize=False)
    #normalize by number of games
    bin_statistic["statistic"] = bin_statistic["statistic"]/no_games
    #make a heatmap
    pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
    #legend to our plot
    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    fig.suptitle('Danger passes by ' + team + " per game", fontsize = 30)
    plt.show()
    return fig

def passes_map(match_id, player, home_team, away_team, competition_stage):
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(match_id=match_id)
    passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')

    #drawing pitch
    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))

    for i,thepass in passes.iterrows():
        #if pass made by Lamine Yamal
        if thepass['player_name']== player:
            x=thepass['x']
            y=thepass['y']
            #plot circle
            passCircle=plt.Circle((x,y),2,color="blue")
            passCircle.set_alpha(.2)
            ax.add_patch(passCircle)
            dx=thepass['end_x']-x
            dy=thepass['end_y']-y
            #plot arrow
            passArrow=plt.Arrow(x,y,dx,dy,width=3,color="blue")
            ax.add_patch(passArrow)

    ax.set_title(f"{player} passes against {away_team}", fontsize = 24)
    fig.set_size_inches(10, 7)
    plt.show()
    return fig

def shot_map(match_id, home_team, away_team, competition_stage):
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(match_id=match_id)
    #get team names
    team1, team2 = df.team_name.unique()
    #A dataframe of shots
    shots = df.loc[df['type_name'] == 'Shot'].set_index('id')

    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))
    #Size of the pitch in yards (!!!)
    pitchLengthX = 120
    pitchWidthY = 80

    goals_team1 = []
    goals_team2 = []

    #Plot the shots by looping through them.
    for i,shot in shots.iterrows():
        #get the information
        x=shot['x']
        y=shot['y']
        goal=shot['outcome_name']=='Goal'
        team_name=shot['team_name']
        #set circlesize
        circleSize=2
        #plot England
        if (team_name==team1):
            if goal:
                shotCircle=plt.Circle((x,y),circleSize,color="red")
                plt.text(x+1,y-2,shot['player_name'])
                goals_team1.append(shot['player_name'])
            else:
                shotCircle=plt.Circle((x,y),circleSize,color="red")
                shotCircle.set_alpha(.2)
        #plot Sweden
        else:
            if goal:
                shotCircle=plt.Circle((pitchLengthX-x,pitchWidthY - y),circleSize,color="blue")
                plt.text(pitchLengthX-x+1,pitchWidthY - y - 2 ,shot['player_name'])
                goals_team2.append(shot['player_name'])
            else:
                shotCircle=plt.Circle((pitchLengthX-x,pitchWidthY - y),circleSize,color="blue")
                shotCircle.set_alpha(.2)
        ax.add_patch(shotCircle)
    #set title
    fig.suptitle(f"{home_team} (red) and {away_team} (blue) shots", fontsize = 24)
    # Subtítulo para los jugadores que marcaron gol

    goals_text = f"Goleadores {home_team}: {', '.join(goals_team1)}\nGoleadores {away_team}: {', '.join(goals_team2)}"
    fig.text(0.5, -0.05, goals_text, ha='center', fontsize=16, color='black')
    fig.set_size_inches(10, 7)
    plt.show()
    return fig

def passing_network(match_id, home_team, away_team, competition_stage):
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(match_id=match_id)

    #check for index of first sub
    sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == home_team].iloc[0]["index"]
    #make df with successfull passes by England until the first substitution
    mask_england = (df.type_name == 'Pass') & (df.team_name == home_team) & (df.index < sub) & (df.outcome_name.isnull()) & (df.sub_type_name != "Throw-in")
    #taking necessary columns
    df_pass = df.loc[mask_england, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
    #adjusting that only the surname of a player is presented.
    df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
    df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])    

    scatter_df = pd.DataFrame()
    for i, name in enumerate(df_pass["player_name"].unique()):
        passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
        recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
        recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player_name"] = name
        #make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        #calculate number of passes
        scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

    #adjust the size of a circle so that the player who made more passes
    scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)   
    
    #counting passes between players
    df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
    #setting a treshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df['pass_count']>2]     

    #Drawing pitch
    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #Scatter the location on the pitch
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
    #annotating player name
    for i, row in scatter_df.iterrows():
        pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

    fig.suptitle("Nodes location - {home_team}", fontsize = 30)

    #plot once again pitch and vertices
    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
    for i, row in scatter_df.iterrows():
        pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

    for i, row in lines_df.iterrows():
            player1 = row["pair_key"].split("_")[0]
            player2 = row['pair_key'].split("_")[1]
            #take the average location of players to plot a line between them
            player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
            player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
            player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
            player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
            num_passes = row["pass_count"]
            #adjust the line width so that the more passes, the wider the line
            line_width = (num_passes / lines_df['pass_count'].max() * 10)
            #plot lines on the pitch
            pitch.lines(player1_x, player1_y, player2_x, player2_y,
                            alpha=1, lw=line_width, zorder=2, color="red", ax = ax["pitch"])

    fig.suptitle(f"{home_team} Passing Network against {away_team}", fontsize = 30)
    plt.show()        
    return fig 

def xg_flow(match_id, home_team, away_team, competition_stage):
        shots = sb.events(match_id=match_id, split=True, flatten_attrs=False)["shots"]
        shots.index = range(len(shots))

        shots["xG"] = [shots["shot"][i]["statsbomb_xg"] for i in range(len(shots))]
        shots["outcome"] = [shots["shot"][i]["outcome"]["name"] for i in range(len(shots))]
        shots = shots[["minute", "second", "team", "player", "xG", "outcome"]]

        # a - away, h - home
        a_xG = [0]
        h_xG= [0]
        a_min = [0]
        h_min = [0]

        for i in range(len(shots)):
                if shots['team'][i]==away_team:
                        a_xG.append(shots['xG'][i])
                        a_min.append(shots['minute'][i])
                if shots['team'][i]==home_team:
                        h_xG.append(shots['xG'][i])
                        h_min.append(shots['minute'][i])
                
        def cumsum(the_list):
                return [sum(the_list[:i+1]) for i in range(len(the_list))]
        
        a_xG = cumsum(a_xG)
        h_xG = cumsum(h_xG)

        # make the plot finish at the end of an axis for both teams
        if(a_min[-1]>h_min[-1]):
                h_min.append(a_min[-1])
                h_xG.append(h_xG[-1])
        elif (h_min[-1]>a_min[-1]):
                a_min.append(h_min[-1])
                a_xG.append(a_xG[-1])

        a_xG_total = round(a_xG[-1], 2)
        h_xG_total = round(h_xG[-1], 2)

        mpl.rcParams['xtick.color'] = 'white'
        mpl.rcParams['ytick.color'] = 'white'

        fig, ax = plt.subplots(figsize = (10,5))
        fig.set_facecolor('#0e1117')
        ax.patch.set_facecolor('#0e1117')

        ax.step(x=a_min, y=a_xG, color='red', where='post', linewidth=4)
        ax.step(x=h_min, y=h_xG, color='white', where='post', linewidth=4)
        plt.xticks([0,15,30,45,60,75,90,105,120])
        plt.xlabel('Minute',fontname='Monospace',color='white',fontsize=16)
        plt.ylabel('xG',fontname='Monospace',color='white',fontsize=16)

        ax.grid(ls='dotted',lw=.5,color='lightgrey',axis='y',zorder=1)
        # remove the frame of the plot
        spines = ['top','bottom','left','right']
        for x in spines:
                if x in spines:
                        ax.spines[x].set_visible(False)
                
        ax.margins(x=0)
        h_text = str(h_xG_total)+' '+home_team
        a_text = str(a_xG_total)+' '+away_team
        plt.text(5, h_xG_total+0.05, h_text, fontsize = 10, color='white')
        plt.text(5, a_xG_total+0.05, a_text, fontsize = 10, color='red')

        ax.set_title(f"{home_team} vs {away_team}, World Cup {competition_stage}\nxG Flow",
                fontsize=18, color="w", fontfamily="Monospace", fontweight='bold', pad=-8)
        return fig

#TODO
def passing_sonnars(match_id, home_team, away_team, competition_stage):
    passes = sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"]
    passes = passes[passes['team']==home_team]
    df = passes[['pass', 'player']]
    df.iloc[0]['pass']
    df['angle'] = [df['pass'][i]['angle'] for i in df.index]
    df['length'] = [df['pass'][i]['length'] for i in df.index]

    df['angle_bin'] = pd.cut(
                        df['angle'],
                        bins=np.linspace(-np.pi,np.pi,21),
                        labels=False,
                        include_lowest=True
                    )
    
    # average length
    sonar_df = df.groupby(["player", "angle_bin"], as_index=False)
    sonar_df = sonar_df.agg({"length": "mean"})

    # counting passes for each angle bin
    pass_amt  = df.groupby(['player', 'angle_bin']).size().to_frame(name = 'amount').reset_index()

    # concatenating the data
    sonar_df = pd.concat([sonar_df, pass_amt["amount"]], axis=1)

    # extracting coordinates
    passes["x"], passes["y"] = zip(*passes["location"])

    average_location = passes.groupby('player').agg({'x': ['mean'], 'y': ['mean']})
    average_location.columns = ['x', 'y']

    sonar_df = sonar_df.merge(average_location, left_on="player", right_index=True)    

    lineups = sb.lineups(match_id)[home_team]
    lineups['starter'] = [
                    lineups['positions'][i][0]['start_reason']=='Starting XI'
                    if lineups['positions'][i]!=[]
                    else None
                    for i in range(len(lineups))
                    ]
    lineups = lineups[lineups["starter"]==True]

    startingXI =lineups['player_name'].to_list()

    sonar_df = sonar_df[sonar_df['player'].isin(startingXI)]


    fig ,ax = plt.subplots(figsize=(13, 8),constrained_layout=False, tight_layout=True)
    fig.set_facecolor('#0e1117')
    ax.patch.set_facecolor('#0e1117')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1117', line_color='#c7d5cc')
    pitch.draw(ax=ax)

    for player in startingXI:
            for _, row in sonar_df[sonar_df.player == player].iterrows():
                    degree_left_start = 198

                    color = "gold" if row.amount < 3 else "darkorange" if row.amount < 5 else '#9f1b1e'

                    n_bins = 20
                    degree_left = degree_left_start +(360 / n_bins) * (row.angle_bin)
                    degree_right = degree_left - (360 / n_bins)

                    pass_wedge = pat.Wedge(
                            center=(row.x, row.y),
                            r=row.length*0.16, # scaling the sonar segments
                            theta1=degree_right,
                            theta2=degree_left,
                            facecolor=color,
                            edgecolor="black",
                            alpha=0.6
                    )
                    ax.add_patch(pass_wedge)

    for _, row in average_location.iterrows():
        if row.name in startingXI:

            pitch.annotate(
                xy=(row.x, row.y-4.5),
                c='white',
                va='center',
                ha='center',
                size=9,
                fontweight='bold',
                ax=ax
            )

    ax.set_title(
    f"{home_team} vs {away_team}: {competition_stage}\nPassing Sonars for {home_team} (starting XI)",
    fontsize=18, color="w", fontfamily="Monospace", fontweight='bold', pad=-8
    )

    pitch.annotate(
    text='Sonar length corresponds to average pass length\nSonar color corresponds to pass frequency (dark = more)',
    xy=(0.5, 0.01), xycoords='axes fraction', fontsize=10, color='white', ha='center', va='center', fontfamily="Monospace", ax=ax
    )

    return fig

def xt_momentum(match_id, home_team, away_team, competition_stage):
    df = sb.events(match_id = match_id)
    xT = pd.read_csv("https://raw.githubusercontent.com/AKapich/WorldCup_App/main/app/xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

    def get_xT(df, event_type):
        df = df[df['type']==event_type]

        df['x'], df['y'] = zip(*df['location'])
        df['end_x'], df['end_y'] = zip(*df[f'{event_type.lower()}_end_location'])
        df[f'start_x_bin'] = pd.cut(df['x'], bins=xT_cols, labels=False)
        df[f'start_y_bin'] = pd.cut(df['y'], bins=xT_rows, labels=False)
        df[f'end_x_bin'] = pd.cut(df['end_x'], bins=xT_cols, labels=False)
        df[f'end_y_bin'] = pd.cut(df['end_x'], bins=xT_rows, labels=False)
        df['start_zone_value'] = df[[f'start_x_bin', f'start_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)
        df['end_zone_value'] = df[[f'end_x_bin', f'end_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)
        df['xT'] = df['end_zone_value']-df['start_zone_value']

        return df[['xT', 'minute', 'second', 'team', 'type']]
    

    
    aux_df  = deepcopy(df)
    aux_df = aux_df[aux_df['type']=='Pass']
    aux_df['x'], aux_df['y'] = zip(*aux_df['location'])
    aux_df['end_x'], aux_df['end_y'] = zip(*aux_df[f'pass_end_location'])

    aux_df[f'start_x_bin'] = pd.cut(aux_df['x'], bins=xT_cols, labels=False)
    aux_df[f'start_y_bin'] = pd.cut(aux_df['y'], bins=xT_rows, labels=False)
    aux_df[f'end_x_bin'] = pd.cut(aux_df['end_x'], bins=xT_cols, labels=False)
    aux_df[f'end_y_bin'] = pd.cut(aux_df['end_x'], bins=xT_rows, labels=False)

    aux_df['start_zone_value'] = aux_df[[f'start_x_bin', f'start_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)
    aux_df['end_zone_value'] = aux_df[[f'end_x_bin', f'end_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)

    aux_df['xT'] = aux_df['end_zone_value']-aux_df['start_zone_value']

    xT_data = pd.concat([get_xT(df=df, event_type='Pass'), get_xT(df=df, event_type='Carry'), get_xT(df=df, event_type='Shot')], axis=0)
    xT_data['xT_clipped'] = np.clip(xT_data['xT'], 0, 0.1)

    max_xT_per_minute = xT_data.groupby(['team', 'minute'])['xT_clipped'].max().reset_index()

    minutes = sorted(xT_data['minute'].unique())
    weighted_xT_sum = {
        home_team: [],
        away_team: []
    }
    momentum = []

    window_size = 4
    decay_rate = 0.25

    for current_minute in minutes:
        for team in weighted_xT_sum.keys():

            recent_xT_values = max_xT_per_minute[
                                                (max_xT_per_minute['team'] == team) &
                                                (max_xT_per_minute['minute'] <= current_minute) &
                                                (max_xT_per_minute['minute'] > current_minute - window_size)
                                            ]

            weights = np.exp(-decay_rate * (current_minute - recent_xT_values['minute'].values))
            weighted_sum = np.sum(weights * recent_xT_values['xT_clipped'].values)
            weighted_xT_sum[team].append(weighted_sum)

        momentum.append(weighted_xT_sum[home_team][-1] - weighted_xT_sum[away_team][-1])

    momentum_df = pd.DataFrame({
        'minute': minutes,
        'momentum': momentum
    })


    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.tick_params(axis='x', colors='white')
    ax.margins(x=0)
    ax.set_xticks([0,15,30,45,60,75,90])

    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_ylim(-0.08, 0.08)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    momentum_df['smoothed_momentum'] = gaussian_filter1d(momentum_df['momentum'], sigma=1)
    ax.plot(momentum_df['minute'], momentum_df['smoothed_momentum'], color='white')
    ax.axhline(0, color='white', linestyle='--', linewidth=0.5)

    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] > 0), color='red', alpha=0.5, interpolate=True)
    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] < 0), color='blue', alpha=0.5, interpolate=True)

    scores = df[df['shot_outcome'] == 'Goal'].groupby('team')['shot_outcome'].count().reindex(set(df['team']), fill_value=0)
    ax.set_xlabel('Minute', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
    ax.set_ylabel('Momentum', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
    ax.set_title(f'xT Momentum\n{home_team} {scores[home_team]}-{scores[away_team]} {away_team}', color='white', fontsize=20, fontweight='bold', fontfamily='Monospace', pad=-5)

    home_team_text = ax.text(7, 0.064, home_team, fontsize=12, ha='center', fontfamily="Monospace", fontweight='bold', color='white')
    home_team_text.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='white', boxstyle='round'))
    away_team_text = ax.text(7, -0.064, away_team, fontsize=12, ha='center', fontfamily="Monospace", fontweight='bold', color='white')
    away_team_text.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='white', boxstyle='round'))

    goals = df[df['shot_outcome']=='Goal'][['minute', 'team']]
    for _, row in goals.iterrows():
        ymin, ymax = (0.5, 0.8) if row['team'] == home_team else (0.14, 0.5)
        ax.axvline(row['minute'], color='white', linestyle='--', linewidth=0.8, alpha=0.5, ymin=ymin, ymax=ymax)
        ax.scatter(row['minute'], (1 if row['team'] == home_team else -1)*0.06, color='white', s=100, zorder=10, alpha=0.7)
        ax.text(row['minute']+0.1, (1 if row['team'] == home_team else -1)*0.067, 'Goal', fontsize=10, ha='center', va='center', fontfamily="Monospace", color='white')
    
    return fig

def pressure_heatmap(match_id, home_team, away_team, competition_stage):
    
    return fig

def team_passes_heatmap(match_id, home_team, away_team, competition_stage):
    return fig

def individual_passes_map(match_id, player, home_team, away_team, competition_stage):
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(MATCH_ID)
    passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')

    #drawing pitch
    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))

    for i,thepass in passes.iterrows():
        #if pass made by Lamine Yamal
        if thepass['player_name']=='Lamine Yamal Nasraoui Ebana':
            x=thepass['x']
            y=thepass['y']
            #plot circle
            passCircle=plt.Circle((x,y),2,color="blue")
            passCircle.set_alpha(.2)
            ax.add_patch(passCircle)
            dx=thepass['end_x']-x
            dy=thepass['end_y']-y
            #plot arrow
            passArrow=plt.Arrow(x,y,dx,dy,width=3,color="blue")
            ax.add_patch(passArrow)

    ax.set_title("Lamine Yamal passes against England", fontsize = 24)
    fig.set_size_inches(10, 7)
    plt.show()


def single_convex_hull(match_id, player, home_team, away_team, competition_stage):
        events = sb.events(match_id=match_id)

        # single player
        fig ,ax = plt.subplots(figsize=(13, 8),constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#0e1117')
        ax.patch.set_facecolor('#0e1117')
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1117', line_color='#c7d5cc')
        pitch.draw(ax=ax)

        events = events[pd.isna(events["location"])==False]
        events['x'] = [location[0] for location in events.loc[:,"location"]]
        events['y'] = [location[1] for location in events.loc[:,"location"]]
        player_events = events[events["player"]==player]
        before_filter = player_events

        # eliminate points that lay over 1.5 standard deviations away from the mean coords
        # zscore tells how many standard deviations away the value is from the mean
        player_events = player_events[np.abs(stats.zscore(player_events[['x','y']])) < 1.5]
        # where the zscore is greater than 1.5 values are set to NaN
        player_events = player_events[['x','y']][(pd.isna(player_events['x'])==False)&(pd.isna(player_events['y'])==False)]
        points = player_events[['x','y']].values

        plt.scatter(before_filter.x, before_filter.y, color='white')
        plt.scatter(player_events.x, player_events.y, color='white')
        # create a convex hull
        hull = ConvexHull(player_events[['x','y']])
        for i in hull.simplices:
                plt.plot(points[i, 0], points[i, 1], 'green')
                plt.fill(points[hull.vertices,0], points[hull.vertices,1], c='green', alpha=0.03)

        ax.set_title(f"{home_team} vs {away_team}, World Cup {competition_stage}\nConvex Hull of {player} actions",
                fontsize=18, color="w", fontfamily="Monospace", fontweight='bold', pad=-8)

        return fig


def team_convex_hull(match_id, home_team, away_team, competition_stage):
        events = sb.events(match_id=match_id)
        events = events[events["team"]==home_team]
        players = events[pd.isna(events["player"])==False]["player"].unique()
        starters = players[:11] # first eleven

        events = events[pd.isna(events["location"])==False]
        events['x'] = [location[0] for location in events.loc[:,"location"]]
        events['y'] = [location[1] for location in events.loc[:,"location"]]

        # for every starter
        fig ,ax = plt.subplots(figsize=(13, 8),constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#0e1117')
        ax.patch.set_facecolor('#0e1117')
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1117', line_color='#c7d5cc')
        pitch.draw(ax=ax)

        colours = ['#eb4034', '#ebdb34', '#98eb34', '#34eb77', '#be9cd9', '#5797e6',
                   '#fbddad', '#de34eb', '#eb346b', '#34ebcc', '#dbd5d5']
        colourdict = dict(zip(starters, colours))

        for player in starters:
                tempdf = events[events["player"]==player]
                # threshold of 0.75 sd
                tempdf = tempdf[np.abs(stats.zscore(tempdf[['x','y']])) < 0.75]
                
                tempdf = tempdf[['x','y']][(pd.isna(tempdf['x'])==False)&(pd.isna(tempdf['y'])==False)]
                points = tempdf[['x','y']].values
                
                pitch.annotate(player, xy=(np.mean(tempdf.x), np.mean(tempdf.y)),
                                c=colourdict[player], va='center', ha='center',
                                size=10, fontweight='bold',
                                ax=ax)
        
                try:
                        hull = ConvexHull(tempdf[['x','y']])
                except:
                        pass

                try:
                        for i in hull.simplices:
                                plt.plot(points[i, 0], points[i, 1], colourdict[player])
                                plt.fill(points[hull.vertices,0], points[hull.vertices,1], c=colourdict[player], alpha=0.03)
                except:
                        pass

        ax.set_title(f"{home_team} vs {away_team}\n{home_team}: Convex Hulls of actions",
                fontsize=18, color="w", fontfamily="Monospace", fontweight='bold', pad=-8)
        
        return fig


def progressive_passes(match_id, player, team, home_team, away_team, competition_stage):
        passes = sb.events(match_id=match_id, split=True, flatten_attrs=False)["passes"]
        df = passes[passes.team==team]
        df = df[df.player==player]
        df.index = range(len(df))

        df[['start_x', 'start_y']] = pd.DataFrame(df['location'].tolist(), index=df.index)
        df["end_x"] = [df["pass"][i]['end_location'][0] for i in range(len(df))]
        df["end_y"] = [df["pass"][i]['end_location'][1] for i in range(len(df))]
        df['beginning'] = np.sqrt(np.square(120-df['start_x'])+np.square(80-df['start_y']))
        df['end'] = np.sqrt(np.square(120-df['end_x'])+np.square(80-df['end_y']))
        # according to definiton pass is progressive if it brings the ball closer to the goal by at least 25%
        df['progressive'] = df['end'] < 0.75*df['beginning']

        fig ,ax = plt.subplots(figsize=(13, 8),constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#0e1117')
        ax.patch.set_facecolor('#0e1117')
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1117', line_color='#c7d5cc')
        pitch.draw(ax=ax)

        df = df[df['progressive']==True]
        df.index = range(len(df))
        pitch.lines(xstart=df["start_x"], ystart=df["start_y"], xend=df["end_x"], yend=df["end_y"],
                ax=ax, comet=True, color='red')

        ax.set_title(f"{home_team} vs {away_team}, World Cup {competition_stage}\n{player}: Progressive Passes",
                fontsize=18, color="w", fontfamily="Monospace", fontweight='bold', pad=-8)

        return fig

def voronoi_diagram(match_id, player, team, home_team, away_team, competition_stage):
    #declare mplsoccer parser
    parser = Sbopen()

    #open event dataset
    df_event = parser.event(match_id=match_id)[0]
    #find Bennison goal
    event = df_event.loc[df_event["outcome_name"] == 'Goal'].loc[df_event["player_name"] == player]
    #save it's id
    event_id = event["id"].iloc[0]

    #open 360
    df_frame, df_visible = parser.frame(match_id=match_id)
    #get visible area
    visible_area = np.array(df_visible.loc[df_visible["id"] == event_id]['visible_area'].iloc[0]).reshape(-1, 2)

    pitch  = VerticalPitch(line_color='grey', line_zorder = 1, half = True, pad_bottom=-30, linewidth=5)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #add visible area
    pitch.polygon([visible_area], color=(0, 0, 1, 0.3), ax=ax["pitch"], zorder = 2)
    fig.suptitle(f"Area catched by Statsbomb 360 data - {player}'s goal", fontsize = 45)
    plt.show()    

    #get player position for this event
    player_position = df_frame.loc[df_frame["id"] == event_id]
    #get swedish player position
    teammates = player_position.loc[player_position["teammate"] == True]
    #get swiss player positions
    opponent = player_position.loc[player_position["teammate"] == False]

    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #plot visible area
    pitch.polygon([visible_area], color=(0, 0, 1, 0.3), ax=ax["pitch"], zorder = 2)
    #plot sweden players - yellow
    pitch.scatter(teammates.x, teammates.y, color = 'yellow', edgecolors = 'black', s = 400, ax=ax['pitch'], zorder = 3)
    #plot swiss players - red
    pitch.scatter(opponent.x, opponent.y, color = 'red', edgecolors = 'black', s = 400, ax=ax['pitch'], zorder = 3)
    #add shot
    pitch.lines(event.x, event.y,
                    event.end_x, event.end_y, comet = True, color='green', ax=ax['pitch'], zorder = 1, linestyle = ':', lw = 2)
    fig.suptitle(f"Player position during {player}'s goal", fontsize = 45)
    plt.show()    

    team1, team2 = pitch.voronoi(teammates.x, teammates.y,
                         teammates.teammate)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #plot voronoi diagrams as polygons
    t1 = pitch.polygon(team1, ax = ax["pitch"], color = 'yellow', ec = 'black', lw=3, alpha=0.4, zorder = 2)
    #mark visible area
    visible = pitch.polygon([visible_area], color = 'None', linestyle = "--", ec = "black", ax=ax["pitch"], zorder = 2)
    #plot swedish players
    pitch.scatter(teammates.x, teammates.y, color = 'yellow', edgecolors = 'black', s = 600, ax=ax['pitch'], zorder = 4)
    #plot swiss players
    pitch.scatter(opponent.x, opponent.y, color = 'red', edgecolors = 'black', s = 600, ax=ax['pitch'], zorder = 3)
    #plot shot
    pitch.lines(event.x, event.y,
                    event.end_x, event.end_y, comet = True, color='green', ax=ax['pitch'], zorder = 1, linestyle = ':', lw = 5)
    #limit voronoi diagram to polygon
    for p1 in t1:
        p1.set_clip_path(visible[0])
    fig.suptitle("Voronoi diagram for Sweden (in the visible area) - Hanna Bennison's goal", fontsize = 30)
    plt.show()


    #TODO

    