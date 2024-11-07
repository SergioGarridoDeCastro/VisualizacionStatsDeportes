import seaborn as sns
from matplotlib import pyplot as plt, colormaps
import matplotlib.pyplot as plt
import matplotlib as mpl
import fastf1
import fastf1.plotting
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np





def positions_change_race(season, gp, session_type, telemetry, weather):
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False,
                          color_scheme='fastf1')
    session = fastf1.get_session(season, gp, session_type)
    # Se indica si se muestran datos de telemetria y tiempo
    session.load(telemetry=False, weather=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for driver in session.drivers:
        driver_laps = session.laps.pick_driver(driver)

        abb = driver_laps['Driver'].iloc[0]
        style = fastf1.plotting.get_driver_style(identifier=abb,
                                                style=['color', 'linestyle'],
                                                session=session)

        ax.plot(driver_laps['LapNumber'], driver_laps['Position'],
                label=abb, **style)
        
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.legend(bbox_to_anchor=(1.0, 1.02))
    plt.tight_layout()

    plt.show()

    return fig

def strategies(season, gp, type_session):
    session = fastf1.get_session(season, gp, type_session)
    session.load()
    laps = session.laps
    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    fig, ax = plt.subplots(figsize=(5, 10))

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]

        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars
            compound_color = fastf1.plotting.get_compound_color(row["Compound"],
                                                                session=session)
            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=compound_color,
                edgecolor="black",
                fill=True
            )

            previous_stint_end += row["StintLength"]

    plt.title(f"{season} {gp} Strategies")
    plt.xlabel("Lap Number")
    plt.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig

def team_pace(season, gp, type_session):
    race = fastf1.get_session(season, gp, type_session)
    race.load()
    laps = race.laps.pick_quicklaps()
    transformed_laps = laps.copy()

    transformed_laps.loc[:, "LapTime (s)"] = laps['LapTime'].dt.total_seconds()

    team_order = (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .sort_values()
        .index
    )

    # make a color palette associating team names to hex codes
    team_palette = {team: fastf1.plotting.get_team_color(team, session=race)
                    for team in team_order}

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color="white"),
        boxprops=dict(edgecolor="white"),
        medianprops=dict(color="grey"),
        capprops=dict(color="white"),
    )

    plt.title(f"{season} {gp}")
    plt.grid(visible=False)

    # x-label is redundant
    ax.set(xlabel=None)
    plt.tight_layout()
    plt.show()
    return fig

# Muestra los cambios de marcha a lo largo del circuito. Se muestra la vuelta más rapida en general
def gear_changes_track(season, gp, type_session):
    session = fastf1.get_session(season, gp, type_session)
    session.load()

    lap = session.laps.pick_fastest()
    telemetry = lap. get_telemetry()

    x = np.array(telemetry['X'].values)
    y = np.array(telemetry['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)
    gear = telemetry['nGear'].to_numpy().astype(float)

    cmap = colormaps['Paired']
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.grid(visible=False)

    # x-label is redundant
    ax.set(xlabel=None)
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear",
                    boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))


    plt.show()
    return fig

# Muestra la velocidad a lo largo del circuito. Se muestra la vuelta más rapida en general
def driver_lap_speed(season, gp, type_session):
    session = fastf1.get_session(season, gp, type_session)
    session.load()

    lap = session.laps.pick_fastest() # Se puede cambiar a la vuelta mas rapida de cada piloto
    
    x = lap.telemetry['X']              # values for x-axis
    y = lap.telemetry['Y']              # values for y-axis
    color = lap. telemetry['Speed']

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)

    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(15, 10))
    fig.suptitle(f"{session} {season} - {lap['Driver']} - Speed", size=24, y=0.97)

    # Adjust margins and turn of axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    colormap = mpl.cm.plasma
    # After this, we plot the data itself.
    # Create background track line
    ax.plot(lap.telemetry['X'], lap.telemetry['Y'],
        color='black', linestyle='-', linewidth=16, zorder=0)

    # Crear un ajuste de aspecto igual para respetar las proporciones del circuito
    ax.set_aspect('equal', 'box')

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm,
                        linestyle='-', linewidth=5)

    # Set the values used for colormapping
    lc.set_array(color)

    # Merge all line segments together
    line = ax.add_collection(lc)


    # Finally, we create a color bar as a legend.
    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap,
                                    orientation="horizontal")
    
    # Show the plot
    plt.show()
    return fig

def drives_laptime(season, gp, type_session):
    race = fastf1.get_session(season, gp, type_session)
    race.load()

    # Se muestran los tiempos de los 10 pilotos que han puntuado en la carrera
    point_drivers = race.drivers[:10]
    driver_laps = race.laps.pick_drivers(point_drivers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()

    finish_order = [race.get_driver(i)["Abbreviation"] for i in point_drivers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Seaborn doesn't have proper timedelta support,
    # so we have to convert timedelta to float (in seconds)
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    sns.violinplot(data=driver_laps,
                x="Driver",
                y="LapTime(s)",
                hue="Driver",
                inner=None,
                density_norm="area",
                order=finish_order,
                palette=fastf1.plotting.get_driver_color_mapping(session=race)
                )

    sns.swarmplot(data=driver_laps,
                x="Driver",
                y="LapTime(s)",
                order=finish_order,
                hue="Compound",
                palette=fastf1.plotting.get_compound_mapping(session=race),
                hue_order=["SOFT", "MEDIUM", "HARD"],
                linewidth=0,
                size=4,
                )

    ax.set_xlabel("Driver")
    ax.set_ylabel("Lap Time (s)")
    plt.suptitle(f"{season} {gp} Lap Time Distributions")
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()

    return fig

def track_with_corners(season, gp, type_session):
    race = fastf1.get_session(season, gp, type_session)
    race.load()

    lap = race.laps.pick_fastest()
    pos = lap.get_pos_data()
    circuit_info = race.get_circuit_info()

    # Get an array of shape [n, 2] where n is the number of points and the second
    # axis is x and y.
    track = pos.loc[:, ('X', 'Y')].to_numpy()

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate and plot the track map.
    rotated_track = rotate(track, angle=track_angle)
    fig, ax = plt.subplots()
    plt.plot(rotated_track[:, 0], rotated_track[:, 1])

    offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Draw a circle next to the track.
        plt.scatter(text_x, text_y, color='grey', s=140)

        # Draw a line from the track to this circle.
        plt.plot([track_x, text_x], [track_y, text_y], color='grey')

        # Finally, print the corner number inside the circle.
        plt.text(text_x, text_y, txt,
                va='center_baseline', ha='center', size='small', color='white')
        

    plt.title(race.event['Location'])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.show()
    return fig

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

# Funcion para realizar una animacion de la clasificación del mundial
def animate_classification(before, after):
    fig, ax = plt.subplots(figsize=(10, 6))

    drivers = before['DriverNumber']
    y_pos = np.arange(len(drivers))
    
    def update(i):
        ax.clear()
        ax.barh(y_pos, before['Points'], color='blue', label='Antes de la carrera')
        if i > 0:
            ax.barh(y_pos, after['Points'], color='green', alpha=i/10, label='Después de la carrera')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(drivers)
        ax.set_xlabel('Puntos')
        ax.legend()
        ax.set_title('Clasificación del Campeonato de F1')

    ani = FuncAnimation(fig, update, frames=10, repeat=False)
    return ani