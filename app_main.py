import tkinter as tk
from tkinter import messagebox
from app_viewmodel import ViewModel
from visulization_football import passing_sonnars, passses_heatmap, passing_network, pressure_heatmap, team_passes_heatmap, xg, shot_map, xt_momentum
from statsbombpy import sb  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AppVista(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("Predicción de Resultados Deportivos")
        self.geometry("800x600")

        # Crear los widgets del layout principal
        self.create_widgets()

    def create_widgets(self):
        # Lógica de la vista (Interfaz)
        self.label = tk.Label(self, text="Selecciona un deporte")
        self.label.pack()

        self.deporte_var = tk.StringVar(value="Fútbol")
        self.deporte_selector = tk.OptionMenu(self, self.deporte_var, "Fútbol", "Fórmula 1", "Baloncesto", command=self.update_layout)
        self.deporte_selector.pack()

        self.text_area = tk.Text(self, height=10, width=80)
        self.text_area.pack()

        # Frame para contener las opciones de gráficos
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(pady=10)

    def update_layout(self, deporte):
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, f"Has seleccionado: {deporte}\n")

        # Limpiar el frame anterior
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        if deporte == "Fútbol":
            self.mostrar_opciones_futbol()

    def mostrar_opciones_futbol(self):
        # Obtener todos los partidos de statsbombpy
        partidos = sb.matches(competition_id=43, season_id=3)  # Ejemplo: La Liga 2020-2021
        partidos_list = partidos[['match_id', 'home_team', 'away_team']]

        # Crear selector de partidos
        tk.Label(self.graph_frame, text="Selecciona un partido:").pack()

        # Crear un diccionario para asociar el partido con su descripción
        partidos_dict = {f"{row['home_team']} vs {row['away_team']}": row['match_id'] for _, row in partidos_list.iterrows()}
        partidos_nombres = list(partidos_dict.keys())

        # Variable de selección de partido
        seleccion_partido = tk.StringVar()
        seleccion_partido.set(partidos_nombres[0])

        partido_selector = tk.OptionMenu(self.graph_frame, seleccion_partido, *partidos_nombres)
        partido_selector.pack()

        # Selección de gráficos para fútbol
        tk.Label(self.graph_frame, text="Selecciona un gráfico de fútbol:").pack()

        opciones = ["Heatmap de Pases", "Red de Pases", "Sonar de Pases", "Mapa de Tiros", "xG", "Momentum xT", "Heatmap de Pases de Equipo", "Heatmap de Presion"]
        seleccion_grafico = tk.StringVar()
        seleccion_grafico.set(opciones[0])

        grafico_selector = tk.OptionMenu(self.graph_frame, seleccion_grafico, *opciones)
        grafico_selector.pack()

        boton_generar = tk.Button(self.graph_frame, text="Generar Gráfico", command=lambda: self.generar_grafico_futbol(seleccion_partido.get(), seleccion_grafico.get(), partidos_dict))
        boton_generar.pack()

    def generar_grafico_futbol(self, partido_seleccionado, grafico, partidos_dict):
        match_id = partidos_dict[partido_seleccionado]  # Obtener el match_id del partido seleccionado
        home_team, away_team = partido_seleccionado.split(' vs ')
        competition_stage = "La Liga"  # Ejemplo, puede ser dinámico según los datos del partido

        # Llamar a la función adecuada en visualization.py
        if grafico == "Heatmap de Pases":
            fig = passses_heatmap(match_id, home_team, home_team, away_team, competition_stage)
        elif grafico == "Red de Pases":
            fig = passing_network(match_id, home_team, home_team, away_team, competition_stage)
        elif grafico == "Sonar de Pases":
            fig = passing_sonnars(match_id, home_team, home_team, away_team, competition_stage)
        elif grafico == "Mapa de Tiros":
            fig = shot_map(match_id, home_team, away_team, competition_stage)
        elif grafico == "xG":
            fig = xg(match_id, home_team, home_team, away_team, competition_stage)
        elif grafico == "Momentum xT":
            fig = xt_momentum(match_id, home_team, home_team, away_team, competition_stage)

        # Mostrar el gráfico en el layout principal (dentro de graph_frame)
        self.mostrar_grafico(fig)

    def mostrar_grafico(self, fig):
        # Limpiar el frame actual
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Crear canvas para mostrar el gráfico
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    app = AppVista()
    app.mainloop()