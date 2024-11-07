# viewmodel.py


class ViewModel:
    def __init__(self, deporte):
        self.deporte = deporte

    def obtener_datos_y_prediccion(self):
        if self.deporte == "Fútbol":
            datos = obtener_datos_futbol()
        elif self.deporte == "Fórmula 1":
            datos = obtener_datos_f1()
        else:
            datos = obtener_datos_baloncesto()
        
        # Predicción del resultado
        prediccion = predecir_resultado(self.deporte, datos)
        
        return datos, prediccion
