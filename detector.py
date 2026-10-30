#----------------------------------No TOCARRRRR----------------------------------------------
from ultralytics import YOLO
import cv2
import numpy as np
import time

class DetectorSuspension:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
        self.MIN_LAPLACIAN = 100   

        self.estado = "IDLE"
        self.rebotes = 0
        
        # Variables para el GPS y conteo
        self.historial_y = []
        self.t_liberacion = None
        
        self.direccion_actual = 0 
        self.pos_max = 0
        self.pos_min = 9999

    def reset(self):
        self.estado = "IDLE"
        self.rebotes = 0
        self.historial_y = []
        self.t_liberacion = None
        self.direccion_actual = 0
        self.pos_max = 0
        self.pos_min = 9999

    def iniciar(self):
        self.reset()
        self.estado = "MIDIENDO"
        self.t_liberacion = time.time()

    def _detectar_llantas(self, frame):
        results = self.model.predict(frame, conf=0.8, verbose=False)[0] # 0.92
        annotated = frame.copy()
        cajas = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian >= self.MIN_LAPLACIAN:
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"llanta {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cajas.append((x1, y1, x2, y2))
                
        return annotated, cajas

#------------------------------PROBAR ESTO--------------------------------------------------------------------------


    def _actualizar_maquina_estados(self, pos_y_actual, t_actual, hay_llanta):
        if self.estado == "IDLE" or not hay_llanta:
            return

        TIEMPO_MIN_MS = 9000   # tiempo de espera antes de finalizar la medición 

        if self.estado == "MIDIENDO":
            self.historial_y.append(pos_y_actual)
            if len(self.historial_y) > 5:
                self.historial_y.pop(0)

            if len(self.historial_y) >= 3:
                promedio_anterior = sum(self.historial_y[:-1]) / len(self.historial_y[:-1])
                movimiento = pos_y_actual - promedio_anterior

                self.pos_max = max(self.pos_max, pos_y_actual)
                self.pos_min = min(self.pos_min, pos_y_actual)

                UMBRAL = 0.8 # sensibilidad de los RebotesSS 

                if movimiento > UMBRAL: 
                    if self.direccion_actual == -1: 
                        self.rebotes += 1
                    self.direccion_actual = 1
                elif movimiento < -UMBRAL: 
                    if self.direccion_actual == 1: 
                        self.rebotes += 1
                    self.direccion_actual = -1

            if (t_actual - self.t_liberacion) * 1000 > TIEMPO_MIN_MS:
                self.estado = "FIN"

#------------------------------------------------------------------------------------------------------

    def _diagnostico(self):
       
        rango = self.pos_max - self.pos_min
        
        if self.estado == "IDLE": return "En espera", (200, 200, 200)
        if self.estado == "MIDIENDO": return "Midiendo rebotes...", (255, 160, 0)

        if rango <6.0 and self.rebotes == 0: # 1.0  mas estrcito en el movimiento
            return "sin movimiento", (150, 150, 150)

        if self.rebotes <= 1: return "Amortiguador OK", (0, 200, 0)
        elif self.rebotes <= 3: return "Amortiguador: REVISAR", (0, 215, 255)
        else: return "Amortiguador MALO", (0, 0, 255)
            
            
#---------------------------------------------------------------------------------------------------------
    def procesar(self, frame):
        annotated, cajas = self._detectar_llantas(frame)
        hay_llanta = len(cajas) > 0

        centro_y = 0
        if hay_llanta:
            x1, y1, x2, y2 = cajas[0] 
            alto_llanta = y2 - y1
            
            # Recuadro azul (25%)
            subida = int(alto_llanta * 0.25) 
            crop_y1 = max(0, y1 - subida)
            crop_y2 = y1
            
            cv2.rectangle(annotated, (x1, crop_y1), (x2, crop_y2), (255, 0, 0), 2)
            cv2.putText(annotated, "Suspension", (x1, crop_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # El GPS (Punto rojo)
            centro_y = (crop_y1 + crop_y2) // 2
            centro_x = (x1 + x2) // 2
            cv2.circle(annotated, (centro_x, centro_y), 4, (0, 0, 255), -1) 

        t = time.time()
        

        if hay_llanta:
            self._actualizar_maquina_estados(centro_y, t, hay_llanta)

        texto, color = self._diagnostico()
        progreso = 0
        if self.estado == "MIDIENDO" and self.t_liberacion:
            transcurrido = (t - self.t_liberacion) * 1000
            progreso = min(100, (transcurrido / 9000) * 100)
        elif self.estado == "FIN":
            progreso = 100

        return annotated, {
            "estado": self.estado, "rebotes": self.rebotes, "texto": texto, "color": color,
            "rango": float(self.pos_max - self.pos_min) if self.pos_max else 0,
            "llantas": len(cajas), "progreso": round(progreso, 1),
        }