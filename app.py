
from flask import Flask, render_template, Response, jsonify
from waitress import serve
import cv2
import threading
import time
from detector import DetectorSuspension

app = Flask(__name__)
detector = DetectorSuspension(model_path="best.pt")

lock = threading.Lock()
ultimo_frame_jpg = None
estado_actual = {
    "estado": "IDLE",
    "rebotes": 0,
    "texto": "En espera",
    "rango": 0,
    "llantas": 0,
    "progreso": 0,
}
hilo_activo = True


def loop_captura():
    global ultimo_frame_jpg, estado_actual, hilo_activo

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara")
        return

   
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while hilo_activo:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        annotated, info = detector.procesar(frame)

        ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        with lock:
            estado_actual = info
            if ok_enc:
                ultimo_frame_jpg = buf.tobytes()

        time.sleep(0.001)

    cap.release()


hilo = threading.Thread(target=loop_captura, daemon=True)
hilo.start()


def generar_frames():
    global ultimo_frame_jpg
    while True:
        with lock:
            jpg = ultimo_frame_jpg
        if jpg is None:
            time.sleep(0.02)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generar_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/estado")
def estado():
    with lock:
        return jsonify(estado_actual)


@app.route("/iniciar", methods=["POST"])
def iniciar():
    detector.iniciar()
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset():
    detector.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  Servidor de Diagnóstico de Suspensión")
    print("  http://127.0.0.1:5001")
    print("=" * 50)
    serve(app, host="0.0.0.0", port=5001, threads=12)