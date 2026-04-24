import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

BEST_THRESHOLDS_JSON = Path("outputs/results/best_thresholds.json")


def load_best_thresholds():
    try:
        data = json.loads(BEST_THRESHOLDS_JSON.read_text())
        baseline = data.get("baseline", {})
        return float(baseline["conf"]), float(baseline["iou"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.25, 0.45


def get_weights_path() -> Path:
    candidates = [
        Path("outputs/checkpoints/baseline/weights/best.pt"),
        Path("outputs/checkpoints/baseline2/weights/best.pt"),
        Path("outputs/checkpoints/baseline3/weights/best.pt"),
        Path("outputs/checkpoints/baseline4/weights/best.pt"),
        Path("outputs/checkpoints/baseline5/weights/best.pt"),
        Path("outputs/checkpoints/improved/weights/best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No trained weights found. Run `make train-baseline` first."
    )


app = Flask(__name__)

DEFAULT_CONF, DEFAULT_IOU = load_best_thresholds()
print(f"Loaded thresholds: conf={DEFAULT_CONF}, iou={DEFAULT_IOU}")

try:
    _weights_path = get_weights_path()
    model = YOLO(str(_weights_path))
    print(f"Loaded model from: {_weights_path}")
except FileNotFoundError as exc:
    model = None
    print(f"Warning: {exc}")


@app.route("/")
def index():
    return render_template("index.html", default_conf=DEFAULT_CONF)


@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model weights not found. Run make train-baseline first."}), 500

    files = request.files.getlist("images[]")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images provided"}), 400

    conf = float(request.form.get("conf", DEFAULT_CONF))
    results_out = []

    for f in files[:3]:
        img_bytes = f.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        _, orig_buf = cv2.imencode(".png", img_bgr)
        original_b64 = base64.b64encode(orig_buf).decode("utf-8")

        results = model.predict(
            source=img_bgr,
            conf=conf,
            iou=DEFAULT_IOU,
            save=False,
            verbose=False,
        )

        annotated = results[0].plot()
        _, ann_buf = cv2.imencode(".png", annotated)
        annotated_b64 = base64.b64encode(ann_buf).decode("utf-8")

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            names = results[0].names
            detections = [
                f"{names[int(cls)]} {float(score):.2f}"
                for cls, score in zip(boxes.cls, boxes.conf)
            ]
        else:
            detections = []

        results_out.append({
            "filename": f.filename,
            "original": original_b64,
            "annotated": annotated_b64,
            "detections": detections,
        })

    return jsonify(results_out)


if __name__ == "__main__":
    app.run(debug=True)
