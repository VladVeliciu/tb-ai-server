import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Load the model
interpreter = tf.lite.Interpreter(model_path="model_hybrid.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ThingsBoard credentials
TB_TOKEN = "0mxqRJLizu154m1ZfAwY"
TB_URL = "https://demo.thingsboard.io"

@app.route("/", methods=["POST"])
def infer():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    features = np.array(data["features"], dtype=np.float32).reshape(1,-1)
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    result = (float)(np.argmax(output))

    # Send result to ThingsBoard
    try:
        requests.post(
            f"{TB_URL}/api/v1/{TB_TOKEN}/attributes",
            json={"Classification Result": result},
            timeout=3
        )
    except Exception as e:
        print("Failed to send to ThingsBoard:", e)

    return jsonify({"Classification Result": result})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)

