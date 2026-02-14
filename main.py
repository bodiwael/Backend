from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI(title="Facial Emotion API")

# Allow your frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = tf.keras.models.load_model("model.h5")
emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Preprocess to model input
    img = cv2.resize(img, (48,48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0,-1))  # shape (1,48,48,1)

    # Predict
    pred = model.predict(img)
    emotion_idx = int(np.argmax(pred, axis=-1)[0])
    emotion = emotions[emotion_idx]

    return {"emotion": emotion, "confidence": float(np.max(pred))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
