

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = FastAPI()


origin = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MODEL=tf.keras.models.load_model("./1")
CLASS_NAMES = ["Banana_G1", "Banana_G2", "Rotten"]

@app.get("/")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image=read_file_as_image(await file.read())
    shape=image.shape
    img_batch=np.expand_dims(image, 0)
    # resize image to (256,256,3)
    img_batch=tf.image.resize(img_batch,(256,256))
    prediction=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    if predicted_class=="Banana_G2":
        predicted_class="Green Banana- not ripen"
    elif predicted_class=="Banana_G1":
        predicted_class="Mature Banana -ripen"
    else:
        predicted_class="Rotten Banana"
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }



