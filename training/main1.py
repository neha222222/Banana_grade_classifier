
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

MODEL = tf.keras.models.load_model("./1")
CLASS_NAMES = ["Banana_G1", "Banana_G2", "Rotten"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     first_image=image.numpy().astype('uint8')
#     plt.imshow(first_image)

#     # first_image=image_batch[0].numpy().astype('uint8')
#     # first_label=label_batch[0].numpy()
#     # plt.imshow(first_image)
#     img_batch = np.expand_dims(first_image, 0)
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

#     confidence = np.max(predictions[0])
#     print("the banana is : ",predicted_class)
#     print("confidence is : ",confidence )
#     # img_batch = np.expand_dims(image, 0)
    
#     # predictions = MODEL.predict(img_batch)

#     # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     # confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    # first_image=image.numpy().astype('uint8')
    first_image=image.astype('uint8')
    plt.imshow(first_image)

    # first_image=image_batch[0].numpy().astype('uint8')
    # first_label=label_batch[0].numpy()
    # plt.imshow(first_image)
    img_batch = np.expand_dims(first_image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    print("the banana is : ",predicted_class)
    print("confidence is : ",confidence )
    # img_batch = np.expand_dims(image, 0)
    
    # predictions = MODEL.predict(img_batch)

    # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
