from flask import Flask, render_template, request
import numpy as np
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from PIL import ImageOps, Image


model = load_model("model_identification.h5")

app = Flask(__name__)


# routes
@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


# routes
@app.route("/plants", methods=["GET", "POST"])
def main1():
    return render_template("indexplants.html")


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]

        img_path = "static/" + img.filename
        img.save(img_path)

        p = areaDetection(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


@app.route("/submit1", methods=["GET", "POST"])
def get_output1():
    if request.method == "POST":
        img = request.files["my_image"]

        img_path = "static/" + img.filename
        img.save(img_path)
        image = Image.open(img_path)
        p = predict_plant(image)

    return render_template("indexplants.html", prediction=p, img_path=img_path)


def areaDetection(img_path):
    img = cv2.imread(img_path)
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([102, 255, 255])
    mask = cv2.inRange(grid_HSV, lower_green, upper_green)
    green_perc = (np.sum(mask) / np.size(mask)) / 255
    green_perc = green_perc * 100
    return str(round(green_perc, 3))


def predict_plant(img_path):
    
    img = ImageOps.fit(img_path, (224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    prediction = np.argmax(model.predict(preprocessed_img), axis=-1)
    index_to_label = {
        0: "Alstonia Scholaris",
        1: "Arjun",
        2: "Chinar",
        3: "Guava",
        4: "Jamun",
        5: "Jatropha",
        6: "Lemon",
        7: "Mango",
        8: "Pomegranate",
        9: "Pongamia Pinnata",
    }
    result = index_to_label[prediction[0]]
    return{result}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
