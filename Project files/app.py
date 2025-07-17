# import os
# import numpy as np
# from flask import Flask, request, render_template, url_for
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image, ImageOps

# app = Flask(__name__)

# # Load model
# model = load_model('model/blood_cell_model.h5')

# # Class labels
# class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# # Educational data
# cell_info = {
#     'Eosinophil': {
#         'desc': "Involved in allergic responses and parasitic infections.",
#         'features': "Bilobed nucleus, reddish-orange granules.",
#         'function': "Combat parasites and modulate inflammation.",
#         'image': 'eosinophil.jpg'
#     },
#     'Lymphocyte': {
#         'desc': "Key players in the immune response.",
#         'features': "Large round nucleus, thin cytoplasm.",
#         'function': "Produce antibodies and regulate immunity.",
#         'image': 'lymphocyte.jpg'
#     },
#     'Monocyte': {
#         'desc': "Largest white blood cells.",
#         'features': "Kidney-shaped nucleus, abundant cytoplasm.",
#         'function': "Become macrophages and fight infections.",
#         'image': 'monocyte.jpg'
#     },
#     'Neutrophil': {
#         'desc': "Most common type of white blood cell.",
#         'features': "Multilobed nucleus, small granules.",
#         'function': "Engulf and destroy bacteria (phagocytosis).",
#         'image': 'neutrophil.jpg'
#     }
# }
# app.config['VERSION'] = '1.0.1'  # Change this every time you update style.css


# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     if not file:
#         return "No file uploaded", 400


#     upload_path = os.path.join(app.root_path, 'static', file.filename)
#     file.save(upload_path)

#     # For display in HTML
#     image_path = 'static/' + file.filename


#     # Preprocess image
#     img = Image.open(image_path).convert('RGB')
#     img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)
#     confidence = float(np.max(prediction))
#     class_idx = np.argmax(prediction)
#     predicted_class = class_names[class_idx].strip().capitalize()

#     # # Format response
#     # if confidence < 0.75:
#     #     result = f"Low confidence prediction: {predicted_class} ({confidence * 100:.2f}%)"
#     # else:
#     #     result = f"Predicted: {predicted_class} ({confidence * 100:.2f}%)"

#     edu = cell_info.get(predicted_class, {})

#     if confidence < 0.75:
#         result = (
#             f"Low confidence prediction: {predicted_class} ({confidence * 100:.2f}%)\n"
#             f"Description: {edu.get('desc')}\n"
#             f"Features: {edu.get('features')}\n"
#             f"Function: {edu.get('function')}"
#     )
#     else:
#         result = (
#             f"Predicted: {predicted_class} ({confidence * 100:.2f}%)\n"
#             f"Description: {edu.get('desc')}\n"
#             f"Features: {edu.get('features')}\n"
#             f"Function: {edu.get('function')}"
#     )

#     # edu = cell_info.get(predicted_class, {})
#     print("Prediction:", predicted_class)
#     print("Educational Info:", edu)
#     return render_template("index.html",
#                            prediction=result,
#                            image_path=image_path,
#                            edu_info=edu,
#                            cell_name=predicted_class)
    
# if __name__ == '__main__':
#     app.run(debug=True)
import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
from keras.models import load_model

app = Flask(__name__)

# Load model
# model = load_model('model/blood_cell_model.h5')

from keras.layers import TFSMLayer
import tensorflow as tf

# Wrap SavedModel using TFSMLayer
model = tf.keras.Sequential([
    TFSMLayer("model/blood_cell_model_saved", call_endpoint="serve")
])



# Class labels
class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Educational info
cell_info = {
    'Eosinophil': {
        'desc': "Involved in allergic responses and parasitic infections.",
        'features': "Bilobed nucleus, reddish-orange granules.",
        'function': "Combat parasites and modulate inflammation.",
        'image': 'eosinophil.jpg'
    },
    'Lymphocyte': {
        'desc': "Key players in the immune response.",
        'features': "Large round nucleus, thin cytoplasm.",
        'function': "Produce antibodies and regulate immunity.",
        'image': 'lymphocyte.jpg'
    },
    'Monocyte': {
        'desc': "Largest white blood cells.",
        'features': "Kidney-shaped nucleus, abundant cytoplasm.",
        'function': "Become macrophages and fight infections.",
        'image': 'monocyte.jpg'
    },
    'Neutrophil': {
        'desc': "Most common type of white blood cell.",
        'features': "Multilobed nucleus, small granules.",
        'function': "Engulf and destroy bacteria (phagocytosis).",
        'image': 'neutrophil.jpg'
    }
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    cells = list(cell_info.values())
    for idx, name in enumerate(cell_info.keys()):
        cells[idx]['name'] = name
    return render_template("about.html", cells=cells)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Feedback received from {name} ({email}): {message}")
        return render_template("contact.html", success=True, name=name)
    return render_template("contact.html", success=False)

    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded image
    upload_path = os.path.join('static', file.filename)
    file.save(upload_path)
    image_path = 'static/' + file.filename

    # Preprocess image
    img = Image.open(upload_path).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model(np.array(img_array))
    confidence = float(np.max(prediction))
    class_idx = np.argmax(prediction)
    predicted_class = class_names[class_idx]

    edu = cell_info.get(predicted_class, {})

    if confidence < 0.75:
        result = (
            f"Low confidence prediction: {predicted_class} ({confidence * 100:.2f}%)<br>"
            
        )
    else:
        result = (
            f"<strong>Predicted:</strong> {predicted_class} ({confidence * 100:.2f}%)"
           
        )

    return render_template("index.html",
                           prediction=result,
                           image_path=image_path,
                           edu_info=edu,
                           cell_name=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
