from flask import Flask, request, jsonify
import pandas as pd
from utillc import *
import os
from dotenv import dotenv_values 
from Script_ETL import load_image

app = Flask(__name__)
config = dotenv_values(".env")
UPLOAD_PATH = config["UPLOAD_PATH"]


info = pd.read_csv("infos_especes.csv",delimiter=";")

@app.route('/image_info/<prediction>')
def get_image_info(prediction):
    # Fetch information based on the prediction
    if (info["Espece"] == prediction).any() :
        json_info = info[info["Espece"] == prediction].to_dict(orient = "records")
        return jsonify(json_info[0])
    else:
        return jsonify({'error': 'Prediction not found'}), 404

# Post data to CSV file
@app.route('/post_data', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Save the uploaded file
    filename = 'uploaded_image.png'
    # A ajouter l'image dans le dossier associé et les data dans le data.csv de dataiku 
    
    img_path = os.path.join(UPLOAD_PATH, filename) 
    file.save(img_path)
    load_image(UPLOAD_PATH)
    return jsonify({'image_url': f'/uploads/{filename}'})

if __name__ == '__main__':
    app.run(debug=True)
