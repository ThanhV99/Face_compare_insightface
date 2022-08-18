from flask import Flask, render_template, request, redirect, url_for, jsonify
import io
import base64
import numpy as np
import cv2
from source.detect import InsightFace
from PIL import Image

insightface = InsightFace()
app = Flask(__name__)

ALLOWED_EXTENTIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        image_face_1 = ''
        image_face_2 = ''
        face1 = request.files['face1']
        if face1 and allowed_file(filename=face1.filename):
            buffer = io.BytesIO(face1.read())
            arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            image_face_1 = cv2.imdecode(arr, -1)
        
        face2 = request.files['face2']
        if face2 and allowed_file(filename=face2.filename):
            buffer = io.BytesIO(face2.read())
            arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            image_face_2 = cv2.imdecode(arr, -1)
        
        if len(image_face_1) != 0 and len(image_face_2) != 0:
            status, distance = insightface.compare_two_face(image_face_1, image_face_2)
            if status == 'Success':
                if distance > 0.35:
                    result = "It is one person"
                else:
                    result = "It is two person"
                buffered = io.BytesIO()
                im_base64 = Image.fromarray(cv2.cvtColor(image_face_1, cv2.COLOR_BGR2RGB))
                im_base64.save(buffered, format="PNG")
                images = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return render_template('home.html', result=result, images=images)
            else:
                return render_template('home.html', result=status)
        return redirect(url_for('index'))
    
@app.route('/api-detect', methods=['POST'])
def api_detect():
    if request.method == 'POST':
        image_face_1 = ''
        image_face_2 = ''
        face1 = request.files['face1']
        if face1 and allowed_file(filename=face1.filename):
            buffer = io.BytesIO(face1.read())
            arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            image_face_1 = cv2.imdecode(arr, -1)
        
        face2 = request.files['face2']
        if face2 and allowed_file(filename=face2.filename):
            buffer = io.BytesIO(face2.read())
            arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            image_face_2 = cv2.imdecode(arr, -1)
        
        if len(image_face_1) != 0 and len(image_face_2) != 0:
            status, distance = insightface.compare_two_face(image_face_1, image_face_2)
            result = ''
            if status == 'Success':
                if distance > 0.35:
                    result = "It is one person"
                else:
                    result = "It is two person"
                content = {'error message':0,
                           'data':result}
                return jsonify(content)
            else:
                return jsonify({'error message':'Image must only 1 face', 'data':result})
        return jsonify('Method not allowed!')

if __name__ == "__main__":
    app.run(port=5000, debug=True)