from flask import Flask, request, send_file, render_template_string
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = '''
<!doctype html>
<html>
<head><title>Iris Pupil Animation</title></head>
<body style="font-family:Arial; text-align:center; padding:50px;">
    <h1>Upload Iris Image</h1>
    <p>JPG ya PNG image upload karo</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*"><br><br>
        <input type="submit" value="Upload & Create Breathing Animation" style="padding:10px 20px; font-size:16px;">
    </form>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        output_path = create_breathing_video(filepath)
        
        return send_file(output_path, as_attachment=True, download_name='iris_pupil_breathing.mp4')
    
    return render_template_string(HTML)

def create_breathing_video(input_path):
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Could not load image")

    h, w = img.shape[:2]
    
    # Pupil detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        center = (int(x), int(y))
        pupil_radius = int(radius * 0.78)
    else:
        center = (w//2, h//2)
        pupil_radius = int(min(w, h) * 0.18)

    output_path = input_path.replace('.jpg', '_breathing.mp4').replace('.jpeg', '_breathing.mp4').replace('.png', '_breathing.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    for i in range(240):        # 8 seconds
        frame = img.copy()
        scale = 1 + 0.26 * np.sin(2 * np.pi * i / 55)
        curr_r = int(pupil_radius * scale)

        overlay = frame.copy()
        cv2.circle(overlay, center, curr_r, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.93, frame, 0.07, 0, frame)

        # Glare for realism
        cv2.circle(frame, (center[0]-int(curr_r*0.28), center[1]-int(curr_r*0.28)), 
                   int(curr_r*0.22), (255,255,255), -1)

        out.write(frame)

    out.release()
    return output_path

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
