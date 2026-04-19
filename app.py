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
<head>
    <title>Iris Pupil Breathing</title>
    <style>
        body {font-family: Arial, sans-serif; text-align:center; padding:60px; background:#f4f7fc;}
        h1 {color:#000;}
        input[type="file"] {margin:20px;}
        input[type="submit"] {padding:14px 30px; font-size:17px; background:#ff0000; color:white; border:none; border-radius:8px; cursor:pointer;}
    </style>
</head>
<body>
    <h1>Upload Your Iris Image</h1>
    <p>JPG ya PNG image upload karo</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br><br>
        <input type="submit" value="Create Real Pupil Breathing Animation">
    </form>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            output_path = create_real_pupil_breathing(filepath)
            return send_file(output_path, as_attachment=True, download_name="real_pupil_breathing.mp4")
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template_string(HTML)


def create_real_pupil_breathing(input_path):
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Image could not be loaded")

    h, w = img.shape[:2]
    original = img.copy()

    # Better Pupil Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Hough Circle Detection for pupil
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=50, param2=30, minRadius=15, maxRadius=80)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sabse bada circle (pupil) le lo
        x, y, radius = circles[0]
        center = (x, y)
        pupil_radius = int(radius * 0.85)
    else:
        # Fallback
        center = (w//2, h//2)
        pupil_radius = int(min(w, h) * 0.18)

    output_path = input_path.rsplit('.', 1)[0] + "_real_breathing.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    for i in range(220):   # ~8.8 seconds
        frame = original.copy()
        scale = 1 + 0.24 * np.sin(2 * np.pi * i / 55)   # Natural breathing speed

        new_radius = int(pupil_radius * scale)

        # Create mask for pupil area
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, new_radius, 255, -1)

        # Extract pupil region
        pupil_region = cv2.bitwise_and(frame, frame, mask=mask)

        # Resize only pupil region
        resized_pupil = cv2.resize(pupil_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Place back the resized pupil at center
        rh, rw = resized_pupil.shape[:2]
        x1 = max(0, center[0] - rw // 2)
        y1 = max(0, center[1] - rh // 2)
        x2 = min(w, x1 + rw)
        y2 = min(h, y1 + rh)

        frame[y1:y2, x1:x2] = resized_pupil[0:y2-y1, 0:x2-x1]

        out.write(frame)

    out.release()
    return output_path


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
