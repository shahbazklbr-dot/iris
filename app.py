from flask import Flask, request, send_file, render_template_string
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple HTML Upload Page
HTML = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Pupil Breathing</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f4f7fc; }
        h1 { color: #000; }
        input[type="file"] { margin: 20px; }
        input[type="submit"] { padding: 12px 30px; font-size: 16px; background: #ff0000; color: white; border: none; border-radius: 8px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Upload Iris Image</h1>
    <p>Select your iris JPG/PNG image</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br><br>
        <input type="submit" value="Create Pupil Breathing Animation">
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
            return send_file(output_path, as_attachment=True, download_name="iris_pupil_breathing.mp4")
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template_string(HTML)


def create_real_pupil_breathing(input_path):
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Could not read image")

    h, w = img.shape[:2]
    original = img.copy()

    # Pupil Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        center = (int(x), int(y))
        pupil_radius = int(radius * 0.82)
    else:
        center = (w//2, h//2)
        pupil_radius = int(min(w, h) * 0.18)

    # Video Output
    output_path = input_path.rsplit('.', 1)[0] + "_breathing.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    for i in range(200):   # ~8 seconds
        frame = original.copy()
        scale = 1 + 0.20 * np.sin(2 * np.pi * i / 50)   # Natural slow breathing

        new_radius = int(pupil_radius * scale)

        # Create mask for pupil area
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, new_radius, 255, -1)

        # Extract and resize pupil region
        pupil_area = cv2.bitwise_and(frame, frame, mask=mask)
        resized_pupil = cv2.resize(pupil_area, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Place back the resized pupil
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
