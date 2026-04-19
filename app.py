import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, messagebox

def create_real_pupil_breathing(input_image_path, output_path="real_pupil_breathing.mp4", duration=8, fps=30):
    img = cv2.imread(input_image_path)
    if img is None:
        print("Image load failed!")
        return False

    h, w = img.shape[:2]
    original = img.copy()

    # Grayscale aur blur for better pupil detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Threshold to find dark pupil area
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Pupil not detected, using center as fallback")
        center = (w//2, h//2)
        radius = int(min(w, h) * 0.18)
    else:
        # Largest contour as pupil
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        center = (int(x), int(y))
        radius = int(radius * 0.85)

    print(f"Pupil detected at center: {center}, radius: {radius}")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = int(duration * fps)

    for i in range(total_frames):
        frame = original.copy()

        # Breathing scale (slow and natural)
        scale = 1 + 0.22 * np.sin(2 * np.pi * i / (fps * 2.2))

        new_radius = int(radius * scale)

        # Create mask for pupil area
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, new_radius, 255, -1)

        # Get the pupil region
        pupil_region = cv2.bitwise_and(frame, frame, mask=mask)

        # Resize pupil region
        pupil_resized = cv2.resize(pupil_region, (0,0), fx=scale, fy=scale)

        # Create new frame with resized pupil
        new_frame = frame.copy()

        # Calculate new position to keep center fixed
        new_h, new_w = pupil_resized.shape[:2]
        x1 = center[0] - new_w // 2
        y1 = center[1] - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Place resized pupil back
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            # If out of bounds, skip this frame or clip
            pass
        else:
            new_frame[y1:y2, x1:x2] = pupil_resized

        out.write(new_frame)

    out.release()
    print(f"✅ Video saved: {output_path}")
    return True


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=== Real Pupil Breathing Animation ===\n")
    
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Iris JPG Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        messagebox.showinfo("Cancelled", "No file selected.")
    else:
        output_file = "real_pupil_" + os.path.basename(file_path).rsplit('.', 1)[0] + ".mp4"
        success = create_real_pupil_breathing(file_path, output_file, duration=8, fps=25)
        
        if success:
            messagebox.showinfo("Success", f"Animation created!\nFile: {output_file}")
