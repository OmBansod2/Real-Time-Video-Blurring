import cv2
import re
import numpy as np
import easyocr
import torch
import csv
import os
import gdown

# === Step 0: Download YOLO Weights from Google Drive if missing ===
def download_weights():
    file_id = "1Hd3fh6jLHMKu7ysqN9C001n8-XpxiDAW"  # Replace with your Google Drive file ID
    weights_path = "yolov3_training_2000.weights"
    download_url = f"https://drive.google.com/drive/folders/1Hd3fh6jLHMKu7ysqN9C001n8-XpxiDAW?usp=drive_link"
    if not os.path.exists(weights_path):
        print("Downloading YOLO weights from Google Drive...")
        gdown.download(download_url, weights_path, quiet=False)
    else:
        print("YOLO weights already exist.")

download_weights()

# === Load Malicious Words from CSV ===
def load_malicious_words_from_csv(filepath):
    words = []
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    words.append(row[0].strip())
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
    return words

malicious_words = load_malicious_words_from_csv("Hindi.csv")

# === Sensitive Patterns (e.g., Credit Card Numbers) ===
sensitive_patterns = [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b']

# === Initialize EasyOCR Reader ===
reader = easyocr.Reader(['en', 'hi'], gpu=True)

# === Load YOLO Model for Weapon Detection ===
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# === Function to Blur Region ===
def blur_region(image, top_left, bottom_right):
    sub_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    blurred_sub_img = cv2.GaussianBlur(sub_img, (51, 51), 0)
    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_sub_img

# === Function to Process Sensitive/Malicious Text ===
def process_sensitive_text(frame, result):
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Expand the bounding box slightly
        expansion = 20
        top_left = (max(0, top_left[0] - expansion), max(0, top_left[1] - expansion))
        bottom_right = (min(frame.shape[1], bottom_right[0] + expansion),
                        min(frame.shape[0], bottom_right[1] + expansion))

        # Check for sensitive or malicious text
        blur_text = any(re.search(pattern, text) for pattern in sensitive_patterns) or \
                    any(word.upper() in text.upper() for word in malicious_words)

        if blur_text:
            blur_region(frame, top_left, bottom_right)

# === Function to Detect and Blur Weapons ===
def detect_weapons(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            top_left = (max(0, x), max(0, y))
            bottom_right = (min(x + w, width), min(y + h, height))
            blur_region(frame, top_left, bottom_right)

# === Start Video Capture ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = reader.readtext(frame)
    process_sensitive_text(frame, result)
    detect_weapons(frame)

    cv2.imshow("Webcam Stream (Processed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
