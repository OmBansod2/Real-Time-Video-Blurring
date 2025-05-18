import cv2
import re
import numpy as np
import easyocr
import torch

# Sensitive patterns (e.g., credit card numbers)
sensitive_patterns = [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b']

# Malicious words to detect
malicious_words = ["STARTUP", "PLAN", "Hello", "bye"]

# Initialize OCR Reader
reader = easyocr.Reader(['en', 'hi'], gpu=True)

# Initialize YOLO for weapon detection
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to blur sensitive regions
def blur_region(image, top_left, bottom_right):
    sub_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    blurred_sub_img = cv2.GaussianBlur(sub_img, (51, 51), 0)
    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_sub_img

# Function to process sensitive/malicious text
# Function to process sensitive/malicious text
def process_sensitive_text(frame, result):
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Increase the blur area by expanding the bounding box
        expansion_factor = 20  # This controls how much bigger the blur box will be
        top_left = (top_left[0] - expansion_factor, top_left[1] - expansion_factor)
        bottom_right = (bottom_right[0] + expansion_factor, bottom_right[1] + expansion_factor)

        # Ensure the coordinates don't go out of frame bounds
        top_left = (max(0, top_left[0]), max(0, top_left[1]))
        bottom_right = (min(frame.shape[1], bottom_right[0]), min(frame.shape[0], bottom_right[1]))

        # Check for sensitive or malicious text
        blur_text = any(re.search(pattern, text) for pattern in sensitive_patterns) or \
                    any(word.upper() in text.upper() for word in malicious_words)

        if blur_text:
            blur_region(frame, top_left, bottom_right)



# Function to detect weapons
def detect_weapons(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

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
            # Blur the weapon region
            top_left = (max(0, x), max(0, y))
            bottom_right = (min(x + w, width), min(y + h, height))
            blur_region(frame, top_left, bottom_right)


# Start video capture from laptop camera
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

