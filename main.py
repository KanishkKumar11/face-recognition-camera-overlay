import cv2
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise SystemExit("Error: Could not open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
imageBackground = cv2.imread("Resources/IMG_0300.jpeg")
if imageBackground is None:
    raise SystemExit("Error: Could not load background image")
REGION_W, REGION_H = 2472, 2034
REGION_X = 50
bg_h, bg_w = imageBackground.shape[:2]
REGION_Y = (bg_h - REGION_H) // 2
pad = int(0.03 * min(REGION_W, REGION_H))
avail_w = REGION_W - 2 * pad
avail_h = REGION_H - 2 * pad
hsv = cv2.cvtColor(imageBackground, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 40, 30])
upper_blue = np.array([140, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
face_bbox = None
if contours:
    largest = max(contours, key=cv2.contourArea)
    x_b, y_b, w_b, h_b = cv2.boundingRect(largest)
    face_bbox = (x_b, y_b, w_b, h_b)
else:
    face_bbox = (REGION_X + REGION_W // 2, REGION_Y, REGION_W // 2, REGION_H)
while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab frame")
        break
    img = cv2.flip(img, 1)
    target_w = int(avail_w * 0.8)
    target_h = int(avail_h * 0.9)
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    paste_x = REGION_X + pad
    offset_y = 100
    paste_y = REGION_Y + (avail_h - target_h) // 2 + offset_y
    canvas = imageBackground.copy()
    canvas[paste_y:paste_y + target_h, paste_x:paste_x + target_w] = img_resized
    cv2.imshow("Face Recognition", img_resized)
    cv2.imshow("Background", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break
cap.release()
cv2.destroyAllWindows()