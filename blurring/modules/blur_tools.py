# # import cv2
# # import pytesseract
# # import numpy as np
# # import os

# # # Point pytesseract to your tesseract executable
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HP\Downloads\uss\download\tesseract.exe"
# # MODEL_DIR = r"C:\Users\HP\Downloads\uss\models"  # folder where you saved both files
# # configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
# # modelFile = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
# # net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# # def detect_faces(image):
# #     """
# #     Detect faces using OpenCV DNN (preferred) or fallback to Haar Cascade if model files missing.
# #     """
# #     MODEL_DIR = r"C:\Users\HP\Downloads\uss\models"
# #     configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
# #     modelFile = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# #     # ✅ Check if both files exist
# #     if os.path.exists(configFile) and os.path.exists(modelFile):
# #         try:
# #             net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# #             (h, w) = image.shape[:2]
# #             blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
# #                                          (300, 300), (104.0, 177.0, 123.0))
# #             net.setInput(blob)
# #             detections = net.forward()

# #             faces = []
# #             for i in range(0, detections.shape[2]):
# #                 confidence = detections[0, 0, i, 2]
# #                 if confidence > 0.5:
# #                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# #                     (startX, startY, endX, endY) = box.astype("int")
# #                     faces.append((startX, startY, endX - startX, endY - startY))
# #             return faces
# #         except Exception as e:
# #             print(f"[WARN] DNN model failed: {e}")
    
# #     # ⚙️ Fallback — Haar Cascade
# #     print("Using Haar Cascade fallback for face detection.")
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# #     return faces

# # import cv2
# # import numpy as np
# # import pytesseract
# # import os

# # def detect_and_blur_plates(image, debug=False):
# #     """
# #     Detect probable license-plate regions using edge+contour heuristics and blur them.
# #     Returns (image_with_blurred_plates, list_of_plate_boxes).
# #     boxes are (x, y, w, h).
# #     Set debug=True to see intermediate windows for tuning.
# #     """

# #     orig = image.copy()
# #     h, w = image.shape[:2]

# #     # 1) Preprocess: grayscale -> bilateral -> morphological -> edges
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.bilateralFilter(gray, 9, 75, 75)         # Preserve edges, reduce noise
# #     # optional CLAHE (can help contrast small plates)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     gray_clahe = clahe.apply(gray)

# #     # 2) Edge detection
# #     edged = cv2.Canny(gray_clahe, 50, 200)

# #     # 3) Close gaps (morphological)
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
# #     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

# #     if debug:
# #         cv2.imshow("gray_clahe", cv2.resize(gray_clahe, (min(800, w), int(h * min(800/w,1.0)))))
# #         cv2.imshow("edged", cv2.resize(edged, (min(800, w), int(h * min(800/w,1.0)))))
# #         cv2.imshow("closed", cv2.resize(closed, (min(800, w), int(h * min(800/w,1.0)))))

# #     # 4) Find contours and filter by shape/ratio/area
# #     contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]  # top candidates

# #     plate_boxes = []
# #     for cnt in contours:
# #         peri = cv2.arcLength(cnt, True)
# #         approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

# #         # We'll accept approx polygons with 4 vertices or bounding boxes that look rectangular
# #         if len(approx) >= 4:
# #             x, y, cw, ch = cv2.boundingRect(approx)
# #             aspect = cw / float(ch) if ch > 0 else 0
# #             area = cw * ch

# #             # Heuristics: license plates are wider than tall, moderate area
# #             if 2.0 <= aspect <= 6.5 and area > 1500 and cw > 60 and ch > 15:
# #                 # Expand region slightly for safety
# #                 pad_x = int(0.03 * cw)
# #                 pad_y = int(0.12 * ch)
# #                 x1 = max(0, x - pad_x)
# #                 y1 = max(0, y - pad_y)
# #                 x2 = min(w, x + cw + pad_x)
# #                 y2 = min(h, y + ch + pad_y)

# #                 # Avoid duplicates: check overlap with already found boxes
# #                 add = True
# #                 for (ax, ay, aw, ah) in plate_boxes:
# #                     iou_x1 = max(x1, ax)
# #                     iou_y1 = max(y1, ay)
# #                     iou_x2 = min(x2, ax + aw)
# #                     iou_y2 = min(y2, ay + ah)
# #                     if iou_x2 > iou_x1 and iou_y2 > iou_y1:
# #                         # overlap exists -> skip if this one is smaller
# #                         add = False
# #                         break
# #                 if add:
# #                     plate_boxes.append((x1, y1, x2 - x1, y2 - y1))

# #     # 5) If no plates found, try Haar cascade fallback (if available)
# #     if len(plate_boxes) == 0:
# #         cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
# #         if os.path.exists(cascade_path):
# #             plate_cascade = cv2.CascadeClassifier(cascade_path)
# #             gray_for_cascade = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #             found = plate_cascade.detectMultiScale(gray_for_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(30, 10))
# #             for (x, y, cw, ch) in found:
# #                 plate_boxes.append((x, y, cw, ch))

# #     # 6) Optionally validate via simple OCR or brightness check (skip if not desired)
# #     validated_boxes = []
# #     for (x, y, cw, ch) in plate_boxes:
# #         roi = orig[y:y+ch, x:x+cw]
# #         if roi.size == 0:
# #             continue

# #         # Quick heuristic: high-contrast text-like region -> check further
# #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #         mean, stddev = cv2.meanStdDev(gray_roi)
# #         # if stddev small, unlikely text; if too dark/bright, still try
# #         if stddev[0][0] < 10 and stddev[0][0] > 80:
# #             # skip weird cases; but we won't be too strict
# #             pass

# #         # (Optional) Try OCR with tuned config to see if text is present
# #         # comment out if you don't want OCR checks
# #         try:
# #             ocr_cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# #             text = pytesseract.image_to_string(gray_roi, config=ocr_cfg).strip()
# #             # If OCR finds 2+ chars/digits, treat as plate-like
# #             if len(text) >= 2:
# #                 validated_boxes.append((x, y, cw, ch))
# #             else:
# #                 # even if OCR weak, still add — plates sometimes fail OCR due to angle/blur
# #                 validated_boxes.append((x, y, cw, ch))
# #         except Exception:
# #             validated_boxes.append((x, y, cw, ch))

# #     # If validated list empty but plate_boxes existed, keep plate_boxes
# #     final_boxes = validated_boxes if len(validated_boxes) > 0 else plate_boxes

# #     # 7) Blur the final boxes
# #     for (x, y, cw, ch) in final_boxes:
# #         roi = image[y:y+ch, x:x+cw]
# #         if roi.size == 0:
# #             continue
# #         # use a stronger blur for plates
# #         k = max(31, (cw//3)|1)  # odd kernel size
# #         blur = cv2.GaussianBlur(roi, (k, k), 0)
# #         image[y:y+ch, x:x+cw] = blur
# #         if debug:
# #             cv2.rectangle(image, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
# #             cv2.putText(image, "PLATE", (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# #     if debug:
# #         cv2.imshow("plates-blurred-debug", cv2.resize(image, (min(1000, w), int(h * min(1000/w,1.0)))))
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()

# #     return image, final_boxes


# # def blur_sensitive_content(image_path):
# #     """Detects faces and text, lets user choose which faces to blur, and saves blurred copy."""
# #     image = cv2.imread(image_path)
# #     if image is None:
# #         print("❌ Error: Could not load image.")
# #         return

# #     image_copy = image.copy()
# #     faces = detect_faces(image_copy)

# #     if len(faces) > 0:
# #         print(f"Detected {len(faces)} face(s). (Close window to continue) Select which ones to blur:")
# #     else:
# #         print('No faces detected.')

# #     # Draw boxes + large numbered circles
# #     for i, (x, y, w, h) in enumerate(faces):
# #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
# #         center_x, center_y = x + w // 2, y - 20
# #         cv2.circle(image_copy, (center_x, center_y), 25, (0, 255, 255), -1)  # Bigger circle
# #         cv2.putText(image_copy, str(i), (center_x - 12, center_y + 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # Bigger, bold number

# #     # Resize preview to fit on screen
# #     max_width = 1000
# #     scale = min(max_width / image_copy.shape[1], 1.0)
# #     display_image = cv2.resize(image_copy, None, fx=scale, fy=scale)

# #     # Show preview
# #     cv2.imshow("Detected Faces (Close window to continue)", display_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# #     # Ask user which faces to blur
# #     blur_choice = input("Enter face numbers to blur (comma-separated, e.g., 0,2) or 'all': ").strip().lower()

# #     if blur_choice == "all":
# #         to_blur = range(len(faces))
# #     else:
# #         try:
# #             to_blur = [int(i) for i in blur_choice.split(",") if i.strip().isdigit()]
# #         except:
# #             to_blur = []

# #     # Blur selected faces
# #     for i, (x, y, w, h) in enumerate(faces):
# #         if i in to_blur:
# #             face_region = image[y:y+h, x:x+w]
# #             blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
# #             image[y:y+h, x:x+w] = blurred_face

# #     # inside blur_sensitive_content after you handled faces
# #     image, plates = detect_and_blur_plates(image, debug=True)  # debug=True helps tune; set False for normal run
# #     print("Detected plates:", plates)

# #     # then continue with the text (pytesseract) blurring if you still want to try OCR on whole image


# #     # ---- Text Detection ----
# #     print("Detecting and blurring text/faces...")
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.bilateralFilter(gray, 11, 17, 17)
# #     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                 cv2.THRESH_BINARY, 11, 2)
# #     data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config="--psm 11")

# #     n_boxes = len(data['level'])
# #     for i in range(n_boxes):
# #         if int(data['conf'][i]) > 60:
# #             (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
# #             if y >= 0 and x >= 0 and y + h <= image.shape[0] and x + w <= image.shape[1]:
# #                 roi = image[y:y+h, x:x+w]
# #                 if roi.size > 0:
# #                     blur = cv2.GaussianBlur(roi, (25, 25), 0)
# #                     image[y:y+h, x:x+w] = blur

# #     # Save blurred version as new file
# #     output_path = image_path.replace(".jpg", "_blurred.jpg")
# #     cv2.imwrite(output_path, image)
# #     print(f"Saved blurred image as: {output_path}")

# #     # Resize final view for clarity
# #     final_display = cv2.resize(image, None, fx=scale, fy=scale)
# #     cv2.imshow("Final Blurred Image", final_display)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


# def detect_and_blur_plates(image, debug=False):
#     """
#     Detect probable license-plate regions using contour heuristics (and cascade fallback),
#     blur them on a copy of image, and return (blurred_image, list_of_plate_boxes).
#     boxes are (x, y, w, h).
#     """
#     # Work on a local copy so original caller-supplied image isn't modified externally
#     img = image.copy()
#     orig = img.copy()
#     h, w = img.shape[:2]

#     # Preprocess
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray_clahe = clahe.apply(gray)

#     # Edge detection and closing
#     edged = cv2.Canny(gray_clahe, 50, 200)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
#     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # Find contours and filter
#     contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:60]

#     plate_boxes = []
#     for cnt in contours:
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
#         if len(approx) >= 4:
#             x, y, cw, ch = cv2.boundingRect(approx)
#             aspect = cw / float(ch) if ch > 0 else 0
#             area = cw * ch
#             if 2.0 <= aspect <= 6.5 and area > 1500 and cw > 60 and ch > 15:
#                 pad_x = int(0.03 * cw)
#                 pad_y = int(0.12 * ch)
#                 x1 = max(0, x - pad_x)
#                 y1 = max(0, y - pad_y)
#                 x2 = min(w, x + cw + pad_x)
#                 y2 = min(h, y + ch + pad_y)
#                 # filter duplicates
#                 add = True
#                 for (ax, ay, aw, ah) in plate_boxes:
#                     iou_x1 = max(x1, ax)
#                     iou_y1 = max(y1, ay)
#                     iou_x2 = min(x2, ax + aw)
#                     iou_y2 = min(y2, ay + ah)
#                     if iou_x2 > iou_x1 and iou_y2 > iou_y1:
#                         add = False
#                         break
#                 if add:
#                     plate_boxes.append((x1, y1, x2 - x1, y2 - y1))

#     # Cascade fallback
#     if len(plate_boxes) == 0:
#         cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
#         if os.path.exists(cascade_path):
#             plate_cascade = cv2.CascadeClassifier(cascade_path)
#             gray_for_cascade = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             found = plate_cascade.detectMultiScale(gray_for_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(30, 10))
#             for (x, y, cw, ch) in found:
#                 plate_boxes.append((x, y, cw, ch))

#     # Validate / optionally OCR-check (not required for blur)
#     final_boxes = []
#     for (x, y, cw, ch) in plate_boxes:
#         roi = orig[y:y+ch, x:x+cw]
#         if roi.size == 0:
#             continue
#         final_boxes.append((x, y, cw, ch))

#     # Blur the final boxes on the local copy
#     for (x, y, cw, ch) in final_boxes:
#         roi = img[y:y+ch, x:x+cw]
#         if roi.size == 0:
#             continue
#         k = max(31, (cw // 3) | 1)
#         blur = cv2.GaussianBlur(roi, (k, k), 0)
#         img[y:y+ch, x:x+cw] = blur
#         if debug:
#             cv2.rectangle(img, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
#             cv2.putText(img, "PLATE", (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     if debug:
#         cv2.imshow("plates-blurred-debug", cv2.resize(img, (min(1000, w), int(h * min(1000/w, 1.0)))))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return img, final_boxes

# import cv2
# import pytesseract
# import numpy as np
# import os
# import easyocr

# # Path setup
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HP\Downloads\uss\download\tesseract.exe"
# MODEL_DIR = r"C:\Users\HP\Downloads\uss\models"
# configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
# modelFile = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# # Load Face DNN
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# def detect_faces(image):
#     """Detect faces using OpenCV DNN."""
#     (h, w) = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     faces = []
#     for i in range(0, detections.shape[2]):
#         confidence = float(detections[0, 0, i, 2])
#         if confidence > 0.3:  # Slightly relaxed threshold
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             faces.append((startX, startY, endX - startX, endY - startY))
#     return faces

# def blur_sensitive_content(image_path):
#     """Detect faces, text, and plates. Ask which to blur. Save blurred copy."""
#     image = cv2.imread(image_path)
#     if image is None:
#         print("❌ Error: Could not load image.")
#         return

#     image_copy = image.copy()

#     # --- FACE DETECTION ---
#     faces = detect_faces(image_copy)
#     print(f"🧠 Detected {len(faces)} face(s).")

#     for i, (x, y, w, h) in enumerate(faces):
#         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         cv2.putText(image_copy, f"Face {i}", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

#     # --- LICENSE PLATE DETECTION ---
#     print("🔍 Detecting license plates...")
#     plates = detect_and_blur_plates(image_copy)
#     print(f"🚗 Detected {len(plates)} possible license plate(s).")

#     for i, (x, y, w, h, text) in enumerate(plates):
#         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(image_copy, f"Plate {i}: {text}", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

#     # --- DISPLAY SAFE FALLBACK ---
#     try:
#         cv2.imshow("Detected Faces & Plates (Close to continue)", image_copy)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception:
#         temp_path = image_path.replace(".jpg", "_preview.jpg")
#         cv2.imwrite(temp_path, image_copy)
#         print(f"[INFO] Display not supported. Preview saved to: {temp_path}")

#     # --- CHOOSE WHAT TO BLUR ---
#     face_choice = input("Enter face numbers to blur (comma-separated or 'all'/'none'): ").strip().lower()
#     plate_choice = input("Enter plate numbers to blur (comma-separated or 'all'/'none'): ").strip().lower()

#     if face_choice == "all":
#         faces_to_blur = range(len(faces))
#     elif face_choice == "none" or not face_choice:
#         faces_to_blur = []
#     else:
#         faces_to_blur = [int(i) for i in face_choice.split(",") if i.strip().isdigit()]

#     if plate_choice == "all":
#         plates_to_blur = range(len(plates))
#     elif plate_choice == "none" or not plate_choice:
#         plates_to_blur = []
#     else:
#         plates_to_blur = [int(i) for i in plate_choice.split(",") if i.strip().isdigit()]

#     # --- APPLY BLURS ---
#     for i, (x, y, w, h) in enumerate(faces):
#         if i in faces_to_blur:
#             image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)

#     for i, (x, y, w, h, text) in enumerate(plates):
#         if i in plates_to_blur:
#             image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)

#     # --- TEXT BLURRING ---
#     print("🔤 Detecting and blurring text...")
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     n_boxes = len(data['level'])
#     for i in range(n_boxes):
#         if int(data['conf'][i]) > 70:
#             x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#             roi = image[y:y+h, x:x+w]
#             if roi.size > 0:
#                 image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (25, 25), 0)

#     # --- SAVE OUTPUT ---
#     output_path = image_path.replace(".jpg", "_blurred.jpg").replace(".png", "_blurred.png")
#     cv2.imwrite(output_path, image)
#     print(f"✅ Saved blurred image as: {output_path}")

import cv2
import pytesseract
import numpy as np
import os
import easyocr

# --- PATH SETUP ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HP\Downloads\uss\download\tesseract.exe"
MODEL_DIR = r"C:\Users\HP\Downloads\uss\models"
configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
modelFile = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# ----------------------------- FACE DETECTION -----------------------------
def detect_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

# -------------------------- LICENSE PLATE DETECTION -----------------------
def detect_plates(image, debug=False):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edged = cv2.Canny(gray, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:80]

    plate_boxes = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h > 0 else 0
            area = w * h
            if 2.0 <= aspect <= 6.5 and area > 1500 and w > 60 and h > 15:
                plate_boxes.append((x, y, w, h))
    return plate_boxes

# ----------------------- MAIN BLUR LOGIC -----------------------
def blur_sensitive_content(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image.")
        return

    display_img = image.copy()

    # --- FACE DETECTION ---
    faces = detect_faces(display_img)
    print(f"🧠 Detected {len(faces)} face(s).")

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(display_img, f"FACE {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- PLATE DETECTION ---
    print("🔍 Detecting license plates...")
    plates = detect_plates(display_img)
    print(f"🚗 Detected {len(plates)} possible license plate(s).")

    for i, (x, y, w, h) in enumerate(plates):
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(display_img, f"PLATE {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- SHOW DETECTIONS ---
    try:
        cv2.imshow("Detected Faces & Plates (Close to continue)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        preview_path = image_path.replace(".jpg", "_preview.jpg").replace(".png", "_preview.png")
        cv2.imwrite(preview_path, display_img)
        print(f"[INFO] GUI not supported — saved preview image at:\n{preview_path}")
        print("👉 Open that image to see the numbers for faces/plates before typing your choices.")


    # --- CHOOSE ---
    face_choice = input("Enter face numbers to blur (comma-separated or 'all'/'none'): ").strip().lower()
    plate_choice = input("Enter plate numbers to blur (comma-separated or 'all'/'none'): ").strip().lower()

    if face_choice == "all":
        faces_to_blur = range(len(faces))
    elif face_choice == "none" or not face_choice:
        faces_to_blur = []
    else:
        faces_to_blur = [int(i) for i in face_choice.split(",") if i.strip().isdigit()]

    if plate_choice == "all":
        plates_to_blur = range(len(plates))
    elif plate_choice == "none" or not plate_choice:
        plates_to_blur = []
    else:
        plates_to_blur = [int(i) for i in plate_choice.split(",") if i.strip().isdigit()]

    # --- BLUR ---
    for i, (x, y, w, h) in enumerate(faces):
        if i in faces_to_blur:
            roi = image[y:y+h, x:x+w]
            image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)

    for i, (x, y, w, h) in enumerate(plates):
        if i in plates_to_blur:
            roi = image[y:y+h, x:x+w]
            image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)

    # --- TEXT DETECTION ---
    print("🔤 Detecting and blurring text...")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    for i in range(len(data['level'])):
        if int(data['conf'][i]) > 70:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (25, 25), 0)

    output_path = image_path.replace(".jpg", "_blurred.jpg").replace(".png", "_blurred.png")
    cv2.imwrite(output_path, image)
    print(f"✅ Saved blurred image as: {output_path}")
