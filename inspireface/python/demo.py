# sample in ./python/README.md
import cv2
import inspireface as isf


isf.launch(model_name="Megatron_TRT")

# Create session with required features enabled
session = isf.InspireFaceSession(
    # opt=isf.HF_ENABLE_NONE,  # Optional features
    param=1,
    detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT,  # Detection mode
)

# Set detection confidence threshold
session.set_detection_confidence_threshold(0.5)

# Read image
image = cv2.imread(
    "/kaggle/temp/insightface/python-package/insightface/data/images/t1.jpg"
)
assert image is not None, "Please check if the image path is correct"

# Perform face detection
faces = session.face_detection(image)
print(f"Detected {len(faces)} faces")

# Draw detection results on image
draw = image.copy()
for idx, face in enumerate(faces):
    # Get face bounding box coordinates
    x1, y1, x2, y2 = face.location

    # Calculate rotated box parameters
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    angle = face.roll

    # Draw rotated box
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.drawContours(draw, [box], 0, (100, 180, 29), 2)

    # Draw landmarks
    landmarks = session.get_face_dense_landmark(face)
    for x, y in landmarks.astype(int):
        cv2.circle(draw, (x, y), 0, (220, 100, 0), 2)
