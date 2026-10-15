
import cv2
from ultralytics import YOLO

# Load your trained waste classification model
model = YOLO("runs/classify/train/weights/best.pt")

# Open Mac webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

print("Real-time waste classification started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read camera frame.")
        break

    # Classify the camera frame
    results = model.predict(frame, verbose=False)
    result = results[0]

    class_id = result.probs.top1
    confidence = float(result.probs.top1conf)
    category = result.names[class_id]

    # Display prediction on the video
    label = f"{category.upper()} - {confidence:.1%}"

    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
