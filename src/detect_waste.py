
from pathlib import Path
from ultralytics import YOLO

def main():
    image_path = Path("images/test.png")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("/Users/pramodnagisetty/Downloads/Recyclables and Waste Still Life.png")
        return

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("Detecting objects...")
    results = model.predict(
        source=str(image_path),
        save=True,
        conf=0.25
    )

    print("Detection complete!")

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            print(f"{label}: {confidence:.2%} confidence")

    print("Annotated image saved in the runs/detect folder.")

if __name__ == "__main__":
    main()
