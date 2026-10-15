
from pathlib import Path
import random
import shutil

source = Path("dataset")
destination = Path("dataset_split")

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

random.seed(42)

for category in categories:
    images = [
        p for p in (source / category).iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)
    split = int(len(images) * 0.8)

    for folder, selected in [
        ("train", images[:split]),
        ("val", images[split:])
    ]:
        target = destination / folder / category
        target.mkdir(parents=True, exist_ok=True)

        for image in selected:
            shutil.copy2(image, target / image.name)

        print(f"{folder}/{category}: {len(selected)} images")

print("Dataset splitting completed!")
