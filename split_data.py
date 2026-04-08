import os
import shutil
import random

random.seed(42)

source_dir = "Training"
target_base = "data"

classes = os.listdir(source_dir)

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split in splits:
        os.makedirs(os.path.join(target_base, split, cls), exist_ok=True)

        for img in splits[split]:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(target_base, split, cls, img)
            shutil.copy(src, dst)

print("✅ Dataset split completed!")