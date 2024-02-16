from ultralytics import YOLO
from pathlib import Path
import sys
import os

def main():
    os.environ['WANDB_DISABLED'] = 'true'
    # Load a YOLOv8n model
    model = YOLO(Path('yolov8n-mobilenet.yaml').absolute())

    model.tune(
        project=sys.argv[1],
        data=str(Path('robot-detection.yaml').absolute()),
        imgsz=96,
        batch=1024,
        epochs=200,
        iterations=1000,
        plots=False,
        devices=[0,1],
        space={
            # 'optimizer': ['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            "label_smoothing": (0.0, 0.3),  # label smoothing (epsilon)
        }
    )

if __name__ == "__main__":
    main()