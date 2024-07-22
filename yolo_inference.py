from ultralytics import YOLO
import torch

model = YOLO('yolov8l')

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

results = model.predict('/Users/amananand/PycharmProjects/Football-analysis/input_videos/08fd33_4.mp4',save=True).to(device)

print(results[0])

print('=============================')

for box in results[0].boxes:
    print(box)