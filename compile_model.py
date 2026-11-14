from ultralytics import YOLO

# This downloads the PyTorch framework model
model = YOLO("yolo11n.pt")

# This converts it into NVIDIA's lightning-fast hardware map
model.export(format="engine", half=True, device=0)
