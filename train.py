from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data/data.yaml", epochs=50)  

metrics = model.val()

# results = model("https://ultralytics.com/images/bus.jpg")

path = model.export(format="onnx")