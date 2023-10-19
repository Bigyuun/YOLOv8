from roboflow import Roboflow
from ultralytics import YOLO

DEVICE = 'cuda'

# rf = Roboflow(api_key="FajXd8SbJcsO2D8EuyCX")
# project = rf.workspace("detr-cjz4w").project("forceps-detection")
# dataset = project.version(5).download("yolov8")

# rf = Roboflow(api_key="FajXd8SbJcsO2D8EuyCX")
# project = rf.workspace("detr-cjz4w").project("forceps-detection-2")
# dataset = project.version(2).download("yolov8")

# rf = Roboflow(api_key="FajXd8SbJcsO2D8EuyCX")
# project = rf.workspace("detr-cjz4w").project("forceps-detection-3")
# dataset = project.version(1).download("yolov8")

# from roboflow import Roboflow
# rf = Roboflow(api_key="FajXd8SbJcsO2D8EuyCX")
# project = rf.workspace("detr-cjz4w").project("forceps-detection-4")
# dataset = project.version(1).download("yolov8")

model = YOLO('yolov8n_custom_20231010_v1.pt')
model.to(DEVICE)
results = model.train(data='Forceps-Detection-4-1/data.yaml', batch=32, epochs=25, imgsz=(1280,720), plots=True)

print('end')