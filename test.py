# from roboflow import Roboflow
# rf = Roboflow(api_key="FajXd8SbJcsO2D8EuyCX")
# project = rf.workspace().project("forceps-detection")
# model = project.version(2).model

# infer on a local image
# print(model.predict(sources="test_image.jpg", confidence=50, overlap=50).json())
from ultralytics import YOLO
model = YOLO('model/yolov8n_custom_20231019.pt')
results = model.predict(source='IMG_8947_640480.mp4', show=True, stream=True, save=True)  # accepts all formats

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
