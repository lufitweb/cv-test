from ultralytics import YOLO

model = YOLO('models/last.pt')

results = model.predict('input_video/3507660-uhd_3840_2160_30fps.mp4', save=True)
print(results[0])
print('==================================')
for box in results[0].boxes:
    print(box)