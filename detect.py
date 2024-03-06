import cv2
from ultralytics import YOLO

model = YOLO('./best.pt')

cat_classes = ['scorpion', 'snake', 'spider']

img_path = './test_images/scorpion/scorpion_1.jpg'

result = model.predict(img_path)

cap = cv2.imread(img_path)

result_obj = model.predict(cap)

result = result_obj[0].boxes

cls = result.cls.numpy()[0]
conf = result.conf.numpy()[0]
xyxy = result.xyxy.int().numpy()
x0 = xyxy[0][0]
y0 = xyxy[0][1]
x1 = xyxy[0][2]
y1 = xyxy[0][3]


cv2.rectangle(cap, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
cv2.putText(cap, f'{cat_classes[0]}: {conf*100:.2f}%', (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
cv2.imshow('image', cap)
cv2.waitKey(-1)
cv2.destroyAllWindows()
