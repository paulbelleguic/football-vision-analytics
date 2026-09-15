from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")  # load a pretrained YOLOv26n model

results = model.predict(
    source="premiere_image.jpg",
    device="cpu",
    save=False,
    imgsz=1280,
    conf=0.10,
)

result = results[0]
image = result.orig_img.copy()

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    class_name = model.names[class_id]

    if confidence >= 0.20:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(class_name, round(confidence, 3))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
saved = cv2.imwrite('image_detectee.jpg', image)
print("Image détectée sauvegardée avec succès :", saved)