from pathlib import Path

import cv2
from ultralytics import YOLO


model = YOLO("yolo26n.pt")

output_dir = Path("detections_filtrees")
output_dir.mkdir(exist_ok=True)

capture = cv2.VideoCapture("input_videos/match.mp4")

count = 0
ball_frames = 0

try:
    if not capture.isOpened():
        print("Erreur : impossible d’ouvrir la vidéo.")
    else:
        while count < 30:
            success, frame = capture.read()

            if not success:
                print("Fin de la vidéo ou lecture impossible.")
                break

            results = model.predict(
                source=frame,
                device="cpu",
                imgsz=1280,
                conf=0.20,
                save=False,
                verbose=False,
            )

            result = results[0]
            image = frame.copy()
            ball_detected = False

            # Examiner et dessiner chaque détection.
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = result.names[class_id]

                if class_name == "sports ball":
                    ball_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Enregistrer avant de mettre à jour les compteurs.
            output_path = output_dir / f"frame_{count:03d}.jpg"
            saved = cv2.imwrite(str(output_path), image)

            if not saved:
                print(f"Erreur d’enregistrement : {output_path}")
                break

            count += 1

            if ball_detected:
                ball_frames += 1

            print(
                f"Image {count}/30 : "
                f"{len(result.boxes)} détections "
                f"— ballon détecté : {ball_detected}"
            )

finally:
    capture.release()

# Bilan final : en dehors de la boucle.
print(f"\nTerminé : {count} images enregistrées.")

if count > 0:
    rate = 100 * ball_frames / count
    print(
        f"Ballon détecté sur {ball_frames}/{count} images "
        f"({rate:.1f} %)."
    )