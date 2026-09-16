from pathlib import Path
import math

import cv2
from ultralytics import YOLO


def draw_box(image, box, label, color):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    project = Path(__file__).resolve().parent

    players_model = YOLO(
        str(project / "runs/train/football_yolo26n_V1/weights/best.pt")
    )
    ball_model = YOLO(
        str(project / "runs/train/ball_smoke_test/weights/best.pt")
    )

    output_dir = project / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_detection.mp4"

    capture = cv2.VideoCapture(str(project / "input_videos/match.mp4"))
    writer = None

    count = 0
    ball_frames = 0

    try:
        if not capture.isOpened():
            raise RuntimeError("Impossible d’ouvrir input_videos/match.mp4")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("FPS indisponibles : export arrêté.")

        max_frames = round(fps * 10)

        while count < max_frames:
            success, frame = capture.read()
            if not success:
                break

            # Créer la vidéo de sortie aux dimensions de l’original.
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError("Impossible de créer la vidéo MP4.")

            # Les deux modèles analysent l’image originale.
            people = players_model.predict(
                frame,
                device=0,
                imgsz=640,
                conf=0.20,
                verbose=False,
            )[0]

            balls = ball_model.predict(
                frame,
                device=0,
                imgsz=640,
                conf=0.20,
                verbose=False,
            )[0]

            annotated = frame.copy()

            for box in people.boxes:
                class_id = int(box.cls.item())
                name = people.names[class_id]

                # Le ballon sera traité par le modèle dédié.
                if name not in {"player", "goalkeeper", "referee"}:
                    continue

                confidence = float(box.conf.item())
                color = (0, 165, 255) if name == "referee" else (0, 255, 0)
                draw_box(
                    annotated,
                    box,
                    f"{name} {confidence:.2f}",
                    color,
                )

            ball_detected = False

            for box in balls.boxes:
                class_id = int(box.cls.item())
                if balls.names[class_id] != "ball":
                    continue

                ball_detected = True
                confidence = float(box.conf.item())
                draw_box(
                    annotated,
                    box,
                    f"ball {confidence:.2f}",
                    (255, 255, 0),
                )

            count += 1
            if ball_detected:
                ball_frames += 1

            cv2.putText(
                annotated,
                f"Frame {count} | Detection uniquement",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(annotated)

            if count % 30 == 0:
                print(f"{count}/{max_frames} images traitées")

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if count == 0:
        raise RuntimeError("Aucune image lue.")

    print(f"\nVidéo enregistrée : {output_path}")
    print(f"Durée exportée : {count / fps:.1f} secondes")
    print(f"Ballon proposé sur {ball_frames}/{count} images")


if __name__ == "__main__":
    main()