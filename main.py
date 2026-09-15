import cv2

video_path = 'input_videos/match.mp4'
capture = cv2.VideoCapture(video_path)

print("La video à été chargé avec succès :", capture.isOpened())

count = 0

while True:
    success, frame = capture.read()

    if not success:
        print("Fin de la vidéo ou lecture impossible.")
        break
    count += 1

print("Nombre de frames lues :", count)



fps = capture.get(cv2.CAP_PROP_FPS)
print("Fréquence d’images (FPS) :", fps)

if fps > 0:
    duration = count / fps
    print("Durée de la vidéo (en secondes) :", duration)
else:
    print("Impossible de calculer la durée : FPS indisponibles.")

capture.release()