import cv2

image= cv2.imread('premiere_image.jpg')

if image is None:
    print("Erreur : Impossible de charger l'image.")

else:
    boxes = [
        (100, 200, 300, 500),
        (400, 250, 550, 600),
        (700, 150, 850, 450)
    ]
   
    for player_id, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessine un rectangle vert autour de la zone d'
        cv2.putText(image, f'ID : {player_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Ajoute un texte au-dessus du rectangle

    saved = cv2.imwrite('image_annotee.jpg', image)
    print("Image annotée sauvegardée avec succès :", saved)