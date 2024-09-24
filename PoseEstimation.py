import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0

# Lista de índices de los landmarks que quieres mostrar y unir
landmarks_to_show = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Conexiones entre los puntos para formar el esqueleto
connections = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # Brazos
    (11, 12),  # Hombros
    (23, 24),  # Caderas
    (11, 23), (12, 24),  # Hombros a caderas
    (23, 25), (25, 27), (24, 26), (26, 28)  # Piernas
]

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        h, w, _ = img.shape
        points = {}

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in landmarks_to_show:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points[idx] = (cx, cy)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                print(f"Landmark {idx}: ({cx}, {cy})")

        # Dibuja líneas entre los puntos seleccionados
        for connection in connections:
            if connection[0] in points and connection[1] in points:
                cv2.line(img, points[connection[0]], points[connection[1]], (0, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
