import cv2
import mediapipe as mp
import numpy as np
import time
import math

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

width = 1080
height = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
pTime = 0

# Lista de índices de los landmarks que quieres mostrar y unir
landmarks_to_show = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

'''
0 - nariz
11 - hombro izquierdo
12 - hombro derecho
13 - codo izquierdo
14 - codo derecho
15 - muñeca izquierda
16 - muñeca derecha
23 - cadera izquierda
24 - cadera derecha
25 - rodilla izquierda
26 - rodilla derecha
27 - tobillo izquierdo
28 - tobillo derecho
'''

# Conexiones entre los puntos para formar el esqueleto
connections = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # Brazos
    (11, 12),  # Hombros
    (23, 24),  # Caderas
    (11, 23), (12, 24),  # Hombros a caderas
    (23, 25), (25, 27), (24, 26), (26, 28)  # Piernas
]

def get_landmark(pose_landmarks, landmark_index):
    landmark = pose_landmarks.landmark[landmark_index]
    return [landmark.x, landmark.y]


def calculate_angle(a, b, c):
    # Calcular el ángulo entre tres puntos
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    angle = round(angle,2)
    if angle > 180:
        angle_new = 360 - angle
        # print(f"original angle:{angle}, new angle: {angle_new}")
        return angle_new
    elif angle < 0:
        angle_new = -1*angle
        # print(f"original angle:{angle}, new angle: {angle_new}")
        return angle_new
    else:
        # print(f"original angle:{angle}, new angle: {angle}")
        return angle
    #return angle + 360 if angle < 0 else angle

def calculate_distance(point1, point2, width = width, height = height):
    x1, y1 = point1[0]*width, point1[1]*height
    x2, y2 = point2[0]*width, point2[1]*height

    distancia = math.hypot(x2 - x1, y2 - y1)

    return round(distancia, 2)
    # return distancia

def is_taichi_push(pose_landmarks ):
    # Obtener las coordenadas de los puntos clave
    cadera_izquierda = get_landmark(pose_landmarks, 23)
    rodilla_izquierda = get_landmark(pose_landmarks, 25)
    tobillo_izquierdo = get_landmark(pose_landmarks, 27)
    cadera_derecha = get_landmark(pose_landmarks, 24)
    rodilla_derecha = get_landmark(pose_landmarks, 26)
    tobillo_derecho = get_landmark(pose_landmarks, 28)
    mano_izquierda = get_landmark(pose_landmarks, 15)
    mano_derecha = get_landmark(pose_landmarks, 16)
    hombro_izquierdo = get_landmark(pose_landmarks, 11)
    hombro_derecho = get_landmark(pose_landmarks, 12)
    codo_izquierdo = get_landmark(pose_landmarks, 13)
    codo_derecho = get_landmark(pose_landmarks, 14)
    
    # Calcular los ángulos de las piernas
    left_leg_angle = calculate_angle(cadera_izquierda, rodilla_izquierda, tobillo_izquierdo)
    right_leg_angle = calculate_angle(cadera_derecha, rodilla_derecha, tobillo_derecho)

    cv2.putText(img, f'pierna izquierda: {int(left_leg_angle)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'pierna derecha:  {int(right_leg_angle)}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Calcular los ángulos de los brazos
    left_arm_angle = calculate_angle(hombro_izquierdo, codo_izquierdo, mano_izquierda)
    right_arm_angle = calculate_angle(hombro_derecho, codo_derecho, mano_derecha)

    cv2.putText(img, f'brazo izquierdo: {int(left_arm_angle)}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'brazo derecho:  {int(right_arm_angle)}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    #Calcular distancia
    hands_distance = calculate_distance(mano_derecha, mano_izquierda)
    elbows_distance = calculate_distance(codo_derecho, codo_izquierdo)

    print(f"distance: {hands_distance}, {elbows_distance}")

    # Definir las condiciones para la pose del push de Tai Chi
    if right_leg_angle > 170 and left_leg_angle < 110 and left_leg_angle > 80 and right_arm_angle > 140 and left_arm_angle > 140:
        if hands_distance < 35 and elbows_distance < 35:
            # print("yes") 
            return True
    
    return False

def is_taichi_split(pose_landmarks):
    # Obtener las coordenadas de los puntos clave
    cadera_izquierda = get_landmark(pose_landmarks, 23)
    rodilla_izquierda = get_landmark(pose_landmarks, 25)
    tobillo_izquierdo = get_landmark(pose_landmarks, 27)
    cadera_derecha = get_landmark(pose_landmarks, 24)
    rodilla_derecha = get_landmark(pose_landmarks, 26)
    tobillo_derecho = get_landmark(pose_landmarks, 28)
    mano_izquierda = get_landmark(pose_landmarks, 15)
    mano_derecha = get_landmark(pose_landmarks, 16)
    hombro_izquierdo = get_landmark(pose_landmarks, 11)
    hombro_derecho = get_landmark(pose_landmarks, 12)
    codo_izquierdo = get_landmark(pose_landmarks, 13)
    codo_derecho = get_landmark(pose_landmarks, 14)

    # Calcular los ángulos de las piernas
    left_leg_angle = calculate_angle(cadera_izquierda, rodilla_izquierda, tobillo_izquierdo)
    right_leg_angle = calculate_angle(cadera_derecha, rodilla_derecha, tobillo_derecho)

    # Calcular los ángulos de los brazos
    left_arm_angle = calculate_angle(hombro_izquierdo, codo_izquierdo, mano_izquierda)
    right_arm_angle = calculate_angle(hombro_derecho, codo_derecho, mano_derecha)

    # Calcular distancias
    hands_distance = calculate_distance(mano_derecha, mano_izquierda)


    # Definir las condiciones para la pose del split de Tai Chi
    if left_leg_angle > 170 and right_leg_angle < 110 and right_leg_angle > 80 and right_arm_angle > 170 and left_arm_angle > 70 and left_arm_angle < 100:
        if hands_distance > 60:
            # print("yes")
            return True

    return False

def is_taichi_ward_off(pose_landmarks):
    # Obtener las coordenadas de los puntos clave
    cadera_izquierda = get_landmark(pose_landmarks, 23)
    rodilla_izquierda = get_landmark(pose_landmarks, 25)
    tobillo_izquierdo = get_landmark(pose_landmarks, 27)
    cadera_derecha = get_landmark(pose_landmarks, 24)
    rodilla_derecha = get_landmark(pose_landmarks, 26)
    tobillo_derecho = get_landmark(pose_landmarks, 28)
    mano_izquierda = get_landmark(pose_landmarks, 15)
    mano_derecha = get_landmark(pose_landmarks, 16)
    hombro_izquierdo = get_landmark(pose_landmarks, 11)
    hombro_derecho = get_landmark(pose_landmarks, 12)
    codo_izquierdo = get_landmark(pose_landmarks, 13)
    codo_derecho = get_landmark(pose_landmarks, 14)

    # Calcular los ángulos de las piernas
    left_leg_angle = calculate_angle(cadera_izquierda, rodilla_izquierda, tobillo_izquierdo)
    right_leg_angle = calculate_angle(cadera_derecha, rodilla_derecha, tobillo_derecho)

    # Calcular los ángulos de los brazos
    left_arm_angle = calculate_angle(hombro_izquierdo, codo_izquierdo, mano_izquierda)
    right_arm_angle = calculate_angle(hombro_derecho, codo_derecho, mano_derecha)

    # Calcular distancias
    hands_distance = calculate_distance(mano_derecha, mano_izquierda)

    # Definir las condiciones para la pose del split de Tai Chi
    if left_leg_angle > 80 and left_leg_angle < 120 and right_leg_angle > 120 and right_leg_angle < 160 and left_arm_angle < 160 and right_arm_angle < 70:
        if hands_distance > 60:
            # print("yes")
            return True

    return False

def is_taichi_pull_down(pose_landmarks):
    # Obtener las coordenadas de los puntos clave
    cadera_izquierda = get_landmark(pose_landmarks, 23)
    rodilla_izquierda = get_landmark(pose_landmarks, 25)
    tobillo_izquierdo = get_landmark(pose_landmarks, 27)
    cadera_derecha = get_landmark(pose_landmarks, 24)
    rodilla_derecha = get_landmark(pose_landmarks, 26)
    tobillo_derecho = get_landmark(pose_landmarks, 28)
    mano_izquierda = get_landmark(pose_landmarks, 15)
    mano_derecha = get_landmark(pose_landmarks, 16)
    hombro_izquierdo = get_landmark(pose_landmarks, 11)
    hombro_derecho = get_landmark(pose_landmarks, 12)
    codo_izquierdo = get_landmark(pose_landmarks, 13)
    codo_derecho = get_landmark(pose_landmarks, 14)

    # Calcular los ángulos de las piernas
    left_leg_angle = calculate_angle(cadera_izquierda, rodilla_izquierda, tobillo_izquierdo)
    right_leg_angle = calculate_angle(cadera_derecha, rodilla_derecha, tobillo_derecho)

    # Calcular los ángulos de los brazos
    left_arm_angle = calculate_angle(hombro_izquierdo, codo_izquierdo, mano_izquierda)
    right_arm_angle = calculate_angle(hombro_derecho, codo_derecho, mano_derecha)

    # Calcular distancias
    hands_distance = calculate_distance(mano_derecha, mano_izquierda)


    # Definir las condiciones para la pose del split de Tai Chi
    if left_leg_angle > 170 and right_leg_angle < 110 and left_arm_angle > 70 and left_arm_angle < 100 and right_arm_angle < 10:
        if hands_distance > 60:
            # print("yes")
            return True

    return False    

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
                # print(f"Landmark {idx}: ({cx}, {cy})")


        # Dibuja líneas entre los puntos seleccionados
        for connection in connections:
            if connection[0] in points and connection[1] in points:
                cv2.line(img, points[connection[0]], points[connection[1]], (0, 0, 255), 2)


        # Verificar si la pose del split de Tai Chi se detecta
        if is_taichi_split(results.pose_landmarks):
            cv2.putText(img, 'Tai Chi Split Detected', (20, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if is_taichi_push(results.pose_landmarks):
            cv2.putText(img, 'Tai Chi Push Detected', (20, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  

        if is_taichi_ward_off(results.pose_landmarks):
            cv2.putText(img, 'Tai Chi Ward Off Detected', (20, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    

        if is_taichi_pull_down(results.pose_landmarks):
            cv2.putText(img, 'Tai Chi Pull Down Detected', (20, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()