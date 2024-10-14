import cv2
import pytesseract
import re

# Configura el path de Tesseract si estás en Windows
pytesseract.pytesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Función para filtrar el texto reconocido y dejar solo letras, números y guiones
def filter_plate_text(text):
    # Usamos una expresión regular para filtrar solo letras, números y guiones
    return re.sub(r'[^A-Za-z0-9-]', '', text)

# Función para leer placas de un archivo .txt
def load_placas(file_path):
    with open(file_path, 'r') as file:
        return {line.strip().upper() for line in file.readlines()}

# Cargamos las placas del archivo
placas_registradas = load_placas('placas_peru.txt')

# Función para procesar el reconocimiento de texto de una imagen
def recognize_plate(image):
    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))  # Aplicamos un pequeño desenfoque para reducir el ruido
    
    # Detectamos los bordes con Canny
    canny = cv2.Canny(gray, 150, 200)
    canny = cv2.dilate(canny, None, iterations=1)  # Dilatamos para resaltar los bordes

    # Encontramos los contornos en la imagen procesada
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Tomamos los 10 contornos más grandes

    plate = None
    for contour in contours:
        # Aproximamos los contornos a formas poligonales
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Si el contorno tiene 4 lados, puede ser un rectángulo (posible placa)
        if len(approx) == 4:
            plate = approx
            break

    if plate is not None:
        # Dibujamos el contorno de la posible placa
        cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)

        # Recortamos el área de la placa
        x, y, w, h = cv2.boundingRect(plate)
        plate_image = gray[y:y + h, x:x + w]  # Usamos la imagen en escala de grises para mejor OCR

        # Usamos PyTesseract para hacer el reconocimiento de texto en la placa
        plate_text = pytesseract.image_to_string(plate_image, config='--psm 11')  # Modo PSM 11 para texto disperso
        
        # Filtramos el texto para obtener solo letras, números y guiones
        filtered_text = filter_plate_text(plate_text).upper().strip()
        print(f"Placa reconocida: {filtered_text}")

        # Verificamos si la placa está en el archivo de registros
        if filtered_text in placas_registradas:
            print("PUEDE INGRESAR")
            cap.release()
            cv2.destroyAllWindows()

        else:
            print("NO REGISTRADO")

        return plate_image

    return None

# Iniciamos la captura de video en tiempo real
cap = cv2.VideoCapture(0)
frame_counter = 0  # Contador de frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 5 == 0:  # Procesamos solo 1 de cada 5 frames para reducir el uso de CPU
        # Procesamos el frame para buscar placas
        plate_image = recognize_plate(frame)

        # Mostramos la imagen en tiempo real
        cv2.imshow("Video en tiempo real", frame)

        if plate_image is not None:
            # Mostramos la imagen de la placa recortada
            cv2.imshow("Placa detectada", plate_image)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()
