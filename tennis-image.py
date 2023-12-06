import cv2
import numpy as np
from openvino.inference_engine import IECore
from IPython.display import display, Image
from PIL import Image

model_path = "raw_models/intel/human-pose-estimation-0005/FP16/human-pose-estimation-0005"
ie = IECore()
net = ie.read_network(model=model_path + ".xml", weights=model_path + ".bin")
exec_net = ie.load_network(network=net, device_name="CPU")

def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (input_shape[3], input_shape[2]))
    input_image = np.transpose(resized_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, 0)
    return image, input_image

input_shape = net.input_info["image"].tensor_desc.dims
image, input_image = preprocess_image("foto/dritto.jpg", input_shape)

# Esegui l'inferenza sul tuo modello.
results = exec_net.infer(inputs={"image": input_image})

# Connessioni tra i keypoints basate sull'ordine standard di OpenPose:
scheletro = [
    (4, 2),  # occhi testa
    (9, 7), (7, 5), (5, 6),  # Braccio sinistro
    (10, 8), (8, 6),  # Braccio destro
    (11, 13), (13, 15),  # Gamba sinistra
    (12, 14), (14, 16),    # Gamba destra
    (12, 11),     # Bacino
    (12, 6),     # Fianco destro
    (5, 11),     # fianco sinistro
]

# Visualizzazione dei keypoints:
# Ogni keypoint ha una coppia (x, y) sulla tua immagine.
print(results.keys())

heatmaps = results['heatmaps'][0]  # Prendiamo il primo batch, assumendo un batch size di 1

# Estraiamo la posizione dei keypoints dalle mappe di calore
keypoints = []
for heatmap in heatmaps:
    _, _, loc, _ = cv2.minMaxLoc(heatmap)  # Questo restituisce la posizione del valore massimo nel heatmap
    x, y = loc
    keypoints.append((x, y))


# Ottieni le dimensioni dell'immagine
height, width, _ = image.shape
print(heatmaps.shape)


# Visualizziamo i keypoints sull'immagine
for idx, (x, y) in enumerate(keypoints):
    cv2.circle(image, (int(x * width / heatmaps.shape[2]), int(y * height / heatmaps.shape[1])), 3, (0, 255, 0), -1)
    cv2.putText(image, str(idx), (int(x * width / heatmaps.shape[2]), int(y * height / heatmaps.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
# Visualizziamo scheletro
for (start, end) in scheletro:
    start_point = (int(keypoints[start][0] * width / heatmaps.shape[2]), int(keypoints[start][1] * height / heatmaps.shape[1]))
    end_point = (int(keypoints[end][0] * width / heatmaps.shape[2]), int(keypoints[end][1] * height / heatmaps.shape[1]))
    cv2.line(image, start_point, end_point, (0, 255, 255), 2)  # colore giallo e spessore 2

# Converti l'immagine di OpenCV in un formato compatibile con PIL e poi mostra l'immagine
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
display(pil_image)

# Supponiamo che tu abbia già estratto i keypoints come mostrato in precedenza
torso_center = ((keypoints[8][0] + keypoints[7][0]) / 2, (keypoints[8][1] + keypoints[7][1]) / 2)

distance_right_arm = abs(keypoints[10][0] - torso_center[0])
distance_left_arm = abs(keypoints[9][0] - torso_center[0])

if distance_right_arm < distance_left_arm:
    print("Dritto")
else:
    print("Rovescio")

def is_serving(keypoints):
    # Estraiamo le coordinate dei keypoints di interesse
    right_shoulder = keypoints[6]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]
    
    left_shoulder = keypoints[5]
    left_elbow = keypoints[7]
    left_wrist = keypoints[9]
    
    right_hip = keypoints[12]
    right_knee = keypoints[14]
    
    # Condizioni che indicano un possibile servizio:
    # 1. Il polso destro è più alto del gomito destro e della spalla destra
    # 2. La palla (assunta vicino al polso sinistro) è sotto il gomito sinistro
    # 3. L'anca destra è leggermente più alta del ginocchio destro (indicando una leggera piega nella preparazione)
    if (right_wrist[1] < right_elbow[1] and right_wrist[1] < right_shoulder[1] and
        left_wrist[1] > left_elbow[1] and 
        right_hip[1] < right_knee[1]):
        return True
    return False

# Usiamo la funzione per determinare se il giocatore sta effettuando un servizio
if is_serving(keypoints):
    print("Il giocatore sta effettuando un servizio.")
else:
    # Puoi estendere quest'altro ramo per altri colpi o azioni se necessario
    print("Non è un servizio.")
