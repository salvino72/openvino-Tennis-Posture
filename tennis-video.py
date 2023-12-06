import cv2
import numpy as np
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Modello e configurazione
model_path = "raw_models/intel/human-pose-estimation-0005/FP16/human-pose-estimation-0005"
ie = IECore()
net = ie.read_network(model=model_path + ".xml", weights=model_path + ".bin")
exec_net = ie.load_network(network=net, device_name="CPU")
input_shape = net.input_info["image"].tensor_desc.dims

# Connessioni tra i keypoints
scheletro = [
    (4, 2),  # occhi testa
    (9, 7), (7, 5), (5, 6),  # Braccio sinistro
    (10, 8), (8, 6),  # Braccio destro
    (11, 13), (13, 15),  # Gamba sinistra
    (12, 14), (14, 16),  # Gamba destra
    (12, 11),  # Bacino
    (12, 6),  # Fianco destro
    (5, 11),  # fianco sinistro
]

def preprocess_frame(frame, input_shape):
    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
    input_frame = np.transpose(resized_frame, (2, 0, 1))
    input_frame = np.expand_dims(input_frame, 0)
    return frame, input_frame

def is_serving(keypoints):
    right_shoulder = keypoints[6]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]
    left_shoulder = keypoints[5]
    left_elbow = keypoints[7]
    left_wrist = keypoints[9]
    right_hip = keypoints[12]
    right_knee = keypoints[14]
    if (right_wrist[1] < right_elbow[1] and right_wrist[1] < right_shoulder[1] and
        left_wrist[1] > left_elbow[1] and 
        right_hip[1] < right_knee[1]):
        return True
    return False

cap = cv2.VideoCapture('video/tennis4.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
MAX_FRAMES = 100

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        break

    frame, input_frame = preprocess_frame(frame, input_shape)
    results = exec_net.infer(inputs={"image": input_frame})
    
    heatmaps = results['heatmaps'][0]
    keypoints = []
    for heatmap in heatmaps:
        _, _, loc, _ = cv2.minMaxLoc(heatmap)
        x, y = loc
        keypoints.append((x, y))

    height, width, _ = frame.shape
    for idx, (x, y) in enumerate(keypoints):
        cv2.circle(frame, (int(x * width / heatmaps.shape[2]), int(y * height / heatmaps.shape[1])), 3, (0, 255, 0), -1)
        cv2.putText(frame, str(idx), (int(x * width / heatmaps.shape[2]), int(y * height / heatmaps.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (start, end) in scheletro:
        start_point = (int(keypoints[start][0] * width / heatmaps.shape[2]), int(keypoints[start][1] * height / heatmaps.shape[1]))
        end_point = (int(keypoints[end][0] * width / heatmaps.shape[2]), int(keypoints[end][1] * height / heatmaps.shape[1]))
        cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    # Analisi del movimento
    if is_serving(keypoints):
        cv2.putText(frame, "Servizio", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Analisi Servizio", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

    # Visualizzazione nel Jupyter Notebook
    clear_output(wait=True)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cap.release()
out.release()
cv2.destroyAllWindows()
