from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("runs/detect/train3/weights/best.pt")

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=2
)

fcc = cv2.VideoWriter_fourcc(*'XVID')
size = (1920, 1080)
video_number = 5
path = f"media/helmet-good/{video_number}"
video_output = cv2.VideoWriter(f"{path}.avi", fcc, 60, size)
cap = cv2.VideoCapture(f"{path}.mp4")

while True:
    ret, frame = cap.read()

    if ret:
        result = model(frame, agnostic_nms=True)[0]
        
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.confidence >= 0.5]

        labels = [f"{model.model.names[class_id]}" for _, _, _, class_id, _ in detections]

        for i in range(len(labels)):
            if labels[i] == 'head':
                 labels[i] = 'no_helmet'

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        video_output.write(frame)
        cv2.imshow("Feed", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
video_output.release()
cap.release()
cv2.destroyAllWindows()