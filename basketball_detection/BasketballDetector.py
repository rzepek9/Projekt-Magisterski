import cv2
from ultralytics import YOLO

YOLO_BBALL_CHECKPOINT = "basketball_detection/yolov8_bball.pt"
OUTPUT_FILENAME = "basketball_detection/basketball_detection.avi"
DETECTION_CONFIDENCE = 0.5


class BasketballDetector:
    """
    Class for detecting basketball objects
    """

    def __init__(self, yolo_weights=None, detection_confidence=None):
        self.yolov8 = (
            YOLO(YOLO_BBALL_CHECKPOINT) if not yolo_weights else YOLO(yolo_weights)
        )
        self.confidence = (
            DETECTION_CONFIDENCE if not detection_confidence else detection_confidence
        )

    def get_detections(self, frame):
        return self.yolov8.predict(source=[frame], conf=self.confidence, save=False)

    def are_objects_detected(self, detections):
        return len(detections[0].boxes.boxes.numpy()) != 0

    def plot_detections(self, frame, detections):
        if self.are_objects_detected(detections):
            for i in range(len(detections[0].boxes.boxes.numpy())):
                boxes = detections[0].boxes
                box = boxes[i]
                class_id = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                label = detections[0].names[class_id]
                color = (0, 0, 255)

                cv2.rectangle(
                    frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 3
                )
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    label + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    def detect_on_video(
        self, source, show_video=True, save_video=False, output_filename=None
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            if not output_filename:
                output_filename = OUTPUT_FILENAME
            frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, frame_size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            detections = self.get_detections(frame)
            self.plot_detections(frame, detections)
            if save_video:
                output_video.write(frame)
            if show_video:
                cv2.imshow("Basketball Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if save_video:
            output_video.release()
        cap.release()
        cv2.destroyAllWindows()
