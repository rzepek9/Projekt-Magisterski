from pathlib import Path

import cv2
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor


class DetrForBasketballDetection:
    def __init__(
        self,
        model_checkpoint,
        processor_checkpoint,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = DetrImageProcessor.from_pretrained(processor_checkpoint)
        self.model = DetrForObjectDetection.from_pretrained(model_checkpoint).to(
            self.device
        )
        self.id2label = {1: "ball", 2: "basket"}
        self.box_annotator = sv.BoxAnnotator()

    def get_detections(self, frame, confidence_threshold=0.85, iou_threshold=0.8):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
            results = self.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes,
            )[0]
        detections = sv.Detections.from_transformers(
            transformers_results=results
        ).with_nms(iou_threshold)
        return detections

    def plot_detections(self, frame, detections: sv.Detections, only_ball=False) -> None:
        """
        Plots detected basketball objects on a frame based
        on supervision Detections
        """
        if only_ball:
            ball_class_id = [k for k, v in self.id2label.items() if v == "ball"][0]
            detections = sv.Detections(
                xyxy=detections.xyxy[detections.class_id == ball_class_id],
                class_id=detections.class_id[detections.class_id == ball_class_id],
                confidence=detections.confidence[detections.class_id == ball_class_id],
            )
        labels = [
            f"{self.id2label[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections, labels=labels
        )
        return annotated_frame

    def detect_on_video(
        self,
        video_path,
        output_path=None,
        display=False,
        save_video=True,
        confidence_threshold=0.85,
        iou_threshold=0.8,
        only_ball=False,
    ):
        if isinstance(video_path, Path):
            video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))
            )

        progress_bar = tqdm(
            total=total_frames, desc="Processing video", ncols=80
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = resize_frame(frame)
                detections = self.get_detections(frame, confidence_threshold, iou_threshold)
                annotated_frame = self.plot_detections(frame, detections, only_ball)
                if save_video:
                    out.write(annotated_frame)
                if display:
                    cv2.imshow("frame", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                progress_bar.update()
            else:
                break

        progress_bar.close()

        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()


def resize_frame(frame, target_width=640, target_height=480):
    """
    Resize the frame to the target width and height while maintaining the aspect ratio.
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_width = min(target_width, int(target_height * aspect_ratio))
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height))
