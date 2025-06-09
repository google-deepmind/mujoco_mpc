import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, detect_model_path='yolov8s.pt', seg_model_path='yolov8s-seg.pt'):
        self.detect_model = YOLO(detect_model_path)
        self.seg_model = YOLO(seg_model_path)

    def detect_objects(self, image):
        results = self.detect_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = [results.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        return boxes, labels

    def annotate_image(self, image, boxes, labels):
        annotated = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[i]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

    def segment_floor(self, image):
        # yolov8s-seg 모델을 사용하여 segmentation 수행
        results = self.seg_model(image)[0]
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        cls_ids = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        names = results.names

        # 'floor' 또는 관련된 label을 가진 mask 추출
        floor_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, cls_id in enumerate(cls_ids):
            label = names[int(cls_id)].lower()
            if 'floor' in label or 'rug' in label or 'carpet' in label:
                mask = (masks[i] > 0.5).astype(np.uint8) * 255
                floor_mask = np.maximum(floor_mask, mask)

        return floor_mask
