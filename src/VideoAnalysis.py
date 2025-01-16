import os
import csv
import cv2
from tqdm import tqdm
from deepface import DeepFace
from ultralytics import YOLO

class VideoAnalysis:
    def __init__(self, person_model_path='models/yolo11s.pt'):
        self.emotion_model = DeepFace
        self.person_model = YOLO(person_model_path, verbose=False)  # Set verbose to False to suppress output

    def process_frames(self, frame_paths):
        output_path = self._prepare_output_path(frame_paths)
        print(f"Output CSV will be saved to: {output_path}")
        self._write_csv_header(output_path)

        for frame_path in tqdm(frame_paths, desc="Inference in progress"):
            frame = self._load_frame(frame_path)
            timestamp = self._extract_timestamp_from_filename(frame_path)
            person_bboxes = self._detect_people(frame)
            self._process_person_bboxes(frame, person_bboxes, timestamp, output_path)

    def _prepare_output_path(self, frame_paths):
        first_frame_path = frame_paths[0]
        video_name = self._extract_video_name(first_frame_path)
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f'{video_name}.csv')

    def _extract_video_name(self, frame_path):
        directory_name = os.path.basename(os.path.dirname(frame_path))
        return directory_name

    def _write_csv_header(self, output_path):
        with open(output_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Timestamp', 'Emotion', 'Confidence'])

    def _load_frame(self, frame_path):
        return cv2.imread(frame_path)

    def _extract_timestamp_from_filename(self, filename):
        base_name = os.path.basename(filename)
        timestamp_str = base_name.split('_')[2].replace('.jpg', '')
        return timestamp_str

    def _detect_people(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.person_model(rgb_frame, verbose=False)  # Add verbose=False to suppress output
        return [
            box.xyxy[0].tolist()
            for result in results
            for box in result.boxes
            if int(box.cls[0]) == 0  # Class '0' corresponds to 'person'
        ]

    def _process_person_bboxes(self, frame, person_bboxes, timestamp, output_path):
        for bbox in person_bboxes:
            cropped_frame = self._crop_frame(frame, bbox)
            emotion, confidence = self._detect_emotion(cropped_frame)
            self._append_to_csv(output_path, [timestamp, emotion, confidence])

    def _crop_frame(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return frame[y1:y2, x1:x2]

    def _detect_emotion(self, frame):
        analysis = self.emotion_model.analyze(
            frame, actions=['emotion'], detector_backend='skip', enforce_detection=False, silent=True  # Add silent=True to suppress output
        )
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_confidence = {emotion: float(conf) for emotion, conf in analysis[0]['emotion'].items()}
        return dominant_emotion, emotion_confidence

    def _append_to_csv(self, output_path, row):
        with open(output_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(row)
