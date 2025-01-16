import os
import cv2
import pandas as pd
import ast
from datetime import datetime, timedelta
from tqdm import tqdm

from FrameExtractor import FrameExtractor
from VideoAnalysis import VideoAnalysis
from VideoDownloader import VideoDownloader  

class VideoEmotionProcessor:
    def __init__(self, video_url):
        self.video_url = video_url
        self.video_path = VideoDownloader(self.video_url)._get_video_name()
        self.frames_path = FrameExtractor(self.video_path).frames_dir
        
        self.helper_path = self.frames_path.split('\\')[1]
        self.csv_path = os.path.join("output", f"{self.helper_path}.csv")

        self.output_path = os.path.join('demos', os.path.basename(self.video_path))
        
        # Ensure the demos directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def parse_timestamp(self, timestamp):
        """Parse timestamp in format HH-MM-SS.sss to seconds."""
        parts = list(map(float, timestamp.split('-')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0

    def extract_confidence(self, confidence_str, emotion):
        """Extract the confidence value for the given emotion from the dictionary string."""
        confidence_dict = ast.literal_eval(confidence_str)
        return confidence_dict[emotion]

    def overlay_emotion(self, frame, emotions_confidences, dominant_emotion, extra_space):
        """Overlay all emotions and their confidences outside the video frame width."""
        height, width, _ = frame.shape

        # Add an additional margin to the right of the video
        frame_with_margin = cv2.copyMakeBorder(
            frame, 0, 0, 0, extra_space, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        text_x = width + 10  # Start text outside the original video width
        y_offset = 30

        for emotion, confidence in emotions_confidences.items():
            text = f"{emotion}: {confidence:.2f}%"
            color = (0, 255, 0) if emotion == dominant_emotion else (255, 255, 255)
            cv2.putText(frame_with_margin, text, (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y_offset += 30

        return frame_with_margin

    def process_video(self):
        # Read CSV data
        df = pd.read_csv(self.csv_path)
        df['Timestamp'] = df['Timestamp'].apply(self.parse_timestamp)

        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        extra_space = 200  # Extra space for text display

        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width + extra_space, frame_height))

        # Process each frame with progress bar
        frame_idx = 0
        with tqdm(total=frame_count, desc="Applying emotions to video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_idx / fps
                closest_row = df.iloc[(df['Timestamp'] - current_time).abs().argsort()[:1]]
                if not closest_row.empty:
                    confidence_str = closest_row['Confidence'].values[0]
                    confidence_dict = ast.literal_eval(confidence_str)
                    dominant_emotion = closest_row['Emotion'].values[0]
                    frame = self.overlay_emotion(frame, confidence_dict, dominant_emotion, extra_space)

                out.write(frame)  # Write the processed frame to the output video
                frame_idx += 1
                pbar.update(1)  # Update progress bar

        # Release resources
        cap.release()
        out.release()

    def display_video(self):
        # Open and display the processed video
        cap = cv2.VideoCapture(self.output_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
