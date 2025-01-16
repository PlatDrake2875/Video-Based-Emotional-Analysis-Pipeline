import os
import subprocess
from tqdm import tqdm

class FrameExtractor:
    def __init__(self, video_path, frame_interval=15, base_dir="saves"):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.base_dir = base_dir
        self.frames_dir = self._create_frames_dir()

    def _create_frames_dir(self):
        video_filename = os.path.basename(self.video_path)
        video_name, _ = os.path.splitext(video_filename)
        frames_dir = os.path.join(self.base_dir, video_name)
        os.makedirs(frames_dir, exist_ok=True)
        return frames_dir

    def extract_frames(self):
        fps = self._get_video_fps()
        if fps is None:
            raise ValueError(f"Unable to retrieve FPS for video file: {self.video_path}")
        frame_rate = fps / self.frame_interval

        # Construct the ffmpeg command to extract frames
        output_pattern = os.path.join(self.frames_dir, "frame_%04d.jpg")
        command = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-i", self.video_path,
            "-vf", f"fps={frame_rate}",
            output_pattern
        ]

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg command failed with error: {e}")

        # Rename frames to include timestamps
        self._rename_frames_with_timestamps()

        # Collect the paths of the renamed frames
        frame_paths = sorted(
            os.path.join(self.frames_dir, fname)
            for fname in os.listdir(self.frames_dir)
            if fname.endswith(".jpg")
        )
        return frame_paths

    def _get_video_fps(self):
        # Use ffprobe to get the frames per second (fps) of the video
        command = [
            "ffprobe",
            "-loglevel", "quiet",
            "-v", "error",
            "-select_streams", "v:0",
            "-print_format", "json",
            "-show_entries", "stream=r_frame_rate",
            self.video_path
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            r_frame_rate = info['streams'][0]['r_frame_rate']
            num, denom = map(int, r_frame_rate.split('/'))
            return num / denom
        except (subprocess.CalledProcessError, KeyError, ValueError, IndexError, json.JSONDecodeError):
            return None

    def _rename_frames_with_timestamps(self):
        # Get the duration of the video in seconds
        duration = self._get_video_duration()
        if duration is None:
            raise ValueError(f"Unable to retrieve duration for video file: {self.video_path}")

        # Calculate the time interval between extracted frames
        time_interval = duration / len(os.listdir(self.frames_dir))

        # Rename each frame to include the timestamp
        for index, filename in enumerate(sorted(os.listdir(self.frames_dir))):
            if filename.endswith(".jpg"):
                timestamp = self._format_timestamp(index * time_interval)
                new_filename = f"frame_{index:04d}_{timestamp}.jpg"
                os.rename(
                    os.path.join(self.frames_dir, filename),
                    os.path.join(self.frames_dir, new_filename)
                )

    def _get_video_duration(self):
        # Use ffprobe to get the duration of the video in seconds
        command = [
            "ffprobe",
            "-loglevel", "quiet",
            "-v", "error",
            "-select_streams", "v:0",
            "-print_format", "json",
            "-show_entries", "format=duration",
            self.video_path
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            return float(info['format']['duration'])
        except (subprocess.CalledProcessError, KeyError, ValueError, IndexError, json.JSONDecodeError):
            return None

    def _format_timestamp(self, seconds):
        """Convert seconds to a formatted string HH-MM-SS.sss"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}-{int(minutes):02d}-{seconds:06.3f}"
