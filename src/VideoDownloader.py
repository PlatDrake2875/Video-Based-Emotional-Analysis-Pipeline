import os
import yt_dlp as youtube_dl
import cv2

class VideoDownloader:
    def __init__(self, url, resolution=(1280, 720), download_dir='videos'):
        self.url = url
        self.resolution = resolution
        self.download_dir = download_dir
        self.output_path = None

    def _get_video_name(self):
        # Define the output template
        output_template = '[%(id)s] %(title)s.%(ext)s'

        # Extract video information without downloading
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]/best[ext=mp4][height<=360]',
            'outtmpl': output_template,
            'noplaylist': True,
            'skip_download': True, 
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.url, download=False)
            # Prepare the filename based on the extracted info
            filename = ydl.prepare_filename(info_dict)

        # Combine the download directory with the filename
        self.output_path = os.path.join(self.download_dir, filename)
        return self.output_path

    def download_video(self):
        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)

        # Get the expected output path
        self.output_path = self._get_video_name()

        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]/best[ext=mp4][height<=360]',
            'outtmpl': self.output_path,
            'noplaylist': True,
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

        return self.output_path

    def resize_video(self, output_filename):
        cap = cv2.VideoCapture(self.output_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.output_path}")

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        target_width, target_height = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.download_dir, output_filename)
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (target_width, target_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out.write(resized_frame)

        cap.release()
        out.release()
        print(f"Resized video saved as {output_path}")
