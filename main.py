import sys
sys.path.append('./src')
from VideoProcessor import VideoProcessor

def main(video_url):
    processor = VideoProcessor(video_url)
    processor.process_video()
    processor.make_demo()

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=MTWkfpa-jJw"
    main(video_url)
