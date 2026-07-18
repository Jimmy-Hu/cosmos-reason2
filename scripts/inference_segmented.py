

"""
Segmented inference example for Cosmos-Reason2 to handle long videos.
This script splits the video into chunks (e.g., 60 seconds) and processes them sequentially
to avoid Out-Of-Memory (OOM) errors.
It also saves the generated captions to a text file.

Usage:
    python scripts/inference_segmented.py --video_path /path/to/video.mp4 --output_file captions.txt
    python scripts/inference_segmented.py --model_path "H:/huggingface/Cosmos-Reason2-2B"
"""

import argparse
import gc
import os
import sys
from typing import List

import av
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.generation.streamers import BaseStreamer

# Constants
ROOT = os.path.dirname(os.path.abspath(__file__))
# Sample rate (frames per second) for the model
TARGET_FPS = 4
# Duration of each segment in seconds (1 minute)
SEGMENT_DURATION_SEC = 60

class ProgressStreamer(BaseStreamer):
    """
    Custom Streamer to display token generation progress and ETA using tqdm.
    """
    def __init__(self, max_new_tokens: int):
        # leave=True keeps the progress bar on screen after completion
        self.pbar = tqdm(total=max_new_tokens, desc="    [Debug] Generation Progress", unit="token", leave=True)
        self.is_prompt = True

    def put(self, value):
        # The first call usually contains the input prompt (passed all at once)
        if self.is_prompt:
            self.is_prompt = False
        else:
            # Subsequent calls represent newly generated tokens
            self.pbar.update(1)

    def end(self):
        self.pbar.close()

def get_video_duration(video_path: str) -> float:
    """
    Get the total duration of the video in seconds.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration in seconds.
    """
    try:
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            # Calculate duration based on stream metadata
            if video_stream.duration:
                return float(video_stream.duration * video_stream.time_base)
            else:
                print("Warning: Could not determine video duration from metadata.")
                return 0.0
    except Exception as e:
        print(f"Error checking video duration: {e}")
        return 0.0

def frame_generator(video_path: str, start_sec: float, end_sec: float, fps: int) -> List[Image.Image]:
    """
    Extract frames from a specific time segment of the video.

    Args:
        video_path (str): Path to the video file.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
        fps (int): Frames per second to sample.

    Returns:
        List[Image.Image]: List of PIL images for this segment.
    """
    frames = []
    try:
        print("    [Debug] frame_generator: Opening video container...")
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            
            # Disabled AUTO thread_type. Multi-threading in FFmpeg can cause deadlocks 
            # if the video file has corrupted frames or unusual encoding at specific timestamps.
            # stream.thread_type = 'AUTO' 
            
            # Calculate the time base interval for seeking
            seek_target = int(start_sec / stream.time_base)
            
            print(f"    [Debug] frame_generator: Seeking to target timestamp: {seek_target}...")
            # Seek to the start point (closest keyframe)
            container.seek(seek_target, stream=stream)
            print("    [Debug] frame_generator: Seek completed. Starting to decode frames...")
            
            last_captured_time = -1.0
            min_interval = 1.0 / fps
            frame_count = 0

            for frame in container.decode(stream):
                frame_count += 1
                current_time = float(frame.time)

                if frame_count % 300 == 0:
                     print(f"    [Debug] frame_generator: Still decoding... Current frame time is {current_time:.2f}s")

                # Stop if we passed the end of the segment
                if current_time > end_sec:
                    print(f"    [Debug] frame_generator: Reached segment end ({end_sec}s). Stopping decode.")
                    break
                
                # Skip frames before the actual start time (due to keyframe seeking)
                if current_time < start_sec:
                    continue

                # Capture frame if enough time has passed since last capture
                if last_captured_time < 0 or (current_time - last_captured_time) >= min_interval:
                    pil_image = frame.to_image()
                    frames.append(pil_image)
                    last_captured_time = current_time
                    
            print(f"    [Debug] frame_generator: Successfully finished decoding sequence.")
    except Exception as e:
        print(f"    [Debug] frame_generator Error: {e}")
        
    return frames

def main():
    """
    Main function to run segmented inference.
    """
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run segmented inference on a video.")
    parser.add_argument(
        "--video_path", 
        type=str, 
            torch.cuda.empty_cache()

            # F. End Check
            if is_last_chunk:
                print("\nReached the end of the video stream.")
                break

            current_time = end_time
            chunk_index += 1

    print("\nProcessing complete. Results saved.")
    sys.exit(0)

if __name__ == "__main__":
    main()