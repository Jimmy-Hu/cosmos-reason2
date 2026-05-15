

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
