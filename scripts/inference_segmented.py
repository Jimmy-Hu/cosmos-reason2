

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
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# Constants
ROOT = os.path.dirname(os.path.abspath(__file__))
# Sample rate (frames per second) for the model
