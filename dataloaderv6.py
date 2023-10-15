import cv2
import torch
import random
import json
import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# Parameters
batch_size = 16
frame_offset = 9
context_pos = 'middle'
height = 360
width = 640
img_scaling = 0.4
video_dir = 'labeller-training/video/'
actions_dir = 'labeller-training/actions/'

def encode_keypresses(action_dict):
    default_actions = {
        "attack": 0, "back": 0, "drop": 0, "forward": 0, 
        "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0,
        "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0, "inventory": 0,
        "jump": 0, "left": 0, "right": 0, "pickItem": 0, "sneak": 0, "sprint": 0,
        "swapHands": 0, "use": 0
    }
    default_actions.update(action_dict)
    values = [value for value in default_actions.values() if isinstance(value, (int, float))]
    return torch.tensor(values, dtype=torch.int32).cuda()

def encode_camera(action_dict):
    return torch.tensor(action_dict.get("camera", [0.0, 0.0]), dtype=torch.float32).cuda()

def get_random_video_and_actions():
    videos = sorted(os.listdir(video_dir))
    idx = random.randint(0, len(videos) - 1)
    video_path = os.path.join(video_dir, videos[idx])
    action_path = os.path.join(actions_dir, f"actions-{idx}.json")
    with open(action_path, 'r') as json_file:
        actions_data = json.load(json_file)
    return video_path, actions_data

def process_video(cap, start_frame, frame_offset):
    frames = []

    # Set the video position only once
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(frame_offset):
        # start_time = time.time()
        ret, frame = cap.read()
        # print(time.time() - start_time)

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (int(width * img_scaling), int(height * img_scaling)))
            frames.append(torch.tensor(frame).float())

    return torch.stack(frames)


    
def partial_dataset(start, end, video_path, actions_data, total_frames):

    
    batch_frames = []
    batch_keypresses = []
    batch_camera_movements = []

    cap = cv2.VideoCapture(video_path)
    for _ in range(start, end):
        start_frame = random.randint(0, total_frames - frame_offset)
        if context_pos == 'middle':
            start_action = start_frame + (frame_offset // 2)
        else:
            raise ValueError('Choose a valid context position')

        action_dict = actions_data[start_action]
        batch_keypresses.append(encode_keypresses(action_dict))
        batch_camera_movements.append(encode_camera(action_dict))
        batch_frames.append(process_video(cap, start_frame, frame_offset))

    cap.release()

    frames_tensor = torch.stack(batch_frames).cuda().reshape(end-start, frame_offset, int(height * img_scaling), int(width * img_scaling))
    keypresses_tensor = torch.stack(batch_keypresses).cuda()
    camera_tensor = torch.stack(batch_camera_movements).cuda()

    return frames_tensor, keypresses_tensor, camera_tensor

def create_dataset_multithreaded(num_workers=4):
    # start_time = time.time()
    video_path, actions_data = get_random_video_and_actions()
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    
    chunk_size = batch_size // num_workers


    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(partial_dataset, i*chunk_size, (i+1)*chunk_size, video_path, actions_data, total_frames) for i in range(num_workers)]
    
    
    batch_frames = []
    batch_keypresses = []
    batch_camera_movements = []
    for future in futures:
        frames_tensor, keypresses_tensor, camera_tensor = future.result()
        batch_frames.append(frames_tensor)
        batch_keypresses.append(keypresses_tensor)
        batch_camera_movements.append(camera_tensor)

    # end_time = time.time()
    # print(end_time-start_time)


    return torch.cat(batch_frames), torch.cat(batch_keypresses), torch.cat(batch_camera_movements)

# frames, keys, camera = create_dataset()
