import os
import csv
import shutil
import urllib.request
from pathlib import Path
import cv2
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

TRAJECTORIES = ["traj_02", "traj_03", "traj_04", "traj_05", "traj_06"]
BASE_URL = "https://huggingface.co/datasets/ntnu-arl/underwater-datasets/resolve/main/mclab-2023-multicam/subset-3-mocap-gt"
OUTPUT_BASE = Path("processed_data")
TOPIC_CAM = "/alphasense_driver_ros/cam0"

for traj in TRAJECTORIES:
    out_dir = OUTPUT_BASE / traj
    img_dir = out_dir / "cam0_images"

    if (out_dir / "images.csv").exists() and img_dir.exists():
        print(f"[{traj}] already extracted. Skipping...")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    bag_path = out_dir / f"{traj}.bag"
    tum_path = out_dir / f"{traj}_mocap.tum"

    # 1. Download MoCap Ground Truth
    if not tum_path.exists():
        print(f"\nDownloading ground truth for {traj}...")
        urllib.request.urlretrieve(f"{BASE_URL}/{traj}/{traj}_mocap.tum", tum_path)

    # 2. Download ROS Bag
    if not bag_path.exists():
        print(f"Downloading {traj}.bag...")
        urllib.request.urlretrieve(f"{BASE_URL}/{traj}/{traj}.bag", bag_path)

    # 3. Extract only cam0 images
    print(f"Extracting cam0 images from {traj}.bag...")
    img_dir.mkdir(parents=True, exist_ok=True)

    with AnyReader([bag_path]) as reader:
        connections = [x for x in reader.connections if x.topic == TOPIC_CAM]
        if connections:
            with open(out_dir / "images.csv", "w", newline="") as img_csv:
                cam_writer = csv.writer(img_csv)
                cam_writer.writerow(["timestamp_sec", "filename"])

                total_msgs = sum(c.msgcount for c in connections)
                msg_gen = reader.messages(connections=connections)

                for conn, ts_ns, rawdata in tqdm(msg_gen, total=total_msgs, desc=f"Extracting {traj}"):
                    msg = reader.deserialize(rawdata, conn.msgtype)
                    t_sec = ts_ns * 1e-9
                    img = message_to_cvimage(msg)
                    img_name = f"{ts_ns}.jpg"
                    cv2.imwrite(str(img_dir / img_name), img)
                    cam_writer.writerow([f"{t_sec:.6f}", img_name])

    # 4. Remove heavy .bag file to save disk space
    if bag_path.exists():
        print(f"Cleaning up {bag_path.name} to free disk space...")
        os.remove(bag_path)

print("\nAll trajectories downloaded, extracted, and cleaned up successfully!")
