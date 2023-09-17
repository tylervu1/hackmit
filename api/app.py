#!/usr/bin/env python

# ----------------------------------------------------------------------
# app.py
# Author: Tyler Vu
# ----------------------------------------------------------------------

import subprocess
import os
import shutil

#-----------------------------------------------------------------------

def process_uploaded_file(filename):
    try:
        print(f"Processing file: {filename}")

        folder_name, _ = os.path.splitext(filename)
        
        # Create the folder with folder_name
        base_path = "C:/Users/waddl/Documents/hackmit/uploads"
        folder_path = os.path.join(base_path, folder_name)
        original_file_path = os.path.join(base_path, filename)
        
        print(f"Creating folder at: {folder_path}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print(f"Folder {folder_path} already exists.")

        # Copy the original video to the new folder
        new_file_path = os.path.join(folder_path, filename)
        shutil.copy(original_file_path, new_file_path)

        # Debugging: List contents of the base directory
        print("Contents of base directory:", os.listdir(base_path))

        # First colmap2nerf.py command
        cmd1 = ["python", "../../instant-ngp/scripts/colmap2nerf.py",
                "--video_in", f"{filename}", "--video_fps", "2",
                "--run_colmap", "--overwrite"]
        
        print(f"Running command: {' '.join(cmd1)}")
        result = subprocess.run(cmd1, cwd=folder_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Command 1 failed with error {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Second colmap2nerf.py command
        cmd2 = ["python", "../../instant-ngp/scripts/colmap2nerf.py",
                "--colmap_matcher", "exhaustive", "--run_colmap",
                "--aabb_scale", "16", "--overwrite"]
        
        print(f"Running command: {' '.join(cmd2)}")
        result = subprocess.run(cmd2, cwd=folder_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Command 2 failed with error {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Call start_instant_ngp function
        start_instant_ngp(folder_name)
        
    except Exception as e:
        print(f"An error occurred: {e}")


# start training model w/ngp
def start_instant_ngp(folder_name):
    # Switch to the target directory
    os.chdir(r'C:/Users/waddl/Documents/hackmit/instant-ngp')
    
    # Build the command
    command = f'start ./instant-ngp.exe ../uploads/{folder_name}'

    print(command)
    
    # Run the command
    subprocess.run(command, shell=True)