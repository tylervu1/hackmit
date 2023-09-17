import cv2
import numpy as np
import os

import json
import sys
import math
from pathlib import Path, PurePosixPath

# Utils
def fix_image_size(image: numpy.array, expected_pixels: float = 2E6):
    ratio = numpy.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: numpy.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def video_to_frames(video_path):
  folder_names = ["images", "blurry"]
  if not os.path.exists(folder_names[0]):
    os.makedirs(folder_names[0])
    print(f"Folder '{folder_names[0]}' created successfully.")
  else:
    print(f"Folder '{folder_names[0]}' already exists.")
    return 0

  os.makedirs(folder_names[1])

  # Generate frames and moves blurry ones to another folder.
  frames = []
  cap = cv2.VideoCapture(video_path)
  if cap.isOpened():
      current_frame = 0
      while True:
          ret, frame = cap.read()
          if ret:
              name = f'/frame{current_frame}.jpg'
              cv2.imwrite(name, frame)
              if is_blurry(cv2.imread(str(name))) == False:
                os.remove(name)
                name = f'/images/frame{current_frame}.jpg'
                # print(f"Creating file... {name}")
                cv2.imwrite(name, frame)
                frames.append(name)
              else:
                os.remove(name)
                name = f'/blurry/frame{current_frame}.jpg'
                cv2.imwrite(name, frame)
          current_frame += 1
          if (current_frame == 268):
            return frames
      cap.release()
  cv2.destroyAllWindows()
  return frames


def is_blurry(frame):
    print(estimate_blur(frame, threshold=10)[1])
    return estimate_blur(frame, threshold=10)[2]

# Tests
def blurry_eval_test():
    # Testing with some sample images.
    image = cv2.imread(str("/testing/blurry.png"))
    print(estimate_blur(image, threshold=100))

def transformers_json(TEXT_FOLDER, IMAGE_FOLDER, SCRIPTS_FOLDER, OUTPUT_PATH):
    cameras = {}
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        camera_angle_x = math.pi / 2
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera

    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        if len(cameras) == 1:
            camera = cameras[camera_id]
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": AABB_SCALE,
                "frames": [],
            }
        else:
            out = {
                "frames": [],
                "aabb_scale": AABB_SCALE
            }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY*2:
                continue
            if i % 2 == 1:
                elems = line.split(" ")
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not args.keep_colmap_coords:
                    c2w[0:3, 2] *= -1
                    c2w[0:3, 1] *= -1
                    c2w = c2w[[1, 0, 2, 3], :]
                    c2w[2, :] *= -1

                    up += c2w[0:3, 1]

                frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                out["frames"].append(frame)
    nframes = len(out["frames"])

    if args.keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)
    else:
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    if len(args.mask_categories) > 0:
        try:
            import detectron2
        except ModuleNotFoundError:
            try:
                import torch
            except ModuleNotFoundError:
                print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
                sys.exit(1)

            input("Detectron2 is not installed. Press enter to install it.")
            import subprocess
            package = 'git+https://github.com/facebookresearch/detectron2.git'
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            import detectron2

        import torch
        from pathlib import Path
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        category2id = json.load(open(SCRIPTS_FOLDER / "category2id.json", "r"))
        mask_ids = [category2id[c] for c in args.mask_categories]

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        for frame in out["frames"]:
            img = cv2.imread(frame["file_path"])
            outputs = predictor(img)

            output_mask = np.zeros((img.shape[0], img.shape[1]))
            for i in range(len(outputs["instances"])):
                if outputs["instances"][i].pred_classes.cpu().numpy()[0] in mask_ids:
                    pred_mask = outputs["instances"][i].pred_masks.cpu().numpy()[0]
                    output_mask = np.logical_or(output_mask, pred_mask)

            rgb_path = Path(frame["file_path"])
            mask_name = str(rgb_path.parents[0] / Path("dynamic_mask_" + rgb_path.name.replace(".jpg", ".png")))
            cv2.imwrite(mask_name, (output_mask*255).astype(np.uint8))
   
if __name__ == "__main__":
    # Turn video into frames
    frames = video_to_frames("/testing/test_video.mp4")
    print(str(len(frames)) + " frames created.")
    print(frames)

    # This needs work, not quite sure where the first input is in repo.
    transformers_json("colmap_text", "/images", "/pre-processing", "/pre-processing/transformers.json")
