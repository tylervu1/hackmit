import cv2
import numpy
import os

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


if __name__ == "__main__":
    # Turn video into frames
    frames = video_to_frames("/testing/test_video.mp4")
    print(str(len(frames)) + " frames created.")
    print(frames)
