import cv2
import numpy as np
from matplotlib import pyplot as plt

def show(image, dpi, title):
  plt.figure(dpi=dpi)
  plt.title(title)
  plt.imshow(image, cmap='gray')
  plt.show()
  plt.close()

def videooooooo(video):
  cap = cv2.VideoCapture(video)
  frames = []

  accumulated_hist_b = np.zeros((256,), dtype=np.float64)
  accumulated_hist_g = np.zeros((256,), dtype=np.float64)
  accumulated_hist_r = np.zeros((256,), dtype=np.float64)
  frame_count = 0

  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      print("Reached the end of the video or cannot read the frame.")

    # Store the frame in the list
    frames.append(frame)

    # Split the frame into its color channels
    B, G, R = cv2.split(frame)

    # Calculate histograms for each channel
    hist_b = cv2.calcHist([B], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([G], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([R], [0], None, [256], [0, 256])

    # Accumulate the histograms
    accumulated_hist_b += hist_b.flatten()
    accumulated_hist_g += hist_g.flatten()
    accumulated_hist_r += hist_r.flatten()

    # Increment the frame count
    frame_count += 1

    # Release video capture
    cap.release()

    # Calculate the average histograms
    average_hist_b = accumulated_hist_b / frame_count
    average_hist_g = accumulated_hist_g / frame_count
    average_hist_r = accumulated_hist_r / frame_count

    plt.figure(figsize=(10, 5))
    plt.plot(average_hist_b, color='b', label='Blue')
    plt.plot(average_hist_g, color='g', label='Green')
    plt.plot(average_hist_r, color='r', label='Red')
    plt.title('Average Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid()
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  plt.show()
  plt.close()
  cv2.destroyAllWindows()