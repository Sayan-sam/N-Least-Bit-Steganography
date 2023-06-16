from Crypto import Steganography
import matplotlib.pyplot as plt
import numpy as np
import cv2

crypt = Steganography()
result = crypt.encode_video("videos/vid.mp4", "images/main_secret", 2)
# plt.imshow(result[0])
# plt.show()
extract = crypt.decode_video(result, "images/output_secret" ,2)
# plt.imshow(extract)
# plt.show()s

# plt.imshow(cv2.cvtColor(cv2.imread("images/main/Cyber Pic.jpg"), cv2.COLOR_BGR2RGB))
# plt.show()

# vid = cv2.VideoCapture("videos/video.mp4")

# ret, frame = vid.read()

# plt.imshow(frame)
# plt.show()


# print(cv2.imread("images/output/output_0.jpg").flatten()[0:25])

# print(cv2.imread("images/main/batman Gray.jpg").flatten()[0:25])

