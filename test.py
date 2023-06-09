from Crypto import Steganography
import matplotlib.pyplot as plt
import cv2

crypt = Steganography()
result = crypt.encode_image_dir("images/main", "images/secret", 2)
# plt.imshow(result[0])
# plt.show()
extract = crypt.decode(result, 2)
plt.imshow(extract[0])
plt.show()

# plt.imshow(cv2.cvtColor(cv2.imread("images/main/Cyber Pic.jpg"), cv2.COLOR_BGR2RGB))
# plt.show()

vid = cv2.VideoCapture("videos/video.mp4")

ret, frame = vid.read()

plt.imshow(frame)
plt.show()
