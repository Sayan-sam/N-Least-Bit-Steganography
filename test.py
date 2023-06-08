from Crypto import Steganography
import matplotlib.pyplot as plt

crypt = Steganography()
result = crypt.image_in_image_encode("high_res.jpg", "small_img.jpg")
plt.imshow(result[0])
plt.show()