
import matplotlib.pyplot as plt
from matplotlib.image import imread


# 이미지 읽기는 잘 안되네 ㅠ
img = imread('//Users//jarvis//code//workspace//learn_ml_bottom//01_hellopython//matplotlib//me.jpeg')
plt.imshow(img)
plt.show()