import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# np.set_printoptions(threshold=np.inf)

MAX_RANK = 830

image = Image.open('img.jpg').convert("L")
img_mat = np.asarray(image)

Image.fromarray(img_mat).save('gr_img.jpg')

# print("Матрица A:", img_mat)
# print("Размерность A:", img_mat.shape)

U, s, V = np.linalg.svd(img_mat, full_matrices=True)
s = np.diag(s)

# print("Левые U сингулярные вектора:", U)
# print("Диагональная матрица:", s)
# print("Правые V сингулярные вектора:", V)

for k in range(MAX_RANK + 1):
  approx = U[:, :k] @ s[0:k, :k] @ V[:k, :]
  img = plt.imshow(approx, cmap='gray')
  plt.title(f'SVD-разложение при значении k = {k}')
  plt.plot()
  pause_length = 0.0001 if k < MAX_RANK else 5
  plt.pause(pause_length)
  plt.clf()
