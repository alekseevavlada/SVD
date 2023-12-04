from PIL import Image
from numpy import asarray

image = Image.open('gr_img.jpg')

data = asarray(image)

image2 = Image.fromarray(data)

print("Класс изображения:", type(image2))
print("Мод:", image2.mode)
print("Размерность матрицы A:", image2.size)
print("Матрица A:", data)
