import cv2
import numpy as np
from PIL import Image
import math

# np.set_printoptions(threshold=np.inf)


def get_u(v, sigma):
    u = []
    for i in range(len(v)):
        ui = (1 / sigma[i]) * np.dot(A_float64, v[:, i])
        u.append(ui)

    return np.transpose(np.array(u))


def get_v(u, sigma):
    v = []
    for i in range(len(u)):
        vi = (1 / sigma[i]) * np.dot(A_float64.transpose(), u[:, i])
        v.append(vi)

    return np.transpose(np.array(v))


def recreate_image(sigma, u, v, k):
    B = np.zeros((m, n))
    A_SVD = np.zeros((m, n), dtype=np.uint8)

    for i in range(k):
        B = np.add(B, sigma[i] * np.outer(u[:, i], v[:, i]))

    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] > 255:
                A_SVD[i][j] = 255
            elif B[i][j] < 0:
                A_SVD[i][j] = 0
            else:
                A_SVD[i][j] = B[i][j]

    return A_SVD


def get_error(A, A_SVD):
    B = A - A_SVD
    return math.sqrt(np.dot(B.transpose(), B).trace())


A = cv2.imread('img.jpg')
A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
A_float64 = np.array(A, dtype=np.float32)

k = 500

m = np.shape(A_float64)[0]
n = np.shape(A_float64)[1]

# print("Количество строк матрицы A = ", m)
# print("Количество столбцов матрицы A = ", n)

if m > n:
    AtA = np.dot(A_float64.transpose(), A_float64)
    sigma, v = np.linalg.eig(AtA)
    u = get_u(v, sigma)

else:
    AAt = np.dot(A_float64, A_float64.transpose())
    sigma, u = np.linalg.eig(AAt)
    v = get_v(u, sigma)


# print("Матрица A: ", A)
# print("Сингулярные числа: ", sigma)
# print(f"Левые U-вектора = {u[0], u[1], u[3], u[4], u[5], u[6], u[7], u[8], u[9]}")
# print("Количесвто U-векторов: ", len(u))
# print("Правые V-вектора:", v)
# print("Количество V-векторов: ", len(v))

A_SVD = recreate_image(sigma, u, v, k)
error = get_error(A_float64, A_SVD)
# print("Степень сжатия (PSNR) при k =", k, "составляет", error)

cv2.imshow("Original", A)
cv2.imshow("Compression using SVD", A_SVD)

Image.fromarray(A_SVD).save('k_img.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
