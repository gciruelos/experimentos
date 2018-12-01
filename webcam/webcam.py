import cv2
import numpy as np
from PIL import Image
from sys import argv

# SQUARE_SIDE_BILLETE es el ancho y el alto de cada cuadradito en el que se
# recortará al billete.
SQUARE_SIDE_BILLETE = 20
# M es en cuanto se debe agrandar la imagen original (por ejemplo, si la webcam
# captura a 640x480 y M es 2, la imagen resultante va a ser 1280x960.
# SQUARE_SIDE_BILLETE DEBE SER DIVISIBLE POR M!
M = 2


# https://en.wikipedia.org/wiki/Luma_(video)#Use_of_relative_luminance
def luma(r,g,b):
    return .2126 * r + .7152 * g + .0722 * b


def cuadrados_por_luminancia(img, square_side):
    res = {}
    for i_ in range(0, img.shape[0] - (square_side - 1), square_side):
        for j_ in range(0, img.shape[1] - (square_side - 1), square_side):
            chiquito = np.zeros((square_side, square_side,3), dtype=np.uint8)
            lumas = []
            for i in range(square_side):
                for j in range(square_side):
                    p = img[i + i_][j + j_]
                    chiquito[i][j] = p
                    lumas.append(luma(p[0], p[1], p[2]))
            median_luma = sorted(lumas)[len(lumas) // 2]
            # TODO: potencial colision
            res[int(median_luma)] = chiquito
    for possible_luma in range(256):
        if possible_luma not in res.keys():
            best_diff = 256
            best_match = None
            for l in res.keys():
                if abs(l - possible_luma) < best_diff:
                    best_diff = abs(l - possible_luma)
                    best_match = res[l]
            res[possible_luma] = best_match

    return res

def generar_imagen(img, square_side, dict_lumas):
    N1, N2, N3 = img.shape
    nsq = square_side // M
    result = np.zeros((N1 * M, N2 * M, N3), dtype=np.uint8)
    for i in range(0, img.shape[0] - (nsq - 1), nsq):
        for j in range(0, img.shape[1] - (nsq - 1), nsq):
            pixel_luma = luma(img[i][j][0], img[i][j][1], img[i][j][2])
            result[i*M:(i+nsq)*M,j*M:(j+nsq)*M] = dict_lumas[int(pixel_luma)]

    return result

# (0, 0)  (0, 5) (0, 10)    (0,  0) (0,  20) (0,  40)
# (5, 0)  (5, 5) (5, 10)    (20, 0) (20, 20) (20, 40)


def buscar_pedazo(dict_lumas, pixel_luma):
    best_diff = 255.0
    best_match = None
    for l in dict_lumas.keys():
        if abs(l - pixel_luma) < best_diff:
            best_diff = abs(l - pixel_luma)
            best_match = dict_lumas[l]
    return best_match


video_capture = cv2.VideoCapture(0)

im_billete = np.asarray(Image.open(argv[1]).convert('RGB'))

r = None
while True:
    # Capture frame-by-frame
    if r is None:
        ret, frame = video_capture.read()
        r = generar_imagen(frame, SQUARE_SIDE_BILLETE, cuadrados_por_luminancia(im_billete, SQUARE_SIDE_BILLETE))

    # Display the resulting frame
    cv2.imshow('Video', r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

Image.fromarray(r).save("resultado.modificada.png")
