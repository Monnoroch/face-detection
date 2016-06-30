from PIL import Image
import numpy as np
import theano


def img_to_arr_bytes(img):
    data = np.array(list(img.tobytes()), dtype=np.uint8).reshape((img.size[0], img.size[1], 3))
    return np.swapaxes(np.swapaxes(data, 2, 1), 1, 0)

def img_to_arr(img):
    return np.array([img_to_arr_bytes(img)]).astype(theano.config.floatX)

def im_to_ocv(img):
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def arr_to_img(arr):
    tmp = np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)
    return Image.frombytes("RGB", (len(tmp), len(tmp[0])), bytes(tmp.reshape(-1).tolist()))
