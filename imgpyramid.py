from imgutils import *
from boxutils import *


class ImagePyramid(object):
    def __init__(self, img, wnd_size, min_wnd_size, coef=0.5):
        self._min_wnd_size = min_wnd_size
        self._wnd_size = wnd_size
        self._img = img
        self._imgs = []
        width = img.size[0]
        height = img.size[1]
        minsize = min(width, height)
        cf = wnd_size / minsize
        while 1 / cf >= 1.0 and wnd_size / cf >= min_wnd_size:
            self._imgs.append((1 / cf, self._resize(img, (round(width * cf), round(height * cf)))))
            cf /= coef
        if cf > 1.0:
            self._imgs.append((1.0, img))

    def arrays(self, sizes, stride):
        if len(sizes) == 1:
            for coef, img, b in self._boxes(self._wnd_size, stride):
                yield coef, box_mul(b, coef), img_to_arr_bytes(self._resize(img.crop(b), (sizes[0], sizes[0])))
        else:
            for coef, img, b in self._boxes(self._wnd_size, stride):
                wnd = img.crop(b)
                arrs = []
                for s in sizes:
                    arrs.append(img_to_arr_bytes(self._resize(wnd, (s, s))))
                yield (coef, box_mul(b, coef)) + tuple(arrs)

    def arrays_gen(self, stride):
        for coef, img, b in self._boxes(self._wnd_size, stride):
            yield (coef, box_mul(b, coef), img.crop(b))

    def _boxes(self, size, stride):
        if isinstance(stride, (tuple, list)):
            stridex = stride[0]
            stridey = stride[1]
        else:
            stridex = stride
            stridey = stride

        if isinstance(size, (tuple, list)):
            sizex = size[0]
            sizey = size[1]
        else:
            sizex = size
            sizey = size

        for coef, img in self._imgs:
            imgw = img.size[0]
            imgh = img.size[1]
            for b in self._boxes_img(sizex, sizey, stridex, stridey, img.size[0], img.size[1]):
                yield coef, img, b

    def _boxes_img(self, sizex, sizey, stridex, stridey, imgw, imgh):
        y = 0
        while y + sizey <= imgh:
            x = 0
            while x + sizex <= imgw:
                yield (x, y, x + sizex, y + sizey)
                x += stridex

            if x - stridex + sizex != imgw:
                yield (imgw - sizex, y, imgw, y + sizey)
            y += stridey

        if y - stridey + sizey != imgh:
            x = 0
            while x + sizex <= imgw:
                yield (x, imgh - sizey, x + sizex, imgh)
                x += stridex

            if x - stridex + sizex != imgw:
                yield (imgw - sizex, imgh - sizey, imgw, imgh)

    def _resize(self, img, size):
        return img.resize(size, Image.ANTIALIAS)
