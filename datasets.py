import time
import numpy as np
from boxutils import *
from imgutils import *


class CalClasses(object):
    def __init__(self):
        self.minxs = -0.17
        self.maxxs = 0.17
        self.minys = -0.17
        self.maxys = 0.17
        self.minss = 0.83
        self.maxss = 1.21
        self.stretchx = 1.15 # (1.15 - 1 / 2) / ((1 / 2 + self.maxxs) * self.maxss)
        self.stretchy = 1.15 # (1.15 - 1 / 2) / ((1 / 2 + self.maxys) * self.maxss)
        self.xs = [self.minxs, 0.0, self.maxxs]
        self.ys = [self.minys, 0.0, self.maxys]
        self.ss = [self.minss, 0.91, 1.0, 1.10, self.maxss]

        self.classes = []
        for x in self.xs:
            for y in self.ys:
                for s in self.ss:
                    self.classes.append((x, y, s, len(self.classes)))

    def closest_class(self, orig_box, box):
        ominx, ominy, omaxx, omaxy = orig_box
        minx, miny, maxx, maxy = box

        c1, c2 = self._closest(minx - ominx, self.xs), self._closest(maxx - omaxx, self.xs)
        if c1 != c2:
            cx = self._closest((self.xs[c1] + self.xs[c2]) / 2, self.xs)
        else:
            cx = c1

        c1, c2 = self._closest(miny - ominy, self.ys), self._closest(maxy - omaxy, self.ys)
        if c1 != c2:
            cy = self._closest((self.ys[c1] + self.ys[c2]) / 2, self.ys)
        else:
            cy = c1

        cs = self._closest((maxx - minx) / (omaxx - ominx), self.ss)

        num = 0
        for x in range(len(self.xs)):
            for y in range(len(self.ys)):
                for s in range(len(self.ss)):
                    if x == cx and y == cy and s == cs:
                        break
                    num += 1
        return num

    def bbox_ratio(self):
        return (1 + 2 * self.maxxs) * self.maxss * self.stretchx

    def bbox(self, box):
        minx, miny, maxx, maxy = box

        w = (maxx - minx)
        hw = w / 2
        midx = minx + hw
        dx = max(abs(self.minxs), abs(self.maxxs))
        newhw = (hw + w * dx) * self.maxss * self.stretchx
        newminx = midx - newhw
        newmaxx = midx + newhw

        h = (maxy - miny)
        hh = h / 2
        midy = miny + hh
        dy = max(abs(self.minys), abs(self.maxys))
        newh = (hh + h * dy) * self.maxss * self.stretchy
        newminy = midy - newh
        newmaxy = midy + newh

        return (math.ceil(newminx), math.ceil(newminy), math.ceil(newmaxx), math.ceil(newmaxy))

    def small(self, newbox):
        newminx, newminy, newmaxx, newmaxy = newbox

        neww = (newmaxx - newminx)
        newhw = neww / 2
        midx = newminx + newhw
        dx = max(abs(self.minxs), abs(self.maxxs))
        hw = newhw / ((2 * dx + 1) * self.maxss * self.stretchy)
        minx = midx - hw
        maxx = midx + hw

        newh = (newmaxy - newminy)
        newhh = newh / 2
        midy = newminy + newhh
        dy = max(abs(self.minys), abs(self.maxys))
        hh = newhh / ((2 * dy + 1) * self.maxss * self.stretchy)
        miny = midy - hh
        maxy = midy + hh

        return (math.floor(minx), math.floor(miny), math.floor(maxx), math.floor(maxy))

    def rnd(self):
        return random.choice(self.classes)

    def rnd_cont(self):
        return (random.uniform(self.minxs, self.maxxs), random.uniform(self.minys, self.maxys), random.uniform(self.minss, self.maxss), -1)

    def rnd_stretch(self):
        if random.random() >= 0.5:
            return (random.uniform(1., self.stretchx), 1)
        else:
            return (1, random.uniform(1., self.stretchy))

    def _closest(self, a, bs):
        minv = abs(bs[0] - a)
        mini = 0
        for i in range(len(bs)):
            v = abs(bs[i] - a)
            if v < minv:
                minv = v
                mini = i
        return mini

    def count(self):
        return len(self.classes)


class Config(object):
    def __init__(self, **kvargs):
        self.train = None
        self.mirror = None
        self.shuffle = None
        self.norm = None
        self.move = None
        self.filter = lambda _: True
        self.extend(**kvargs)

    def extend(self, **kvargs):
        self.train = kvargs.get("train", self.train)
        self.mirror = kvargs.get("mirror", self.mirror)
        self.shuffle = kvargs.get("shuffle", self.shuffle)
        self.norm = kvargs.get("norm", self.norm)
        self.move = kvargs.get("move", self.move)
        self.filter = kvargs.get("filter", self.filter)
        return self

    def clone(self):
        return Config(
            mirror=self.mirror,
            shuffle=self.shuffle,
            norm=self.norm,
            move=self.move,
            filter=self.filter,
        )

    def clone_extend(self, **kvargs):
        return self.clone().extend(**kvargs)


class DataSet(object):
    def __init__(self):
        pass

    def iter(self, batchsize, config):
        return []

    def num_classes(self):
        return 0

    def close(self):
        pass

def iterate_minibatches(data, batchsize, config):
    l = len(data[0])
    for x in data:
        assert len(x) == l

    if config.shuffle:
        indices = np.arange(l)
        np.random.shuffle(indices)

    last_end = 0
    for start_idx in range(0, l - batchsize + 1, batchsize):
        last_end = start_idx + batchsize
        if config.shuffle:
            excerpt = indices[start_idx:last_end]
        else:
            excerpt = slice(start_idx, last_end)

        res = map(lambda x: x[excerpt], data)
        if config.mirror:
            res = list(res)
            res = [res[0]] + list(map(np.copy, res[1:]))
            mask = np.random.rand(batchsize) > 0.5
            for x in res[1:]:
                x[mask] = x[mask,:,:,::-1]
        yield tuple(res)

    if last_end < l:
        start_idx = last_end
        if config.shuffle:
            excerpt = indices[start_idx:l]
        else:
            excerpt = slice(start_idx, l)

        res = map(lambda x: x[excerpt], data)
        if config.mirror:
            res = list(res)
            res = [res[0]] + list(map(np.copy, res[1:]))
            mask = np.random.rand(l - start_idx) > 0.5
            for x in res[1:]:
                x[mask] = x[mask,:,:,::-1]
        yield tuple(res)

class SimpleDataSet(DataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None):
        data = np.load(fname)
        labels = data["labels"].astype(np.int32)
        self._num_classes = len(np.unique(labels))
        if test_mode is not None:
            labels = labels[:test_mode]
        split = int(len(labels) * train_perc)
        y_train, y_val = labels[:split], labels[split:]
        if swap:
            y_train, y_val = y_val, y_train

        self._means = []
        self._stds = []

        res = []
        for s in sizes:
            res.append(self._load_images(data, "images" + str(s), split, swap, test_mode))

        self.train = tuple([y_train] + list(map(lambda x: x[0], res)))
        self.test = tuple([y_val] + list(map(lambda x: x[1], res)))
        if val_perc is not None:
            self.test = tuple(map(lambda x: x[:int(len(x) * val_perc)], self.test))

    def _load_images(self, data, name, split, swap, test_mode):
        images = data[name].astype(theano.config.floatX)
        if test_mode is not None:
            images = images[:test_mode]

        X_train, X_val = images[:split], images[split:]

        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        self._means.append(mean)
        self._stds.append(std)

        # yes, swap after mean, std!
        if swap:
            X_train, X_val = X_val, X_train

        X_train -= mean
        X_train /= std
        X_val -= mean
        X_val /= std

        channels = len(X_train[0])
        image_size = len(X_train[0][0][0])

        X_train = np.reshape(X_train, [-1, channels, image_size, image_size])
        X_val = np.reshape(X_val, [-1, channels, image_size, image_size])

        return X_train, X_val

    def num_classes(self):
        return self._num_classes

    def size(self, train=True):
        if train:
            return len(self.train[0])
        else:
            return len(self.test[0])

    def iter(self, batch_size, config):
        config.extend(shuffle=config.train, mirror=config.train)
        if config.train:
            data = self.train
        else:
            data = self.test
        return iterate_minibatches(data, batch_size, config)

class AllSimpleDataSet(DataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, test_mode=None):
        data = np.load(fname)
        labels = data["labels"].astype(np.int32)
        self._num_classes = len(np.unique(labels))
        if test_mode is not None:
            labels = labels[:test_mode]
        split = int(len(labels) * train_perc)

        self._means = []
        self._stds = []

        res = []
        for s in sizes:
            res.append(self._load_images(data, "images" + str(s), split, swap, test_mode))

        self.train = None
        self.test = tuple([labels] + res)

    def _load_images(self, data, name, split, swap, test_mode):
        images = data[name].astype(theano.config.floatX)
        if test_mode is not None:
            images = images[:test_mode]

        X_train = images[:split]
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        self._means.append(mean)
        self._stds.append(std)

        images -= mean
        images /= std

        channels = len(images[0])
        image_size = len(images[0][0][0])
        return np.reshape(images, [-1, channels, image_size, image_size])

    def num_classes(self):
        return self._num_classes

    def iter(self, batch_size, config):
        assert not config.train
        config.extend(shuffle=False, mirror=False)
        return iterate_minibatches(self.test, batch_size, config)

class BetterCalDataSet(DataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None, norm=True):
        self._sizes = sizes
        self._max_size = max(sizes)
        if self._max_size == 0:
            self._max_size = 48

        data = np.load(fname)
        boxes = data["boxes_true"].astype(np.int32)
        shapes = data["shapes"].astype(np.int64)
        images = data["images"]
        del(data)

        self._cal_classes = CalClasses()

        labels = []
        res = [[]] * len(self._sizes)
        for b in boxes:
            num = b[1]
            ss = shapes[num]
            pos = ss[0]
            shape = ss[1:]
            box = tuple(b[2:])
            im = arr_to_img(images[pos : pos + shape.prod()].reshape(shape))
            for i in range(len(self._cal_classes.classes)):
                cls = self._cal_classes.classes[i]
                mbox = move_box(shape[1], shape[2], box, cls)
                if mbox is None:
                    continue
                img = im.crop(mbox)
                labels.append(i)
                for r, s in zip(res, self._sizes):
                    r.append(img_to_arr_bytes(self._resize_img(img, s)))

        del(boxes)
        del(shapes)
        labels = np.array(labels, dtype=np.int32)
        images = [None] * len(res)
        for i in range(len(images)):
            images[i] = np.array(res[i], dtype=np.uint8)
            res[i] = None
        del(res)

        if test_mode is not None:
            labels = labels[:test_mode]
            images = list(map(lambda x: x[:test_mode], images))

        if test_mode is None:
            np.random.seed(12345)
            perm = np.arange(len(labels))
            np.random.shuffle(perm)
            labels[:] = labels[perm]
            for x in images:
                x[:] = x[perm]
            del(perm)
            np.random.seed(int(time.time()))


        split = int(len(labels) * train_perc)
        labels_train, labels_val = labels[:split], labels[split:]
        images_train, images_val = list(map(lambda x: x[:split], images)), list(map(lambda x: x[split:], images))

        self._means = list(map(lambda x: x.mean(dtype=theano.config.floatX), images_train))
        self._stds = list(map(lambda x: x.std(dtype=theano.config.floatX), images_train))

        for i in range(len(images_train)):
            images_train[i] = images_train[i].astype(theano.config.floatX)

        for i in range(len(images_val)):
            images_val[i] = images_val[i].astype(theano.config.floatX)

        if norm:
            for xt, xv, mean, std in zip(images_train, images_val, self._means, self._stds):
                xt -= mean
                xt /= std
                xv -= mean
                xv /= std

        if swap:
            labels_train, labels_val = labels_val, labels_train
            images_train, images_val = images_val, images_train

        self._train = tuple([labels_train] + images_train)
        self._test = tuple([labels_val] + images_val)
        if val_perc is not None:
            self._test = tuple(map(lambda x: x[:int(len(x) * val_perc)], self._test))

    def num_classes(self):
        return len(self._cal_classes.classes)

    def size(self, train=True):
        if train:
            return len(self._train[0])
        else:
            return len(self._test[0])

    def iter(self, batch_size, config):
        config.extend(shuffle=config.train, mirror=False)
        if config.train:
            data = self._train
        else:
            data = self._test
        return iterate_minibatches(data, batch_size, config)

    def _resize_img(self, img, size):
        if size == 0:
            return img
        return img.resize((size, size), Image.ANTIALIAS)

class AllBetterCalDataSet(BetterCalDataSet):
    def __init__(self, fname, sizes, train_perc=0.75, norm=False):
        super().__init__(fname, sizes, train_perc)
        self._data = tuple(map(np.concatenate, zip(self._train, self._test)))
        self._train = None
        self._test = None

    def size(self, train=False):
        assert not train
        return len(self._data[0])

    def iter(self, batch_size, config):
        assert not config.train
        config.extend(shuffle=False, mirror=False)
        return iterate_minibatches(self._data, batch_size, config)

class BestClsDataSet(DataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None, norm=True):
        self._sizes = sizes
        self._max_size = max(sizes)

        data = np.load(fname)
        self._images = data["images"]
        # tmp = self._images.astype(theano.config.floatX)
        # mean, std = tmp.mean(), tmp.std()
        # del(tmp)
        # NOTE: hardcoded for data/data.npz
        # mean, std = 108.758452193, 67.4077411402
        # NOTE: hardcoded for data/data-with-lfw.npz
        mean, std = 103.668, 70.0486
        self._means = [mean] * len(sizes)
        self._stds = [std] * len(sizes)

        boxes_true = data["boxes_true"].astype(np.int32)
        boxes_false = data["boxes_false"].astype(np.int32)
        self._shapes = data["shapes"].astype(np.int64)
        self._names = data["names"]

        if test_mode is not None and test_mode != True:
            boxes_true = boxes_true[:test_mode]
            boxes_false = boxes_false[:test_mode]

        if test_mode is None:
            np.random.seed(12345)
            np.random.shuffle(boxes_true)
            np.random.shuffle(boxes_false)
            # np.random.seed(int(time.time()))

        split_true, split_false = int(len(boxes_true) * train_perc), int(len(boxes_false) * train_perc)
        self._train = (boxes_true[:split_true], boxes_false[:split_false])
        self._test = (boxes_true[split_true:], boxes_false[split_false:])

        if swap:
            self._train, self._test = self._test, self._train

        if val_perc is not None:
            self._test = tuple(map(lambda x: x[:int(len(x) * val_perc)], self._test))

        self._cal_classes = CalClasses()

    def num_classes(self):
        return 2

    def size(self, train):
        if train:
            return len(self._train[0]) * 2
        else:
            return len(self._test[0]) * 2

    def iter_boxes(self, boxes_true, boxes_false, batch_size, config):
        assert batch_size % 2 == 0

        l = len(boxes_true)

        num = batch_size // 2

        iter_num = 0

        if config.shuffle:
            indices_true = np.arange(len(boxes_true))
            np.random.shuffle(indices_true)
            indices_false = np.random.randint(0, len(boxes_false), size=len(boxes_true))

        last_end = 0
        for start_idx in range(0, l - num + 1, num):
            last_end = start_idx + num
            itn = iter_num
            iter_num += 1

            if not config.filter(itn):
                continue

            if config.shuffle:
                excerpt_true = indices_true[start_idx:last_end]
                excerpt_false = indices_false[start_idx:last_end]
            else:
                excerpt_true = slice(start_idx, last_end)
                excerpt_false = excerpt_true

            yield np.concatenate((boxes_true[excerpt_true], boxes_false[excerpt_false]))

        if last_end < l:
            start_idx = last_end
            if config.filter(iter_num):
                if config.shuffle:
                    excerpt_true = indices_true[start_idx:l]
                    excerpt_false = indices_false[start_idx:l]
                else:
                    excerpt_true = slice(start_idx, l)
                    excerpt_false = excerpt_true

                yield np.concatenate((boxes_true[excerpt_true], boxes_false[excerpt_false]))
            iter_num += 1

    def _resize_img(self, img, size):
        if size == 0:
            return img
        return img.resize((size, size), Image.ANTIALIAS)

    def iter(self, batch_size, config):
        config.extend(shuffle=config.train, mirror=config.train, move=config.train, norm=True)
        if config.train:
            (boxes_true, boxes_false) = self._train
        else:
            (boxes_true, boxes_false) = self._test
        return map(lambda ids: self._get_batch(ids, config), self.iter_boxes(boxes_true, boxes_false, batch_size, config))

    def _get_batch(self, boxes, config):
        batch_size = len(boxes)
        num_channels = 3

        res = [boxes[:, 0]]
        for s in self._sizes:
            res.append(np.zeros(shape=(batch_size, num_channels, s, s), dtype=np.uint8))

        for i in range(len(boxes)):
            b = boxes[i]
            label = b[0]
            ss = self._shapes[b[1]]
            pos = ss[0]
            shape = ss[1:]
            ilen = shape.prod()
            box = tuple(b[2:])
            if label == 0:
                mbox = get_rnd_box(box, self._max_size)
            else:
                if config.move:
                    mbox = move_box_nobound(box, self._cal_classes.rnd_cont())
                    mbox = box_stretch(mbox, self._cal_classes.rnd_stretch())
                    mbox = tuple(map(lambda x: int(round(x)), bbox_bound(mbox, shape[1], shape[2])))
                else:
                    mbox = box
            img = self._images[pos : pos + ilen].reshape(shape)
            img = arr_to_img(img).crop(mbox)
            for j in range(len(self._sizes)):
                im = self._resize_img(img, self._sizes[j])
                res[j + 1][i] = img_to_arr_bytes(im)

        if config.shuffle:
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            for i in range(len(res)):
                res[i] = res[i][indices]

        if config.mirror:
            mask = np.random.rand(batch_size) > 0.5
            for x in res[1:]:
                x[mask] = x[mask,:,:,::-1]

        for i in range(len(res) - 1):
            res[i + 1] = res[i + 1].astype(theano.config.floatX)
            if config.norm:
                res[i + 1] -= self._means[i]
                res[i + 1] /= self._stds[i]

        return tuple(res)

class AllBestClsDataSet(BestClsDataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None, norm=False):
        super().__init__(fname, sizes, train_perc, swap, val_perc, test_mode)
        self._data = (np.concatenate((self._train[0], self._test[0])), np.concatenate((self._train[1], self._test[1])))
        self._train = None
        self._test = None

    def size(self, train=False):
        if train:
            return 0
        return len(self._data[0]) * 2

    def iter(self, batch_size, config):
        assert not config.train
        config.extend(norm=True, shuffle=False, mirror=False, move=False)
        (boxes_true, boxes_false) = self._data
        return map(lambda ids: self._get_batch(ids, config), self.iter_boxes(boxes_true, boxes_false, batch_size, config))

    def _iter_indexes_one(self):
        indices_true = np.arange(len(boxes_true))
        np.random.shuffle(indices_true)
        indices_false = np.arange(len(boxes_false))
        np.random.shuffle(indices_false)

        idx_true = 0
        idx_false = 0
        for i in range(0, len(boxes_true) + len(boxes_false)):
            if idx_true > len(boxes_true):
                yield boxes_false[indices_false[idx_false]]
                idx_false += 1
            elif idx_false > len(boxes_false):
                yield boxes_true[indices_false[idx_true]]
                idx_true += 1
            else:
                if random.random() >= 0.5:
                    yield boxes_false[indices_false[idx_false]]
                    idx_false += 1
                else:
                    yield boxes_true[indices_false[idx_true]]
                    idx_true += 1

    def _get_batch_one(self, box):
        size = self._sizes[0]
        batch_size = len(boxes)
        num_channels = 3

        b = box
        label = b[0]
        ss = self._shapes[b[1]]
        pos = ss[0]
        shape = ss[1:]
        ilen = shape.prod()
        box = tuple(b[2:])
        if label == 0:
            mbox = get_rnd_box(box, self._max_size)
        else:
            if move:
                mbox = move_box_nobound(box, self._cal_classes.rnd_cont())
                mbox = box_stretch(mbox, self._cal_classes.rnd_stretch())
                mbox = tuple(map(lambda x: int(round(x)), bbox_bound(mbox, shape[1], shape[2])))
            else:
                mbox = box
        img = self._images[pos : pos + ilen].reshape(shape)
        img = arr_to_img(img).crop(mbox)
        for j in range(len(self._sizes)):
            im = self._resize_img(img, self._sizes[j])
            res[j + 1][i] = img_to_arr_bytes(im)

        mask = np.random.rand(batch_size) > 0.5
        for x in res[1:]:
            x[mask] = x[mask,:,:,::-1]

        for i in range(len(res) - 1):
            res[i + 1] = res[i + 1].astype(theano.config.floatX)
            if norm:
                res[i + 1] -= self._means[i]
                res[i + 1] /= self._stds[i]


class BestSmilesDataSet(DataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None, norm=True):
        self._sizes = sizes
        self._max_size = max(sizes)

        data = np.load(fname)
        self._images = data["images"]
        # mean, std = self._images.mean(dtype=theano.config.floatX), self._images.std(dtype=theano.config.floatX)
        # NOTE: hardcoded for data/data-smiles.npz
        mean, std = 108.758452193, 67.4077411402
        self._means = [mean] * len(sizes)
        self._stds = [std] * len(sizes)

        boxes_true = data["boxes_true"].astype(np.int32)
        self._shapes = data["shapes"].astype(np.int64)
        self._names = data["names"]

        self._num_classes = len(np.unique(boxes_true[:, 0]))

        if test_mode is not None:
            boxes_true = boxes_true[:test_mode]

        if test_mode is None:
            np.random.seed(12345)
            np.random.shuffle(boxes_true)
            np.random.seed(int(time.time()))

        split_true = int(len(boxes_true) * train_perc)
        self._train = boxes_true[:split_true]
        self._test = boxes_true[split_true:]

        if swap:
            self._train, self._test = self._test, self._train

        if val_perc is not None:
            self._test = self._test[:int(len(self._test) * val_perc)]

        self._cal_classes = CalClasses()

    def num_classes(self):
        return self._num_classes

    def size(self, train):
        if train:
            return len(self._train) * 2
        else:
            return len(self._test) * 2

    def iter_boxes(self, boxes_true, batch_size, config):
        assert batch_size % 2 == 0

        l = len(boxes_true)

        num = batch_size // 2

        iter_num = 0

        if config.shuffle:
            indices_true = np.arange(len(boxes_true))
            np.random.shuffle(indices_true)

        last_end = 0
        for start_idx in range(0, l - num + 1, num):
            last_end = start_idx + num
            itn = iter_num
            iter_num += 1

            if not config.filter(itn):
                continue

            if config.shuffle:
                excerpt_true = indices_true[start_idx:last_end]
            else:
                excerpt_true = slice(start_idx, last_end)

            yield boxes_true[excerpt_true]

        if last_end < l:
            start_idx = last_end
            if config.filter(iter_num):
                if config.shuffle:
                    excerpt_true = indices_true[start_idx:l]
                else:
                    excerpt_true = slice(start_idx, l)

                yield boxes_true[excerpt_true]
            iter_num += 1

    def _resize_img(self, img, size):
        if size == 0:
            return img
        return img.resize((size, size), Image.ANTIALIAS)

    def iter(self, batch_size, config):
        config.extend(shuffle=config.train, mirror=config.train, move=config.train, norm=True)
        if config.train:
            boxes_true = self._train
        else:
            boxes_true = self._test
        return map(lambda ids: self._get_batch(ids, config), self.iter_boxes(boxes_true, batch_size, config))

    def _get_batch(self, boxes, config):
        batch_size = len(boxes)
        num_channels = 3

        res = [boxes[:, 0]]
        for s in self._sizes:
            res.append(np.zeros(shape=(batch_size, num_channels, s, s), dtype=np.uint8))

        for i in range(len(boxes)):
            b = boxes[i]
            label = b[0]
            ss = self._shapes[b[1]]
            pos = ss[0]
            shape = ss[1:]
            ilen = shape.prod()
            box = tuple(b[2:])
            if config.move:
                mbox = move_box_nobound(box, self._cal_classes.rnd_cont())
                mbox = box_stretch(mbox, self._cal_classes.rnd_stretch())
                mbox = tuple(map(lambda x: int(round(x)), bbox_bound(mbox, shape[1], shape[2])))
            else:
                mbox = box
            img = self._images[pos : pos + ilen].reshape(shape)
            img = arr_to_img(img).crop(mbox)
            for j in range(len(self._sizes)):
                im = self._resize_img(img, self._sizes[j])
                res[j + 1][i] = img_to_arr_bytes(im)

        if config.shuffle:
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            for i in range(len(res)):
                res[i] = res[i][indices]

        if config.mirror:
            mask = np.random.rand(batch_size) > 0.5
            for x in res[1:]:
                x[mask] = x[mask,:,:,::-1]

        for i in range(len(res) - 1):
            res[i + 1] = res[i + 1].astype(theano.config.floatX)
            if config.norm:
                res[i + 1] -= self._means[i]
                res[i + 1] /= self._stds[i]

        return tuple(res)


class AllBestSmilesDataSet(BestSmilesDataSet):
    def __init__(self, fname, sizes, train_perc=0.75, swap=False, val_perc=None, test_mode=None, norm=False):
        super().__init__(fname, sizes, train_perc, swap, val_perc, test_mode)
        self._data = np.concatenate((self._train, self._test))
        self._train = None
        self._test = None

    def size(self, train=False):
        if train:
            return 0
        return len(self._data)

    def iter(self, batch_size, config):
        assert not config.train
        config.extend(norm=True, shuffle=False, mirror=False, move=False)
        return map(lambda ids: self._get_batch(ids, config), self.iter_boxes(self._data, batch_size, config))

import multiprocessing as mp
import queue as Q
from multiprocessing import Process, Queue

def run_process(queue, dataset, batch_size, config):
    while True:
        for batch in dataset.iter(batch_size, config=config):
            queue.put(batch)
        queue.put(None)

def run_process_aggr(queues, queue):
    while True:
        done_cnt = 0
        done = [False] * len(queues)
        i = 0
        while done_cnt < len(queues):
            if not done[i]:
                try:
                    obj = queues[i].get(True, 0.1)
                except Q.Empty:
                    pass
                else:
                    if obj is None:
                        done[i] = True
                        done_cnt += 1
                    else:
                        queue.put(obj)

            i += 1
            if i == len(queues):
                i = 0

        queue.put(None)

class QueueDataSet(DataSet):
    def __init__(self, base, batch_size, config, ntrain=None, ntest=None, n=1):
        if ntrain is None:
            ntrain = n
        if ntest is None:
            ntest = n

        self._means = base._means
        self._stds = base._stds
        self._batch_size = batch_size
        self._num_classes = base.num_classes()
        self._processes = []
        self._queue_train = None
        self._queue_test = None

        self._train_size = base.size(train=True)
        if ntrain != 0:
            cnt_train = self._train_size // batch_size
            if self._train_size % batch_size != 0:
                cnt_train += 1

            if ntrain == "all":
                ntr = cnt_train
            else:
                ntr = ntrain


            print("QueueDataSet: starting %d processes for generating training data." % ntr)
            if ntr == 1:
                self._queue_train = Queue((cnt_train + 1) * 2)
                self._processes.append(Process(target=run_process, args=(self._queue_train, base, batch_size, config.clone_extend(train=True, filter=lambda _: True))))
            else:
                self._queue_train = []
                for i in range(ntr):
                    q = Queue((cnt_train // ntr) * 2)
                    self._processes.append(Process(target=run_process, args=(q, base, batch_size, config.clone_extend(train=True, filter=lambda x, vi=i: x % ntr == vi))))
                    self._queue_train.append(q)

                q = Queue((cnt_train + 1) * 2)
                self._processes.append(Process(target=run_process_aggr, args=(self._queue_train[:], q)))
                self._queue_train.append(q)

        self._test_size = base.size(train=False)
        if ntest != 0:
            cnt_test = self._test_size // batch_size
            if self._test_size % batch_size != 0:
                cnt_test += 1

            if ntest == "all":
                ntst = cnt_test
            else:
                ntst = ntest

            print("QueueDataSet: starting %d processes for generating test data." % ntst)
            if ntst == 1:
                self._queue_test = Queue((cnt_test + 1) * 2)
                self._processes.append(Process(target=run_process, args=(self._queue_test, base, batch_size, config.clone_extend(train=False, filter=lambda _: True))))
            else:
                self._queue_test = []
                for i in range(ntst):
                    q = Queue((cnt_test // ntst) * 2)
                    self._processes.append(Process(target=run_process, args=(q, base, batch_size, config.clone_extend(train=False, filter=lambda x, vi=i: x % ntst == vi))))
                    self._queue_test.append(q)

                q = Queue((cnt_test + 1) * 2)
                self._processes.append(Process(target=run_process_aggr, args=(self._queue_test[:], q)))
                self._queue_test.append(q)

        self._closed = False

        import atexit
        atexit.register(self.close)

        for p in self._processes:
            p.start()

    def num_classes(self):
        return self._num_classes

    def size(self, train=True):
        if train:
            return self._train_size
        else:
            return self._test_size

    def iter(self, batch_size, config):
        assert batch_size == self._batch_size
        if config.train:
            q = self._queue_train
        else:
            q = self._queue_test

        if q is None:
            return

        if type(q) is list:
            q = q[-1]

        while True:
            obj = q.get()
            if obj is None:
                break
            yield obj

    def close(self):
        if self._closed:
            return

        self._closed = True

        if self._queue_train is not None:
            if type(self._queue_train) is not list:
                self._queue_train.close()
            else:
                i = 0
                for p in self._queue_train:
                    p.close()
                    i += 1

        if self._queue_test is not None:
            if type(self._queue_test) is not list:
                self._queue_test.close()
            else:
                for p in self._queue_test:
                    p.close()

        for p in self._processes:
            p.terminate()
