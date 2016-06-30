import math
import random


def product(it):
    res = 1
    for v in it:
        res *= v
    return res

def minmax(arr):
    minv = arr[0]
    maxv = arr[0]
    for val in arr:
        if val > maxv:
            maxv = val
        elif val < minv:
            minv = val
    return int(math.floor(minv)), int(math.ceil(maxv))

def get_rnd_box(box, max_size):
    minx, miny, maxx, maxy = box
    w = maxx - minx
    h = maxy - miny
    maxs = min(w, h)
    if max_size >= maxs:
        size = maxs - 1
    else:
        size = random.randrange(max_size, maxs)
    x = minx + random.randint(0, w - size)
    y = miny + random.randint(0, h - size)
    return (x, y, x + size, y + size)

def box_mul(box, coef):
    return tuple(map(lambda x: round(x * coef), box))

def box_area(b):
    minx, miny, maxx, maxy = b
    return (maxy - miny) * (maxx - minx)

def box_intersection(b1, b2):
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2
    maxy = min(maxy1, maxy2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    minx = max(minx1, minx2)
    if maxy <= miny:
        maxy = 0
        miny = 0
    if maxx <= minx:
        maxx = 0
        minx = 0
    return (minx, miny, maxx, maxy)

def box_inside(box, big_box):
    return box_intersection(box, big_box) == box

def box_inside_any(box, big_boxes):
    for b in big_boxes:
        if box_inside(box, b):
            return True
    return False

def box_join(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

def is_neighbours(b, box):
    minx1, miny1, maxx1, maxy1 = b
    minx2, miny2, maxx2, maxy2 = box
    samew = maxx2 - minx2 == maxx1 - minx1
    sameh = maxy2 - miny2 == maxy1 - miny1
    if samew:
        if (minx1 == minx2 and maxx1 == maxx2) and (miny1 == maxy2 or miny2 == maxy1):
            return True
    if sameh:
        if (miny1 == miny2 and maxy1 == maxy2) and (minx1 == maxx2 or minx2 == maxx1):
            return True
    return False

def ellipse_to_bbox(a, b, angle, cx, cy):
    a, b = max(a, b), min(a, b)
    ca = math.cos(angle)
    sa = math.sin(angle)
    if sa == 0.0:
        cta = 2.0 / math.pi
    else:
        cta = ca / sa

    if ca == 0.0:
        ta = math.pi / 2.0
    else:
        ta = sa / ca

    def x(t):
        return cx + a * math.cos(t) * ca - b * math.sin(t) * sa

    def y(t):
        return cy + b * math.sin(t) * ca + a * math.cos(t) * sa

    # x = cx + a * cos(t) * cos(angle) - b * sin(t) * sin(angle)
    # tan(t) = -b * tan(angle) / a
    tx1 = math.atan(-b * ta / a)
    tx2 = tx1 - math.pi
    x1, y1 = x(tx1), y(tx1)
    x2, y2 = x(tx2), y(tx2)

    # y = cy + b * sin(t) * cos(angle) + a * cos(t) * sin(angle)
    # tan(t) = b * cot(angle) / a
    ty1 = math.atan(b * cta / a)
    ty2 = ty1 - math.pi
    x3, y3 = x(ty1), y(ty1)
    x4, y4 = x(ty2), y(ty2)

    minx, maxx = minmax([x1, x2, x3, x4])
    miny, maxy = minmax([y1, y2, y3, y4])
    return math.floor(minx), math.floor(miny), math.ceil(maxx), math.ceil(maxy)

def rectify_bbox(box):
    minx, miny, maxx, maxy = box
    dx = maxx - minx
    dy = maxy - miny
    if dx == dy:
        return box

    d = int(abs(dx - dy) / 2)
    if dx < dy:
        minx -= d
        maxx += d
    else:
        miny -= d
        maxy += d
    maxd = max(maxx - minx, maxy - miny)
    maxx = minx + maxd
    maxy = miny + maxd
    return math.floor(minx), math.floor(miny), math.ceil(maxx), math.ceil(maxy)

def bbox_bound(box, imw, imh):
    minx, miny, maxx, maxy = box
    # step 1: reduce left and top
    v = min(minx, miny)
    if v < 0:
        minx += -v
        miny += -v
    # step 2: reduce right
    if maxx > imw:
        d = (maxx - imw)
        maxx -= d
        maxy -= d
    # step 3: reduce bottom
    if maxy > imh:
        d = (maxy - imh)
        maxx -= d
        maxy -= d
    return (minx, miny, maxx, maxy)

def bbox_bound_move(box, imw, imh):
    minx, miny, maxx, maxy = box
    # step 1: move left
    dx, dy = 0, 0
    if minx < 0:
        dx = -minx
    if miny < 0:
        dy = -miny
    minx += dx
    maxx += dx
    miny += dy
    maxy += dy
    # step 2: move right
    dx, dy = 0, 0
    if maxx > imw:
        dx = maxx - imw
    if maxy > imh:
        dy = maxy - imh
    minx += dx
    maxx += dx
    miny += dy
    maxy += dy
    return bbox_bound((minx, miny, maxx, maxy), imw, imh)

def rand_box(imw, imh, boxsize, boxes, attempts=10):
    orig_bs = boxsize
    attempt = 0
    while True:
        x = random.randint(0, imw - boxsize)
        y = random.randint(0, imh - boxsize)
        res = (x, y, x + boxsize, y + boxsize)
        bad = False
        for b in boxes.keys():
            if box_area(box_intersection(res, b)) / float(box_area(b)) > boxes[b]:
                bad = True
                break
        if bad:
            attempt += 1
            if attempt == attempts and boxsize > orig_bs / 10:
                attempt = 0
                boxsize -= 1
            continue
        return res

def move_box_nobound(box, cls):
    dx, dy, ds, _ = cls
    minx, miny, maxx, maxy = box
    w = (maxx - minx)
    h = (maxy - miny)
    newminx = math.floor(minx - dx * w)
    newminy = math.floor(miny - dy * h)
    neww = w / ds
    newh = h / ds
    return (newminx, newminy, math.ceil(newminx + neww), math.ceil(newminy + newh))

def move_box(imw, imh, box, cls):
    return bound_or_none(imw, imh, move_box_nobound(box, cls))

def move_box_force(imw, imh, box, cls):
    return bbox_bound(move_box_nobound(box, cls), imw, imh)

def box_stretch(box, stretch):
    sx, sy = stretch
    minx, miny, maxx, maxy = box
    if sx != 1:
        half = (maxx - minx) / 2
        mid = minx + half
        half *= sx
        box = (mid - half, miny, mid + half, maxy)
    elif sy != 1:
        half = (maxy - miny) / 2
        mid = miny + half
        half *= sy
        box = (minx, mid - half, maxx, mid + half)
    return box

def move_box_back(imw, imh, box, cls):
    dx, dy, ds, _ = cls
    minx, miny, maxx, maxy = box
    w = (maxx - minx)
    h = (maxy - miny)
    neww = w * ds
    newh = h * ds
    newminx = math.ceil(minx + dx * neww)
    newminy = math.ceil(miny + dy * newh)
    return bbox_bound((newminx, newminy, math.floor(newminx + neww), math.floor(newminy + newh)), imw, imh)

def bound_or_none(imw, imh, box):
    minx, miny, maxx, maxy = box
    if minx < 0:
        return None
    if miny < 0:
        return None
    if maxx > imw:
        return None
    if maxy > imh:
        return None
    return box

def bbox_rel(box, rel):
    minx, miny, maxx, maxy = box
    w = maxx - minx
    h = maxy - miny
    rminx, rminy, rmaxx, rmaxy = rel
    newminx = (minx - rminx)
    newminy = (miny - rminy)
    return (newminx, newminy, newminx + w, newminy + h)

def bbox_resize(box, new_size):
    minx, miny, maxx, maxy = box
    w = maxx - minx
    h = maxy - miny
    newminx = math.floor(minx * new_size / w)
    newminy = math.floor(miny * new_size / h)
    new_size = math.ceil(new_size)
    return (newminx, newminy, newminx + new_size, miny + dh + new_size)

def crop(arr, box):
    minx, miny, maxx, maxy = box
    return arr[:, minx:maxx, miny:maxy]
