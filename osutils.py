#!/usr/bin/env python

from __future__ import print_function

import os
import errno
import itertools

import numpy as np


def all_files(dir):
    for subdir, _, files in os.walk(dir):
        for f in files:
            yield (subdir, f)

def iterate_keys(c, q):
    c = ord(c)
    q = ord(q)
    import cv2
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == q:
            break
        if key == c:
            yield ()

def make_batches(iterator, batch_size):
    buf = [None] * batch_size
    blen = 0
    for v in iterator:
        if blen == batch_size:
            yield buf
            buf[:] = itertools.repeat(None, batch_size)
            blen = 0

        buf[blen] = v
        blen += 1

    if blen != 0:
        yield buf[:blen]
        blen = 0

def make_batches_tr(iterator, batch_size):
    buf = None
    blen = 0
    tbuf = [None] * batch_size
    for v in iterator:
        if buf is None:
            buf = tuple(map(lambda _: tbuf[:], v))

        if blen == batch_size:
            yield buf
            for x in buf:
                x[:] = itertools.repeat(None, batch_size)
            blen = 0

        for i in range(len(v)):
            buf[i][blen] = v[i]
        blen += 1

    if blen != 0:
        yield tuple(map(lambda x: x[:blen], buf))
        blen = 0

def fullprint(arr):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    print(arr)
    np.set_printoptions(**opt)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    import re
    import subprocess

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # http://code.google.com/p/psutil/
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')
