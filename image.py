import os
import numpy as np
import pandas as pd
from uuid import uuid4
from skimage.draw import circle
from skimage.io import use_plugin, imread, imshow, imsave

use_plugin('tifffile', 'imsave')

# base class of simulation objects
class SynthImageObject(object):
    def __init__(self):
        self.oid = uuid4()
        self.index = 0
        self.label = None
        self.loc = (0, 0)
        self.shape = (1, 1, 3)
        self.time_func = lambda obj: obj
        self.time_params = {}
        self.trans_func = lambda obj: obj
        self.trans_params = {}
        self.loc_func = lambda obj: obj
        self.loc_params = {}
        self.img = None
        self.mask = None
    
    def __str__(self):
        return str(pd.DataFrame(columns=['label', 'x_loc', 'y_loc'], 
                                data=[[self.label, self.loc[0], self.loc[1]]],
                                index=[self.oid]))
    
    def set_time_func(self, time_func, kwargs={}):
        self.time_func = time_func
        self.time_params = kwargs
        return
    
    def set_trans_func(self, trans_func, kwargs={}):
        self.trans_func = trans_func
        self.trans_params = kwargs
        return
    
    def set_loc_func(self, loc_func, kwargs={}):
        self.loc_func = loc_func
        self.loc_params = kwargs
        return


# specific types of simulation objects
class Particle(SynthImageObject):
    def __init__(self, label, shape=(9, 9, 3), color=(255,255,255)):
        super().__init__()
        self.label = label
        self.shape = shape
        self.img = np.zeros(shape, dtype=np.uint8)
        cen = round(shape[0]/2)
        rr, cc = circle(cen, cen, shape[0]-cen, shape=shape[:2])
        self.img[rr, cc, :] = np.array(color)


class Picture(SynthImageObject):
    def __init__(self, fname, label):
        super().__init__()
        self.img = imread(fname)[:, :, :2]
        self.label = label
        self.shape = self.img.shape


class PSF(SynthImageObject):
    def __init__(self, label):
        super().__init__()


# class for simulation frame
class SynthImage(object):
    def __init__(self, shape=(512, 512, 3)):
        self.index = 0
        self.shape = shape
        self.img = np.zeros(shape, dtype=np.uint8)
        self.mask = np.zeros(shape, dtype=np.uint8)
        self.objs = []
        self.noise_func = lambda img: img
        self.noise_params = {}
    
    def __str__(self):
        return str(self.get_data())
    
    def get_data(self):
        cols = ['label', 'x_loc', 'y_loc']
        if self.objs:
            data = []
            oids = []
            for obj in self.objs:
                data.append([obj.label, obj.loc[0], obj.loc[1]])
                oids.append(obj.oid)
            df = pd.DataFrame(columns=cols, index=oids, data=data)
        else:
            df = pd.DataFrame(columns=cols)
        return df
    
    def _bound_overflow(self, n0, n1, n2, bsh, fsh):
        if bsh >= fsh:
            raise ValueError('SynthObject should not be bigger than the SynthImage.')
        if n1 < 0 and fsh > n2 >= 0:
            bi1, bi2, fi1, fi2 = abs(n1), bsh, 0, n2
        elif fsh > n1 >= 0 and n2 >= fsh:
            bi1, bi2, fi1, fi2 = 0, fsh-n1, n1, fsh
        elif fsh > n1 >= 0 and fsh > n2 >= 0:
            bi1, bi2, fi1, fi2 = 0, bsh, n1, n2
        else:
            bi1, bi2, fi1, fi2 = 0, 0, 0, 0
        return bi1, bi2, fi1, fi2
    
    def _embed_img(self, img, frame, loc):
        f0, f1 = frame.shape[:2]
        l0, l1 = loc
        b0, b1 = img.shape[:2]
        ybr, xbr = map(lambda b: round(b/2), (b0, b1))
        yrm, xrm = b0-ybr, b1-xbr
        y1, x1 = l0-ybr, l1-xbr
        y2, x2 = l0+yrm, l1+xrm
        ybi1, ybi2, yfi1, yfi2 = self._bound_overflow(l0, y1, y2, b0, f0)
        xbi1, xbi2, xfi1, xfi2 = self._bound_overflow(l1, x1, x2, b1, f1)
        frame[yfi1:yfi2, xfi1:xfi2, :] += img[ybi1:ybi2, xbi1:xbi2, :]
        frame[frame > 255] = 255
        return frame
    
    def add_objs(self, objs, loc_func, loc_params={}):
        try:
            loc_params['shape']
        except KeyError:
            loc_params['shape'] = self.shape[:2]
        for obj in objs:
            obj.loc_func = loc_func
            obj.loc_params = loc_params
            obj = obj.loc_func(obj, **obj.loc_params)
            self.img = self._embed_img(obj.img, self.img, obj.loc)
            self.objs.append(obj)
        self._update_mask()
        return
    
    def update_objs(self):
        self.img = np.zeros(self.shape, dtype=np.uint8)
        self._update_mask()
        if not self.objs:
            raise ValueError('No objects have been added.')
        else:
            for obj in self.objs:
                obj.index += 1
                obj = obj.time_func(obj, **obj.time_params)
                obj = obj.trans_func(obj, **obj.trans_params)
                self.img = self._embed_img(obj.img, self.img, obj.loc)
            self._update_mask()
        return
    
    def _update_mask(self):
        self.mask = np.copy(self.img)
        self.mask[self.img.sum(axis=2) > 0] = 255
        return
    
    def set_noise_func(self, noise_func, kwargs={}):
        self.noise_func = noise_func
        self.noise_params = kwargs
        return
    
    def apply_noise(self):
        self.img = self.noise_func(self.img, **self.noise_params)
        return
    
    def show(self):
        imshow(self.img)
        return
    
    def save(self, fname='image.tif'):
        if os.path.exists(fname):
            os.unlink(fname)
        imsave(fname, self.img)
        return


def generate(ObjClass, n, label, class_params={},
                  time_func=lambda obj: obj, time_params={}, 
                  trans_func=lambda obj: obj, trans_params={},
                  loc_func=lambda obj: obj, loc_params={}):
    objs = []
    for i in range(n):
        obj = ObjClass(label, **class_params)
        obj.set_time_func(time_func, time_params)
        obj.set_trans_func(trans_func, trans_params)
        obj.set_loc_func(loc_func, loc_params)
        objs.append(obj)
    return objs


# modular functions
def brownian(obj, dt=1, delta=5):
    y, x = obj.loc
    y += np.random.randn() * 2*delta*dt
    x += np.random.randn() * 2*delta*dt
    obj.loc = (int(y), int(x))
    return obj


def uniform_noise(frame):
    ys = np.random.randint(0, frame.shape[0], size=100)
    xs = np.random.randint(0, frame.shape[1], size=100)
    noise = np.zeros(frame.shape, dtype=np.uint8)
    noise[ys, xs, :] = 127
    frame += noise
    return frame


def random_loc(obj, shape=(0, 0)):
    x_high, y_high = shape
    x_loc = np.random.randint(0, x_high)
    y_loc = np.random.randint(0, y_high)
    obj.loc = (y_loc, x_loc)
    return obj


# class for image stack
class SynthImageStack(object):
    def __init__(self, frame, n_frames):
        self.stack = np.zeros((n_frames,) + frame.shape, dtype=np.uint8)
        self.frame = frame
        self.n_frames = n_frames
        self.data = []
    
    def build_stack(self):
        for n in range(self.n_frames):
            self.frame.update_objs()
            self.frame.apply_noise()
            self.stack[n, :, :, :] = self.frame.img
            self.data.append(self.frame.get_data())
            self.frame.index += 1
        return
    
    def save(self, fname='image.tif'):
        imsave(fname, self.stack)
        return


# test the API
green = generate(Particle, 50, 'green', class_params={'color': (0,255,0), 
                    'shape': (30, 30, 3)})
purple = generate(Particle, 50, 'purple', class_params={'color': (255,0,255), 
                    'shape': (30, 30, 3)})

frame = SynthImage(shape=(512, 512, 3))
frame.set_noise_func(uniform_noise)

frame.add_objs(green, random_loc)
frame.add_objs(purple, random_loc)
frame.apply_noise()
frame.show()


def two_phase_brownian(obj, dt=1, time_switch=5, delta1=5, delta2=40):
    x, y = obj.loc
    if obj.index < time_switch:
        delta = delta1
    else:
        delta = delta2
    x += np.random.randn() * 2*delta*dt
    y += np.random.randn() * 2*delta*dt
    obj.loc = (int(x), int(y))
    return obj

red_dots = generate(Particle, 20, 'foo', time_func=brownian, 
                         class_params={'color': (255,0,0)})
blue_dots = generate(Particle, 20, 'bar', time_func=two_phase_brownian, 
                          class_params={'color': (0,0,255)})

frame = SynthImage()
frame.add_objs(red_dots, random_loc)
frame.add_objs(blue_dots, random_loc)
frame.set_noise_func(uniform_noise)

stack = SynthImageStack(frame, n_frames=10)
stack.build_stack()
stack.save('two_phase.tif')
