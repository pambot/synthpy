import os
import numpy as np
import pandas as pd
import abc
from uuid import uuid4
from skimage.draw import circle
from skimage.io import use_plugin, imread, imshow, imsave

use_plugin('tifffile')


# base class of simulation objects
class SynthImageObj(object):
    
    __metaclass__  = abc.ABCMeta
    
    def __init__(self, label, img=np.zeros((0, 0, 3), dtype=np.uint8), index=0, 
                 loc=(0, 0), time_loc_func=lambda obj: obj, time_loc_params={}, 
                 transform_func=lambda obj: obj, transform_params={},
                 init_loc_func=lambda obj: obj, init_loc_params={}):
        self.oid = uuid4()
        self.index = index
        self.label = label
        self.img = img
        self.loc = loc
        self.shape = self.img.shape
        self.time_loc_func = time_loc_func
        self.time_loc_params = time_loc_params
        self.transform_func = transform_func
        self.transform_params = transform_params
        self.init_loc_func = init_loc_func
        self.init_loc_params = init_loc_params
    
    def __str__(self):
        return str(pd.DataFrame(columns=['label', 'x_loc', 'y_loc'], 
                                data=[[self.label, self.loc[0], self.loc[1]]],
                                index=[self.oid]))
    
    def set_time_loc_func(self, time_loc_func, time_loc_params={}):
        self.time_loc_func = time_loc_func
        self.time_loc_params = time_loc_params
        return
    
    def set_transform_func(self, transform_func, transform_params={}):
        self.transform_func = transform_func
        self.transform_params = transform_params
        return
    
    def set_init_loc_func(self, init_loc_func, init_loc_params={}):
        self.init_loc_func = init_loc_func
        self.init_loc_params = init_loc_params
        return
    
    @classmethod
    def generate(cls, n, *args, **kwargs):
        objs = []
        for i in range(n):
            obj = cls(*args, **kwargs)
            objs.append(obj)
        return objs


# specific types of simulation objects
class Particle(SynthImageObj):
    def __init__(self, label, shape=(9, 9, 3), color=(255,255,255), **kwargs):
        super().__init__(label, **kwargs)
        self.shape = shape
        self.img = np.zeros(shape, dtype=np.uint8)
        radius = round(shape[0]/2)
        rr, cc = circle(radius, radius, shape[0]-radius, shape=shape[:2])
        self.img[rr, cc, :] = np.array(color)


class FileImage(SynthImageObj):
    def __init__(self, file_img, label, **kwargs):
        if type(file_img) is str:
            img = imread(file_img)
        elif type(file_img) is np.ndarray:
            img = file_img
        else:
            raise ValueError('Parameter "file_img" must be a filename or a (M, N, 3) Numpy ndarray.')
        
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            raise ValueError('Numpy ndarray must be of shape (M, N, 3).')
        
        super().__init__(label, **kwargs)
        self.img = img
        self.shape = self.img.shape
    
    @classmethod
    def generate(cls, n, file_img_array, label, **kwargs):
        random_imgs = np.random.choice(file_img_array, size=n, replace=False)
        objs = []
        for f_img in random_imgs:
            obj = cls(f_img, label, **kwargs)
            objs.append(obj)
        return objs


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
    
    @staticmethod
    def _bound_overflow(n0, n1, n2, b_shape, f_shape):
        if b_shape >= f_shape:
            raise ValueError('SynthObject should not be bigger than the SynthImage.')
        if n1 < 0 and f_shape > n2 >= 0:
            bi1, bi2, fi1, fi2 = abs(n1), b_shape, 0, n2
        elif f_shape > n1 >= 0 and n2 >= f_shape:
            bi1, bi2, fi1, fi2 = 0, f_shape-n1, n1, f_shape
        elif f_shape > n1 >= 0 and f_shape > n2 >= 0:
            bi1, bi2, fi1, fi2 = 0, b_shape, n1, n2
        else:
            bi1, bi2, fi1, fi2 = 0, 0, 0, 0
        return bi1, bi2, fi1, fi2
    
    @classmethod
    def _embed_img(cls, img, frame, loc):
        f0, f1 = frame.shape[:2]
        l0, l1 = loc
        b0, b1 = img.shape[:2]
        ybr, xbr = map(lambda b: round(b/2), (b0, b1))
        yrm, xrm = b0-ybr, b1-xbr
        y1, x1 = l0-ybr, l1-xbr
        y2, x2 = l0+yrm, l1+xrm
        ybi1, ybi2, yfi1, yfi2 = cls._bound_overflow(l0, y1, y2, b0, f0)
        xbi1, xbi2, xfi1, xfi2 = cls._bound_overflow(l1, x1, x2, b1, f1)
        frame[yfi1:yfi2, xfi1:xfi2, :] += img[ybi1:ybi2, xbi1:xbi2, :]
        frame[frame > 255] = 255
        return frame
    
    def add_objs(self, objs, init_loc_func, init_loc_params={}):
        try:
            init_loc_params['shape']
        except KeyError:
            init_loc_params['shape'] = self.shape[:2]
        for obj in objs:
            obj.init_loc_func = init_loc_func
            obj.init_loc_params = init_loc_params
            obj = obj.init_loc_func(obj, **obj.init_loc_params)
            self.img = self._embed_img(obj.img, self.img, obj.loc)
            self.objs.append(obj)
        self._update_mask()
        return
    
    def update_objs(self):
        self.img = np.zeros(self.shape, dtype=np.uint8)
        if not self.objs:
            raise ValueError('No objects have been added.')
        else:
            for obj in self.objs:
                obj.index += 1
                obj = obj.time_loc_func(obj, **obj.time_loc_params)
                obj = obj.transform_func(obj, **obj.transform_params)
                self.img = self._embed_img(obj.img, self.img, obj.loc)
                obj.index += 1
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
green = Particle.generate(30, 'green', shape=(30, 30, 3), color=(0,255,0), 
                          time_loc_func=brownian, init_loc_func=random_loc)
purple = Particle.generate(30, 'purple', shape=(30, 30, 3), color=(255,0,255), 
                           time_loc_func=brownian, init_loc_func=random_loc)

frame = SynthImage(shape=(512, 512, 3))
frame.set_noise_func(uniform_noise)

frame.add_objs(green, random_loc)
frame.add_objs(purple, random_loc)
frame.apply_noise()
frame.show()


def two_phase_brownian(obj, dt=1, time_loc_switch=5, delta1=5, delta2=40):
    x, y = obj.loc
    if obj.index < time_loc_switch:
        delta = delta1
    else:
        delta = delta2
    x += np.random.randn() * 2*delta*dt
    y += np.random.randn() * 2*delta*dt
    obj.loc = (int(x), int(y))
    return obj

red = Particle.generate(20, 'foo', shape=(30, 30, 3), color=(255,0,0), 
                        time_loc_func=brownian)
blue = Particle.generate(20, 'bar', shape=(30, 30, 3), color=(0,0,255), 
                         time_loc_func=two_phase_brownian)

frame = SynthImage()
frame.add_objs(red, random_loc)
frame.add_objs(blue, random_loc)
frame.set_noise_func(uniform_noise)

stack = SynthImageStack(frame, n_frames=10)
stack.build_stack()
stack.save('two_phase.tif')


# now make real examples
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import find_objects

cell_img = imread('cell-1_1.tif')

def segment_channel(cell_img, ch):
    channel_objs = np.zeros(cell_img.shape, dtype=np.uint8)
    channel_objs[:, :, ch] = cell_img[:, :, ch]
    
    blur = gaussian(channel_objs, sigma=10, multichannel=True)
    bval = threshold_otsu(blur.sum(axis=2))
    blobs = measure.label(blur > bval, background=0)
    blob_slices = find_objects(blobs)
    
    img_slices = []
    for b_slice in blob_slices:
        channel_slice = channel_objs[b_slice].sum(axis=2)
        img_slice = np.zeros(channel_slice.shape + (3,), dtype=np.uint8)
        img_slice[:, :, ch] = channel_slice
        img_slices.append(img_slice)
    
    return img_slices


red_segs = segment_channel(cell_img, 0)
green_segs = segment_channel(cell_img, 1)

red = FileImage.generate(20, red_segs, 'foo')
green = FileImage.generate(20, green_segs, 'bar')

frame = SynthImage()
frame.add_objs(red, random_loc)
frame.add_objs(green, random_loc)
frame.save('test.tif')



"""
plt.figure()
plt.imshow(blobs.sum(axis=2), cmap='spectral')
plt.tight_layout()
plt.show()
"""