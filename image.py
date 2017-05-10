import os
import numpy as np
import pandas as pd
import abc
from uuid import uuid4
from skimage.draw import circle
from skimage.io import use_plugin, imread, imshow, imsave

use_plugin('tifffile', 'imsave')


# base class of simulation objects
class SynthImageObj(object):
    
    __metaclass__  = abc.ABCMeta
    
    def __init__(self, label, img=np.zeros((0, 0, 3), dtype=np.uint8), index=0, loc=(0, 0), 
                 time_loc_func=lambda obj, field: (obj, field), time_loc_params={}, 
                 init_loc_func=lambda obj, field: (obj, field), init_loc_params={}):
        self.oid = uuid4()
        self.index = index
        self.label = label
        self.img = img
        self.loc = loc
        self.shape = self.img.shape
        self.time_loc_func = time_loc_func
        self.time_loc_params = time_loc_params
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


# class for simulation field
class SynthImage(object):
    def __init__(self, shape=(512, 512, 3)):
        self.index = 0
        self.shape = shape
        self.img = np.zeros(shape, dtype=np.uint8)
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
            b1, b2, f1, f2 = abs(n1), b_shape, 0, n2
        elif f_shape > n1 >= 0 and n2 >= f_shape:
            b1, b2, f1, f2 = 0, f_shape-n1, n1, f_shape
        elif f_shape > n1 >= 0 and f_shape > n2 >= 0:
            b1, b2, f1, f2 = 0, b_shape, n1, n2
        else:
            b1, b2, f1, f2 = 0, 0, 0, 0
        return (b1, b2), (f1, f2)
    
    @classmethod
    def overflow_slices(cls, img, field, loc):
        fy, fx = field.shape[:2]
        ly, lx = loc
        by, bx = img.shape[:2]
        y_top, x_top = map(lambda b: round(b/2), (by, bx))
        y_bot, x_bot = by-y_top, bx-x_top
        y1, x1 = ly-y_top, lx-x_top
        y2, x2 = ly+y_bot, lx+x_bot
        yb, yf = cls._bound_overflow(ly, y1, y2, by, fy)
        xb, xf = cls._bound_overflow(lx, x1, x2, bx, fx)
        return ((slice(yf[0], yf[1]), slice(xf[0], xf[1])), 
                (slice(yb[0], yb[1]), slice(xb[0], xb[1])))
    
    @classmethod
    def _embed_img(cls, img, field, loc):
        field_slice, img_slice = cls.overflow_slices(img, field, loc)
        draw = np.zeros((img_slice[0].stop-img_slice[0].start, 
                         img_slice[1].stop-img_slice[1].start, 3))
        draw += field[field_slice]
        import pdb
        try:
            draw += img[img_slice]
        except ValueError:
            pdb.set_trace()
        draw[draw > 255] = 255
        field[field_slice] = draw.astype(np.uint8)
        return field
    
    def add_objs(self, objs, init_loc_func, init_loc_params={}):
        for obj in objs:
            obj.set_init_loc_func(init_loc_func, init_loc_params)
            obj.init_loc_func(obj, self, **obj.init_loc_params)
            self.img = self._embed_img(obj.img, self.img, obj.loc)
            self.objs.append(obj)
        return
    
    def adjust_objs(self, adjust_func, adjust_params={}):
        self.img = np.zeros(self.shape, dtype=np.uint8)
        if not self.objs:
            raise ValueError('No objects have been added.')
        else:
            for obj in self.objs:
                adjust_func(obj, self, **adjust_params)
                self.img = self._embed_img(obj.img, self.img, obj.loc)
        return
    
    def update_objs(self):
        self.img = np.zeros(self.shape, dtype=np.uint8)
        if not self.objs:
            raise ValueError('No objects have been added.')
        else:
            for obj in self.objs:
                obj.time_loc_func(obj, self, **obj.time_loc_params)
                self.img = self._embed_img(obj.img, self.img, obj.loc)
                obj.index += 1
        return   
    
    def set_noise_func(self, noise_func, kwargs={}):
        self.noise_func = noise_func
        self.noise_params = kwargs
        return
    
    def apply_noise(self):
        self.noise_func(self, **self.noise_params)
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
def brownian(obj, field, dt=1, delta=5):
    y, x = obj.loc
    y += np.random.randn() * 2*delta*dt
    x += np.random.randn() * 2*delta*dt
    obj.loc = (int(y), int(x))
    return


def uniform_noise(field):
    ys = np.random.randint(0, field.img.shape[0], size=100)
    xs = np.random.randint(0, field.img.shape[1], size=100)
    noise = np.zeros(field.img.shape, dtype=np.uint8)
    noise[ys, xs, :] = 127
    field.img += noise
    return


def random_loc(obj, field, mask=None):
    y_high, x_high = field.shape[:2]
    y_loc = np.random.randint(0, x_high)
    x_loc = np.random.randint(0, y_high)
    if type(mask) == np.ndarray and mask.shape == field.shape[:2]:
        while not mask[y_loc, x_loc].all():
            y_loc = np.random.randint(0, y_high)
            x_loc = np.random.randint(0, x_high)
    obj.loc = (y_loc, x_loc)
    return


def even_grid(obj, field, spacer=50, mask=None):
    y_field, x_field = field.img.shape[:2]
    if not field.objs:
        y_loc, x_loc = spacer, spacer
    else:
        last_obj = field.objs[-1]
        y_loc, x_loc = last_obj.loc
        placed = False
        while not placed:
            if x_loc + spacer < x_field:
                x_loc += spacer
            elif y_loc + spacer < y_field:
                y_loc += spacer
                x_loc = spacer
            else:
                raise ValueError('Grid dimensions exceeded.')
            if type(mask) == np.ndarray and mask.shape == field.shape[:2]:
                if mask[y_loc, x_loc].all():
                    placed = True
                else:
                    continue
            else:
                placed = True
    obj.loc = (y_loc, x_loc)
    return


def random_loc_adjust(obj, field, factor=10):
    y_adj = np.random.randn() * factor
    x_adj = np.random.randn() * factor
    obj.loc = (int(obj.loc[0] + y_adj), int(obj.loc[1] + x_adj))
    return


def random_index_adjust(obj, field, n=10):
    obj.index = np.random.randint(0, n)
    return


# class for image stack
class SynthImageStack(object):
    def __init__(self, field, n_fields):
        self.stack = np.zeros((n_fields,) + field.shape, dtype=np.uint8)
        self.field = field
        self.n_fields = n_fields
        self.data = []
    
    def build_stack(self):
        for n in range(self.n_fields):
            self.field.update_objs()
            self.field.apply_noise()
            self.stack[n, :, :, :] = self.field.img
            self.data.append(self.field.get_data())
            self.field.index += 1
        return
    
    def save(self, fname='image.tif'):
        imsave(fname, self.stack)
        return


# test the API
green = Particle.generate(30, 'green', shape=(30, 30, 3), color=(0,255,0), 
                          time_loc_func=brownian, init_loc_func=random_loc)
purple = Particle.generate(30, 'purple', shape=(30, 30, 3), color=(255,0,255), 
                           time_loc_func=brownian, init_loc_func=random_loc)

field = SynthImage(shape=(512, 512, 3))
field.set_noise_func(uniform_noise)

field.add_objs(green, random_loc)
field.add_objs(purple, random_loc)
field.apply_noise()
field.show()


def two_phase_brownian(obj, field, dt=1, time_loc_switch=5, delta1=5, delta2=40):
    y, x = obj.loc
    if obj.index < time_loc_switch:
        delta = delta1
    else:
        delta = delta2
    y += np.random.randn() * 2*delta*dt
    x += np.random.randn() * 2*delta*dt
    obj.loc = (int(y), int(x))
    return obj

red = Particle.generate(20, 'foo', shape=(30, 30, 3), color=(255, 0, 0), 
                        time_loc_func=brownian)
blue = Particle.generate(20, 'bar', shape=(30, 30, 3), color=(0, 0, 255), 
                         time_loc_func=two_phase_brownian)

field = SynthImage()
field.add_objs(red, random_loc)
field.add_objs(blue, random_loc)
field.set_noise_func(uniform_noise)

stack = SynthImageStack(field, n_fields=10)
stack.build_stack()
stack.save('two_phase.tif')


# now make real examples
# keria's two-color interaction factor
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import find_objects

cell_img = imread('clusters.tif')

def segment_channel(cell_img, ch, sigma=10):
    channel_objs = np.zeros(cell_img.shape, dtype=np.uint8)
    channel_objs[:, :, ch] = cell_img[:, :, ch]
    
    blur = gaussian(channel_objs, sigma=sigma, multichannel=True)
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


def interaction_factor(obj, field, threshold=0.7):
    y_high, x_high = field.shape[:2]
    y_loc = np.random.randint(0, y_high)
    x_loc = np.random.randint(0, x_high)
    if np.random.random() <= threshold:
        while not field.img[y_loc, x_loc, 1]:
            y_loc = np.random.randint(0, y_high)
            x_loc = np.random.randint(0, x_high)
    obj.loc = (y_loc, x_loc)
    return


red_segs = segment_channel(cell_img, 0, sigma=10)
green_segs = segment_channel(cell_img, 1, sigma=10)

red = FileImage.generate(20, red_segs, 'foo')
green = FileImage.generate(20, green_segs, 'bar')

field = SynthImage()
field.add_objs(green, random_loc)
field.add_objs(red, interaction_factor)
field.show()


# wound healing simulation
scratch = np.zeros((512, 512))
border = 300
scratch[:, :border] = 1


def wound_healing(obj, field, border=512, delta=2, forward=20):
    y, x = obj.loc
    t = obj.index
    br_y = y + np.random.randn() * 2*delta
    br_x = x + np.random.randn() * 2*delta
    ff = 1 - (border-x)/border
    ff_x = x + (np.sin(t)+1) * forward
    n_y = int(br_y)
    n_x = int((1-ff)*br_x + ff*ff_x)
    obj.loc = (n_y, n_x)
    return


img = imread('nuclei.tif')
nuclei = segment_channel(img, 2, sigma=1)
blue = FileImage.generate(45, nuclei, 'dapi', time_loc_func=wound_healing)

field = SynthImage()
field.add_objs(blue, even_grid, {'mask': scratch})
field.adjust_objs(random_loc_adjust)
field.adjust_objs(random_index_adjust)
field.show()

stack = SynthImageStack(field, n_fields=20)
stack.build_stack()
stack.save('wound_healing.tif')

