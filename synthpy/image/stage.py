import os
import numpy as np
import pandas as pd
from skimage.io import use_plugin, imshow, imsave

use_plugin('tifffile', 'imsave')


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
        draw += img[img_slice]
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