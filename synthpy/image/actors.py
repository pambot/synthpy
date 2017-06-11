import numpy as np
import pandas as pd
import abc
from uuid import uuid4
from skimage.draw import circle
from skimage.io import use_plugin, imread

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
        self.loc = loc # TODO: make sure is (int, int)
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
