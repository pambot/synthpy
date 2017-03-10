import os
import numpy as np
import pandas as pd
from uuid import uuid4
from PIL import Image, ImageDraw, ImageChops
from PIL.TiffImagePlugin import AppendingTiffWriter


# base class of simulation objects
class SynthImageObject(object):
    def __init__(self):
        self.oid = uuid4()
        self.index = 0
        self.label = None
        self.loc = (0, 0)
        self.shape = (1, 1)
        self.time_func = lambda obj: obj
        self.time_params = {}
        self.trans_func = lambda obj: obj
        self.trans_params = {}
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
    
    def occupied_coords(self):
        coords = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.mask[x, y]:
                    coords.append((self.loc[0]+x, self.loc[1]+y))
        return coords


# specific types of simulation objects
class Particle(SynthImageObject):
    def __init__(self, label, shape=(5, 5), color=(255, 255, 255)):
        super().__init__()
        self.img = Image.new('RGBA', shape)
        draw = ImageDraw.Draw(self.img)
        draw.ellipse((1, 1, shape[0]-1, shape[1]-1), fill=color)
        self.mask = make_mask(self.img)
        self.label = label
        self.shape = shape


class Picture(SynthImageObject):
    def __init__(self, filename, label):
        super().__init__()
        self.img = Image.open(filename)
        self.mask = make_mask(self.img)
        self.label = label
        self.shape = self.img.size


class PSF(SynthImageObject):
    def __init__(self, label):
        super().__init__()


# class for simulation frame
class SynthImage(object):
    def __init__(self, shape=(512, 512)):
        self.index = 0
        self.shape = shape
        self.img = Image.new('RGBA', self.shape, color=(0,0,0,255))
        self.objs = []
        self.occupied = {}
        self.noise_func = lambda img: img
        self.noise_params = {}
    
    def __str__(self):
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
        return str(df)
    
    def add_objs(self, objs, loc_func, **kwargs):
        overlap = Image.new('RGBA', self.shape)
        for obj in objs:
            obj.loc = loc_func(self, **kwargs)
            overlap.paste(obj.img, box=obj.loc, mask=obj.img)
            self.objs.append(obj)
            try:
                self.occupied[obj.label].update(obj.occupied_coords())
            except KeyError:
                self.occupied[obj.label] = set([])
        self.img = ImageChops.add(self.img, overlap)
        return
    
    def update_objs(self):
        self.img = Image.new('RGBA', self.shape, color=(0,0,0,255))
        overlap = Image.new('RGBA', self.shape)
        self.occupied = {}
        if not self.objs:
            raise ValueError('No objects have been added.')
        else:
            for obj in self.objs:
                obj.index += 1
                obj = obj.time_func(obj, **obj.time_params)
                obj = obj.trans_func(obj, **obj.trans_params)
                overlap.paste(obj.img, box=obj.loc, mask=obj.img)
                try:
                    self.occupied[obj.label].update(obj.occupied_coords())
                except KeyError:
                    self.occupied[obj.label] = set([])
        self.img = ImageChops.add(self.img, overlap)
        return
    
    def set_noise_func(self, noise_func, kwargs={}):
        self.noise_func = noise_func
        self.noise_params = kwargs
        return
    
    def apply_noise(self):
        self.img = self.noise_func(self.img, **self.noise_params)
        return
    
    def show(self):
        self.img.show()
        return
    
    def save(self, filename='image.tif'):
        if os.path.exists(filename):
            os.unlink(filename)
        self.img.save(filename)
        return


# class for image stack
class SynthImageStack(object):
    def __init__(self, frame, n_frames):
        self.stack = []
        self.frame = frame
        self.n_frames = n_frames
    
    #def __str__(self):
        #return
    
    def build_stack(self):
        for n in range(self.n_frames):
            self.frame.update_objs()
            self.frame.apply_noise()
            self.stack.append(self.frame.img)
            self.frame.index += 1
        return
    
    def save(self, filename='image.tif'):
        if os.path.exists(filename):
            os.unlink(filename)
        with AppendingTiffWriter(filename) as tf:
            for stack_img in self.stack:
                stack_img.save(tf)
                tf.newFrame()
        return


# modular functions
def brownian(obj, dt=1, delta=5):
    x, y = obj.loc
    x += np.random.randn() * 2*delta*dt
    y += np.random.randn() * 2*delta*dt
    obj.loc = (int(x), int(y))
    return obj


def random_loc(frame, mask=np.array([])):
    x_high, y_high = frame.shape
    x_loc = np.random.randint(0, x_high)
    y_loc = np.random.randint(0, y_high)
    if mask.any():
        coords = np.argwhere(mask)
        x_loc, y_loc = coords[np.random.randint(0, len(coords))]
    return x_loc, y_loc


def uniform_noise(frame):
    x_high, y_high = frame.size
    xs = np.random.randint(0, x_high, size=1000)
    ys = np.random.randint(0, y_high, size=1000)
    draw = ImageDraw.Draw(frame)
    draw.point(list(zip(xs, ys)), fill=(127, 127, 127))
    return frame


# helper functions
def generate_objs(ObjClass, n, label, class_params={},
                  time_func=lambda obj: obj, time_params={}, 
                  trans_func=lambda obj: obj, trans_params={}):
    objs = []
    for i in range(n):
        obj = ObjClass(label, **class_params)
        obj.set_time_func(time_func, time_params)
        obj.set_trans_func(trans_func, trans_params)
        objs.append(obj)
    return objs


def make_mask(img_name):
    if type(img_name) is Image.Image:
        mask = np.asarray(img_name.convert(mode='1')).T
    elif type(img_name) is str:
        mask = np.asarray(Image.open(img_name).convert(mode='1')).T
    return mask


# test the API
# two types of brownian motion particles with masks
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


red_dots = generate_objs(Particle, 200, 'foo', time_func=brownian, 
                         class_params={'color': (255,0,0)})
blue_dots = generate_objs(Particle, 200, 'bar', time_func=two_phase_brownian, 
                          class_params={'color': (0,0,255)})

ying = make_mask('ying.png')
yang = make_mask('yang.png')

frame = SynthImage()
frame.add_objs(red_dots, random_loc, mask=ying)
frame.add_objs(blue_dots, random_loc, mask=yang)
frame.set_noise_func(uniform_noise)

stack = SynthImageStack(frame, 10)
stack.build_stack()

print(stack.frame)
stack.save('yingyang.tif')


# attraction factor simplified demo
def attract_loc(frame, target='green'):
    x_high, y_high = frame.shape
    x_loc = np.random.randint(0, x_high)
    y_loc = np.random.randint(0, y_high)
    while (x_loc+15, y_loc+15) not in frame.occupied[target]:
        x_loc = np.random.randint(0, x_high)
        y_loc = np.random.randint(0, y_high)
    return x_loc, y_loc

green = generate_objs(Particle, 40, 'green', class_params={'color': (0,255,0), 
                    'shape': (30, 30)})
purple = generate_objs(Particle, 40, 'red', class_params={'color': (255,0,255), 
                    'shape': (30, 30)})

frame1 = SynthImage()
frame1.add_objs(green, random_loc)
frame1.add_objs(purple, random_loc)
frame1.save('two_color_random.tif')

frame2 = SynthImage()
frame2.add_objs(green, random_loc)
frame2.add_objs(purple, attract_loc)
frame2.save('two_color_attract.tif')

