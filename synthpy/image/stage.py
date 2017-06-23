import os
import numpy as np
import pandas as pd
from skimage.io import use_plugin, imshow, imsave

use_plugin('tifffile', 'imsave')


class SynthImage(object):
    """A class for generating the stage for the imaging module.
    
    Parameters
    ----------
    shape : tuple, optional (default=(512, 512, 3))
        A tuple of height `m` and wdith `n`, with 3 indicating an RGB image.
    
    Attributes
    ----------
    index : int (default=0)
        Bind a time index to the object, allowing time-dependent behaviour.
    
    img : np.ndarray (default=np.zeros(shape, dtype=np.uint8))
        A `m` by `n` by 3 Numpy array that constitutes the image representation
        of the synthetic stage object.
    
    actors : list (default=[])
        A list of actor objects associated with the stage.
    
    noise_func : function (default=lambda img: img)
        A hook for attaching a noise generating function
    
    noise_params : dict (default={})
        A dictionary of parameters for feeding into `noise_func`
    
    """
    def __init__(self, shape=(512, 512, 3)):
        self.index = 0
        self.img = np.zeros(shape, dtype=np.uint8)
        self.actors = []
        self.noise_func = lambda img: img
        self.noise_params = {}
    
    def __str__(self):
        return str(self.get_data())
    
    def get_data(self):
        """"A method for organizing information about attached actor objects
        into a `pandas.DataFrame` object.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        df : pandas.DataFrame
            A `pandas.DataFrame` of associated actors' labels, locations, etc.
        
        """
        cols = ['label', 'x_loc', 'y_loc']
        if self.actors:
            data = []
            oids = []
            for actor in self.actors:
                data.append([actor.label, actor.loc[0], actor.loc[1]])
                oids.append(actor.oid)
            df = pd.DataFrame(columns=cols, index=oids, data=data)
        else:
            df = pd.DataFrame(columns=cols)
        return df
    
    @property
    def shape(self):
        return self.img.shape
    
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
    def _overflow_slices(cls, img, stage, loc):
        fy, fx = stage.shape[:2]
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
    def _embed_img(cls, img, stage, loc):
        stage_slice, img_slice = cls._overflow_slices(img, stage, loc)
        draw = np.zeros((img_slice[0].stop-img_slice[0].start, 
                         img_slice[1].stop-img_slice[1].start, 3))
        draw += stage[stage_slice]
        draw += img[img_slice]
        draw[draw > 255] = 255
        stage[stage_slice] = draw.astype(np.uint8)
        return stage
    
    def add_actors(self, actors, init_loc_func, init_loc_params={}):
        """A method for attaching a list of actors to the stage object.
        
        Parameters
        ----------
        actors : list (default=[])
            A list of actor objects associated with the stage.
        
        init_loc_func : function, optional (default=lambda actor: actor)
            A hook for binding custom localization initialization. If left unset,
            it will result in a synthetic object that centers at `loc=(0, 0)`.
        
        init_loc_params: dict, optional (default={})
            A dictionary of parameters to feed into `init_loc_func`. Pre-built
            Synthpy functions don't often have additional paramters, but custom ones
            often will.
        
        Returns
        -------
        None
        
        """
        for actor in actors:
            actor.set_init_loc_func(init_loc_func, init_loc_params)
            actor.init_loc_func(actor, self, **actor.init_loc_params)
            self.img = self._embed_img(actor.img, self.img, actor.loc)
            self.actors.append(actor)
        return
    
    def adjust_actors(self, adjust_func, adjust_params={}):
        """A method to adjust some or all properties of the list of actors attached
        to the stage without saving the `adjust_func` to the actors' state.
        
        Parameters
        ----------
        adjust_func : function, optional (default=lambda actor: actor)
            A hook for binding custom actor property adjustments.
        
        adjust_params: dict, optional (default={})
            A dictionary of parameters to feed into `adjust_func`.
        
        Returns
        -------
        None
        
        """
        self.img = np.zeros(self.shape, dtype=np.uint8)
        if not self.actors:
            raise ValueError('No objects have been added.')
        else:
            for actor in self.actors:
                adjust_func(actor, self, **adjust_params)
                self.img = self._embed_img(actor.img, self.img, actor.loc)
        return
    
    def update_actors(self):
        """A method to adjust some or all properties of the list of actors attached
        to the stage using the actors' preset `time_loc_func`.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        self.img = np.zeros(self.shape, dtype=np.uint8)
        if not self.actors:
            raise ValueError('No objects have been added.')
        else:
            for actor in self.actors:
                actor.time_loc_func(actor, self, **actor.time_loc_params)
                self.img = self._embed_img(actor.img, self.img, actor.loc)
                actor.index += 1
        return   
    
    def set_noise_func(self, noise_func, kwargs={}):
        """A method for setting the noise-generating function for the stage.
        
        Parameters
        ----------
        noise_func : function (default=lambda img: img)
            A hook for attaching a noise generating function
        
        noise_params : dict (default={})
            A dictionary of parameters for feeding into `noise_func`
        
        Returns
        -------
        None
        
        """
        self.noise_func = noise_func
        self.noise_params = kwargs
        return
    
    def apply_noise(self):
        """Runs the preset noise-generating function `noise_func` attached to the
        stage.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        self.noise_func(self, **self.noise_params)
        return
    
    def show(self):
        """Shows the stage image.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        imshow(self.img)
        return
    
    def save(self, fname='image.tif'):
        """Saves the stage image.
        
        Parameters
        ----------
        fname : str, optional (default='image.tif')
            The filename that the image will be saved under.
        
        Returns
        -------
        None
        
        """
        if os.path.exists(fname):
            os.unlink(fname)
        imsave(fname, self.img)
        return


class SynthImageStack(object):
    """A class to hold a stack of `SynthImage` stages to generate time series data.
    This class uses all the functions attached to the actors and stages to generate
    the logical conclusion of their function behaviour.
    
    Parameters
    ----------
    stage : SynthImage
        A `SynthImage` object stage. The first frame of the time series.
    
    n_stages : int
        The number of frames you want to generate.
    
    Attributes
    ----------
    stage : SynthImage
        A `SynthImage` object stage. The first frame of the time series.
    
    n_stages : int
        The number of frames you want to generate.
    
    stack : np.ndarray
        A `(n_stages,) + stage.shape` shaped Numpy array that contains the actual
        images generated in the time series.
    
    """
    def __init__(self, stage, n_stages):
        self.stack = np.zeros((n_stages,) + stage.shape, dtype=np.uint8)
        self.stage = stage
        self.n_stages = n_stages
    
    def build_stack(self):
        """A method for generating the time series once the `SynthImageStack` is
        initialized.
        
        Paramters
        ---------
        None
        
        Returns
        -------
        None
        
        """
        for n in range(self.n_stages):
            self.stage.update_actors()
            self.stage.apply_noise()
            self.stack[n, :, :, :] = self.stage.img
            self.stage.index += 1
        return
    
    def save(self, fname='image.tif'):
        """Saves the time series as a multi-frame TIFF image.
        
        Parameters
        ----------
        fname : str, optional (default='image.tif')
            The filename that the images will be saved under.
        
        Returns
        -------
        None
        
        """
        imsave(fname, self.stack)
        return