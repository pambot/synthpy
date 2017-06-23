import numpy as np
import pandas as pd
import abc
from uuid import uuid4
from skimage.draw import circle
from skimage.io import use_plugin, imread

use_plugin('tifffile', 'imsave')


class SynthImageObj(object):
    """Abstract class for generating synthetic objects."""
    
    __metaclass__  = abc.ABCMeta
    
    def __init__(self, label, img=np.zeros((0, 0, 3), dtype=np.uint8), index=0, loc=(0, 0), 
                 time_loc_func=lambda actor, stage: (actor, stage), time_loc_params={}, 
                 init_loc_func=lambda actor, stage: (actor, stage), init_loc_params={}):
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
        """Auto-generates a number of instances of the attached class.
        
        Parameters
        ----------
        n : int
            The number of desired instances.
        
        *args:
            Arguments for instantiating the attached class.
        
        **kwargs:
            Keyword arguments for instantiating the attached class.
        
        Returns
        -------
        actors : list of instances of the attached class.
        
        """
        actors = []
        for i in range(n):
            actor = cls(*args, **kwargs)
            actors.append(actor)
        return actors


class Particle(SynthImageObj):
    """A class for generating circular synthetic objects. This is
    mostly a testing stand-in for when you don't yet have data for the
    FileImage class.
    
    Parameters
    ----------
    label : str, int
        The label to distinguish a set of objects in an experiment.
    
    shape : tuple, optional (default=(9, 9, 3))
        A tuple of height `m` and wdith `n`, with 3 indicating an RGB image.
    
    color : tuple, optional (default=(255, 255, 255))
        A (0-255, 0-255, 0-255) RGB tuple to color the circle.
    
    index : int, optional (default=0)
        Bind a time index to the object, allowing time-dependent behaviour.
    
    loc : tuple, optional (default=(0, 0))
        Bind a location on the stage space to the object
    
    time_loc_func : function, optional (default=lambda actor: actor)
        A hook for binding custom time-dependent behaviour. If left unset, 
        it will result in a synthetic object that doesn't move or change 
        over time.
    
    time_loc_params: dict, optional (default={})
        A dictionary of parameters to feed into `time_loc_func`. Pre-built
        Synthpy functions don't often have additional paramters, but custom ones
        often will.
    
    init_loc_func : function, optional (default=lambda actor: actor)
        A hook for binding custom localization initialization. If left unset,
        it will result in a synthetic object that centers at `loc=(0, 0)`.
    
    init_loc_params: dict, optional (default={})
        A dictionary of parameters to feed into `init_loc_func`. Pre-built
        Synthpy functions don't often have additional paramters, but custom ones
        often will.
    
    Attributes
    ----------
    label : str, int
        The label to distinguish a set of objects in an experiment.
    
    shape : tuple, optional (default=(9, 9, 3))
        A tuple of height `m` and wdith `n`, with 3 indicating an RGB image.
    
    img : np.ndarray
        A `m` by `n` by 3 Numpy array that constitutes the image representation
        of the synthetic object.
    
    index : int, optional (default=0)
        Bind a time index to the object, allowing time-dependent behaviour.
    
    loc : tuple, optional (default=(0, 0))
        Bind a location on the stage space to the object
    
    time_loc_func : function, optional (default=lambda actor: actor)
        The function loaded into the `time_loc_func` hook.
    
    time_loc_params: dict, optional (default={})
        The parameters loaded into `time_loc_func`.
    
    init_loc_func : function, optional (default=lambda actor: actor)
        The function loaded into the `init_loc_func` hook.
    
    init_loc_params: dict, optional (default={})
        The parameters loaded into `init_loc_func`.
    
    """
    def __init__(self, label, shape=(9, 9, 3), color=(255,255,255), **kwargs):
        super().__init__(label, **kwargs)
        self.shape = shape
        self.img = np.zeros(shape, dtype=np.uint8)
        radius = round(shape[0]/2)
        rr, cc = circle(radius, radius, shape[0]-radius, shape=shape[:2])
        self.img[rr, cc, :] = np.array(color)


class FileImage(SynthImageObj):
    """A class for generating synthetic objects from segmented images. A
    `FileImage` object should be derived from pieces of real experimental image
    files that represent an object whose movement across space and time is being
    simulated.
    
    Parameters
    ----------
    label : str, int
        The label to distinguish a set of objects in an experiment.
    
    index : int, optional (default=0)
        Bind a time index to the object, allowing time-dependent behaviour.
    
    loc : tuple, optional (default=(0, 0))
        Bind a location on the stage space to the object
    
    time_loc_func : function, optional (default=lambda actor: actor)
        A hook for binding custom time-dependent behaviour. If left unset, 
        it will result in a synthetic object that doesn't move or change 
        over time.
    
    time_loc_params: dict, optional (default={})
        A dictionary of parameters to feed into `time_loc_func`. Pre-built
        Synthpy functions don't often have additional paramters, but custom ones
        often will.
    
    init_loc_func : function, optional (default=lambda actor: actor)
        A hook for binding custom localization initialization. If left unset,
        it will result in a synthetic object that centers at `loc=(0, 0)`.
    
    init_loc_params: dict, optional (default={})
        A dictionary of parameters to feed into `init_loc_func`. Pre-built
        Synthpy functions don't often have additional paramters, but custom ones
        often will.
    
    Attributes
    ----------
    label : str, int
        The label to distinguish a set of objects in an experiment.
    
    shape : tuple, optional (default=(9, 9, 3))
        A tuple of height `m` and wdith `n`, with 3 indicating an RGB image.
    
    img : np.ndarray
        A `m` by `n` by 3 Numpy array that constitutes the image representation
        of the synthetic object.
    
    index : int, optional (default=0)
        Bind a time index to the object, allowing time-dependent behaviour.
    
    loc : tuple, optional (default=(0, 0))
        Bind a location on the stage space to the object
    
    time_loc_func : function, optional (default=lambda actor: actor)
        The function loaded into the `time_loc_func` hook.
    
    time_loc_params: dict, optional (default={})
        The parameters loaded into `time_loc_func`.
    
    init_loc_func : function, optional (default=lambda actor: actor)
        The function loaded into the `init_loc_func` hook.
    
    init_loc_params: dict, optional (default={})
        The parameters loaded into `init_loc_func`.
    
    """
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
        """Auto-generates a number of instances of the `FileImage` class.
        
        Parameters
        ----------
        n : int
            The number of desired instances.
        
        label : str, int
            The label to distinguish a set of objects in an experiment.
        
        file_img_array : list, tuple of (m, n, 3) shape np.ndarray
            An array of RGB images to become synthetic objects.
        
        *args:
            Arguments for instantiating the `FileImage` class.
        
        **kwargs:
            Keyword arguments for instantiating the `FileImage` class.
        
        Returns
        -------
        actors : list of instances of the `FileImage` class.
        
        """
        random_imgs = np.random.choice(file_img_array, size=n, replace=False)
        actors = []
        for f_img in random_imgs:
            actor = cls(f_img, label, **kwargs)
            actors.append(actor)
        return actors
