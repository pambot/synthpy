import numpy as np
from skimage.io import imread

from synthpy.image import Particle, FileImage
from synthpy.image import SynthImage, SynthImageStack
from synthpy.image import (brownian, random_loc, uniform_noise, 
                           even_grid, random_loc_adjust, random_index_adjust)

from synthpy.sequence import FASTA, BED


# test the image API
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
stack.save('misc/two_phase.tif')


# now make real examples
# keria's two-color interaction factor
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import find_objects

cell_img = imread('misc/clusters.tif')

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


img = imread('misc/nuclei.tif')
nuclei = segment_channel(img, 2, sigma=1)
blue = FileImage.generate(45, nuclei, 'dapi', time_loc_func=wound_healing)

field = SynthImage()
field.add_objs(blue, even_grid, {'mask': scratch})
field.adjust_objs(random_loc_adjust)
field.adjust_objs(random_index_adjust)
field.show()

stack = SynthImageStack(field, n_fields=20)
stack.build_stack()
stack.save('misc/wound_healing.tif')


# test the sequence API
fasta = FASTA()
fasta.from_file('misc/test.fa')

bed = BED(use_exons=True)
bed.from_file('misc/test.bed')

fastq = fasta.generate_fastq(bed, read_len=10, num_entries=20)

for fq in fastq:
    print(fq)
