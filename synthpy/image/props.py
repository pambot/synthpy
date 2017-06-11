import numpy as np


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

