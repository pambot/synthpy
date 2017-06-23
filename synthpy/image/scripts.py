import numpy as np


def brownian(actor, stage, dt=1, delta=5):
    """A pre-built possible function for inserting into an actor's `time_loc_func`
    hook. This function will produce brownian motion behaviour over time for the
    actor it is attached to.
    
    Parameters
    ----------
    actor : SynthImageObj
        The actor this function will be attached to.
    
    stage : SynthImage
        The stage that the actors are attached to.
    
    Returns
    -------
    None
    
    """
    y, x = actor.loc
    y += np.random.randn() * 2*delta*dt
    x += np.random.randn() * 2*delta*dt
    actor.loc = (int(y), int(x))
    return


def uniform_noise(stage):
    ys = np.random.randint(0, stage.img.shape[0], size=100)
    xs = np.random.randint(0, stage.img.shape[1], size=100)
    noise = np.zeros(stage.img.shape, dtype=np.uint8)
    noise[ys, xs, :] = 127
    stage.img += noise
    return


def random_loc(actor, stage, mask=None):
    y_high, x_high = stage.shape[:2]
    y_loc = np.random.randint(0, x_high)
    x_loc = np.random.randint(0, y_high)
    if type(mask) == np.ndarray and mask.shape == stage.shape[:2]:
        while not mask[y_loc, x_loc].all():
            y_loc = np.random.randint(0, y_high)
            x_loc = np.random.randint(0, x_high)
    actor.loc = (y_loc, x_loc)
    return


def even_grid(actor, stage, spacer=50, mask=None):
    y_stage, x_stage = stage.img.shape[:2]
    if not stage.actors:
        y_loc, x_loc = spacer, spacer
    else:
        last_actor = stage.actors[-1]
        y_loc, x_loc = last_actor.loc
        placed = False
        while not placed:
            if x_loc + spacer < x_stage:
                x_loc += spacer
            elif y_loc + spacer < y_stage:
                y_loc += spacer
                x_loc = spacer
            else:
                raise ValueError('Grid dimensions exceeded.')
            if type(mask) == np.ndarray and mask.shape == stage.shape[:2]:
                if mask[y_loc, x_loc].all():
                    placed = True
                else:
                    continue
            else:
                placed = True
    actor.loc = (y_loc, x_loc)
    return


def random_loc_adjust(actor, stage, factor=10):
    y_adj = np.random.randn() * factor
    x_adj = np.random.randn() * factor
    actor.loc = (int(actor.loc[0] + y_adj), int(actor.loc[1] + x_adj))
    return


def random_index_adjust(actor, stage, n=10):
    actor.index = np.random.randint(0, n)
    return

