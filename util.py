import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import pandas as pd


class Timer:
    def __init__(self, msg='Time elapsed'):
        self.msg = msg
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f'{self.msg}: {duration:.2f}s')

class Event:
    __slots__ = 't', 'x', 'y', 'p'
    def __init__(self, t, x, y, p):
        self.t = t
        self.x = x
        self.y = y
        self.p = p
    def __repr__(self):
        return f'Event(t={self.t:.3f}, x={self.x}, y={self.y}, p={self.p})'


def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


def animate(images):
    fig = plt.figure()
    ims = []
    for image in images:
        im = plt.imshow(normalize_image(image), cmap='gray', vmin=0, vmax=1, animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    plt.close(ani._fig)
    return ani

def load_events(path_to_events, n_events=None):
    header = pd.read_csv(path_to_events, delim_whitespace=True, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int}, nrows=1)
    width, height = header.values[0]
    print(f'width, height: {width}, {height}')
    event_pd = pd.read_csv(path_to_events, delim_whitespace=True, header=None,
                              names=['t', 'x', 'y', 'p'],
                              dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int8},
                              engine='c', skiprows=1, nrows=n_events, memory_map=True)
    event_list = []
    for event in event_pd.values:
        t, x, y, p = event
        event_list.append(Event(t, int(x), int(y), -1 if p < 0.5 else 1))
    print('Loaded {:.2f}M events'.format(len(event_list) / 1e6))
    return event_list, width, height
