from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time


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


class EventData:
    def __init__(self, event_list, width, height):
        self.event_list = event_list
        self.width = width
        self.height = height


def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


def animate(images, fig_title=''):
    fig = plt.figure(figsize=(0.1, 0.1))  # don't take up room initially
    fig.suptitle(fig_title)
    fig.set_size_inches(7.2, 5.4, forward=False)  # resize but don't update gui
    ims = []
    for image in images:
        im = plt.imshow(normalize_image(image), cmap='gray', vmin=0, vmax=1, animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    plt.close(ani._fig)
    return ani


def load_events(path_to_events, n_events=None):
    print('Loading events...')
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
    return EventData(event_list, width, height)


def plot_3d(event_data, n_events=-1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, t, c = [], [], [], []
    for e in event_data.event_list[:int(n_events)]:
        x.append(e.x)
        y.append(e.y)
        t.append(e.t * 1e3)
        c.append('r' if e.p == 1 else 'b')
    ax.scatter(t, x, y, c=c, marker='.')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_zlim(*ax.get_zlim()[::-1])  # reverse 'y' image axis


def event_slice(event_data, start=0, duration_ms=30):
    events, height, width = event_data.event_list, event_data.height, event_data.width
    mask = np.zeros((height, width), dtype=np.int8)
    start_idx = int(start * (len(events) - 1))
    end_time = events[start_idx].t + duration_ms / 1000.0
    for e in events[start_idx:]:
        mask[e.y, e.x] = e.p
        if e.t >= end_time:
            break
    img_rgb = np.ones((height, width, 3), dtype=np.uint8) * 255
    img_rgb[mask == -1] = (255, 0, 0)
    img_rgb[mask == 1] = (0, 0, 255)
    fig = plt.figure(figsize=(7.2, 5.4))
    plt.imshow(img_rgb)
