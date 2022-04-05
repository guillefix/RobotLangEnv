import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box

n_colors = 10
def plot_colors(color):
    """
    Plots a sample of colors from the color x shade color class.

    Parameters
    ----------
    color: str
        Color in red, blue, green.
    shade: str
        Shade in light, dark.

    """
    color_class = Color(color)
    array = np.zeros([n_colors, n_colors, 3])
    for i in range(n_colors):
        for j in range(n_colors):
            array[i, j, :] = color_class.sample()
    plt.figure()
    plt.imshow(array)

max_min = 0.8
min_max = 0.2
class Color:
    def __init__(self, color):
        """
        Implements a color class characterized by a color and shade attributes.
        Parameters
        ----------
        color: str
            Color in red, blue, green.
        shade: str
            Shade in light, dark.
        """
        self.color = color
        if color == 'blue':
            self.space = Box(low=np.array([0, 0, max_min]), high=np.array([min_max, min_max, 1]))
        elif color == 'red':
            self.space = Box(low=np.array([max_min, 0, 0]), high=np.array([1, min_max, min_max]))
        elif color == 'green':
            self.space = Box(low=np.array([0, max_min, 0]), high=np.array([min_max, 1, min_max]))
        elif color == 'cyan':
            self.space = Box(low=np.array([0, max_min, max_min]), high=np.array([min_max, 1, 1]))
        elif color == 'yellow':
            self.space = Box(low=np.array([max_min, max_min, 0]), high=np.array([1, 1, min_max]))
        elif color == 'magenta':
            self.space = Box(low=np.array([max_min, 0, max_min]), high=np.array([1, min_max, 1]))
        elif color == 'black':
            self.space = Box(low=np.array([0, 0, 0]), high=np.array([min_max, min_max, min_max]))
        elif color == 'white':
            self.space = Box(low=np.array([max_min, max_min, max_min]), high=np.array([1, 1, 1]))
        else:
            raise NotImplementedError("color is 'red', 'blue' or 'green'")

    def contains(self, rgb):
        """
        Whether the class contains a given rgb code.
        Parameters
        ----------
        rgb: 1D nd.array of size 3

        Returns
        -------
        contains: Bool
            True if rgb code in given Color class.
        """
        return self.space.contains(rgb)

    def sample(self):
        """
        Sample an rgb code from the Color class

        Returns
        -------
        rgb: 1D nd.array of size 3
        """
        return np.random.uniform(self.space.low, self.space.high, 3)


def sample_color(color):
    """
    Sample an rgb code from the Color class

    Parameters
    ----------
    color: str
        Color in red, blue, green.
    shade: str
        Shade in light, dark.

    Returns
    -------
    rgb: 1D nd.array of size 3
    """
    color_class = Color(color)
    return color_class.sample()

def infer_color(rgb):
    rgb = rgb.astype(np.float32)
    for c in ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']:
        color_class = Color(c)
        # import pdb; pdb.set_trace()
        if color_class.contains(rgb):
            return c
    raise ValueError

if __name__ == '__main__':
    for c in ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']:
        plot_colors(c)
    plt.show()
