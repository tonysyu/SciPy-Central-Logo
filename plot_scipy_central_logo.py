"""
Plot concept for SciPy Central logo.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import UnivariateSpline
import bezier


class Circle(object):

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def point_from_angle(self, angle):
        r = self.radius
        # `angle` can be a scalar or 1D array: transpose twice for best results
        pts = r * np.array((np.cos(angle), np.sin(angle))).T + self.center
        return pts.T

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fc = kwargs.pop('fc', 'none')
        c = plt.Circle(self.center, self.radius, fc=fc, **kwargs)
        ax.add_patch(c)


class ScipyLogo(object):
    """Object to generate scipy logo

    Parameters
    ----------
    center : length-2 array
        the Scipy logo will be centered on this point.
    radius : float
        radius of logo
    flip : bool
        If True, flip logo vertically for normal plotting. Note that logo was
        from an image, which has an inverted y-axis.
    """
    CENTER = np.array((254, 246))
    RADIUS = 252.0
    THETA_START = 2.58
    THETA_END = -0.368

    def __init__(self, center=None, radius=None, flip=False):
        if center is None:
            if radius is None:
                center = self.CENTER
            else:
                center = np.array((radius, radius))
        self.center = center
        if radius is None:
            radius = self.RADIUS
        self.radius = radius

        # calculate end points of curve so that it lies exactly on circle
        logo_circle = Circle(self.CENTER, self.RADIUS)
        s_start = logo_circle.point_from_angle(self.THETA_START)
        s_end = logo_circle.point_from_angle(self.THETA_END)

        self.circle = Circle(self.center, self.radius)
        if flip:
            yc = self.CENTER[1]
            def flip_anchor(args):
                xy, theta, length = args
                x, y = xy
                y = 2 * yc - y
                theta = -theta
                return (x, y), theta, length
        else:
            flip_anchor = lambda x: x
        # note that angles are clockwise because of inverted y-axis
        self._anchors = [bezier.SymmetricAnchorPoint(*flip_anchor(t),
                                                     use_degrees=True)
                         for t in [(s_start,    -37, 90),
                                   ((144, 312),   7, 20),
                                   ((205, 375),  52, 50),
                                   ((330, 380), -53, 60),
                                   ((290, 260),-168, 50),
                                   ((217, 245),-168, 50),
                                   ((182, 118), -50, 60),
                                   ((317, 125),  53, 60),
                                   ((385, 198),  10, 20),
                                   (s_end,      -25, 60)]]
        # normalize anchors so they have unit radius and are centered at origin
        for a in self._anchors:
            a.pt = (a.pt - self.CENTER) / self.RADIUS
            a.length = a.length / self.RADIUS

    def snake_anchors(self):
        """Return list of SymmetricAnchorPoints defining snake curve"""
        anchors = []
        for a in self._anchors:
            pt = self.radius * a.pt + self.center
            length = self.radius * a.length
            anchors.append(bezier.SymmetricAnchorPoint(pt, a.theta, length))
        return anchors

    def snake_curve(self):
        """Return x, y coordinates of snake curve"""
        return bezier.curve_from_anchor_points(self.snake_anchors())

    def plot_snake_curve(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.snake_curve()
        ax.plot(x, y, 'k', solid_capstyle='butt', **kwargs)


def calc_arc(xystart, xyend, dn_frac=0.2):
    """Return arc connecting end points and going through point which is offset
    perpendicular to line by `dn_frac` * line length

    """
    ds = np.array([xyend[0] - xystart[0], xyend[1] - xystart[1]])
    length = np.sqrt(np.sum(ds**2))

    # Define arc with 3 points: end points, and midpoint offset perpendicularly
    # s = in direction connecting end points, n = normal to s
    s = [0, length * 0.1, length * 0.5, length]
    n = [0, -0.12 * length, -0.26 * length, 0]

    # Using UnivariateSpline is probably unnecessary, but might as well.
    quad = UnivariateSpline(s, n, k=2)

    # interpolated points
    s_pts = np.linspace(s[0], s[-1])
    n_pts = quad(s_pts)

    # rotate arc back to original coordinate system
    theta = np.arctan2(ds[1], ds[0])
    x0, y0 = xystart
    x_arc = x0 + s_pts * np.cos(theta) - n_pts * np.sin(theta)
    y_arc = y0 + s_pts * np.sin(theta) + n_pts * np.cos(theta)

    return x_arc, y_arc


def plot_arrows(ax):
    """Plot arrow pointing toward a "central" location.

    `plt.annotate` could be used for this---in theory---but currently doesn't
    handle dashed lines very well.
    """
    arrow_defs = [[(89e5, 6.8e5), (68e5, 83e5)],
                  [(114e5, 97e5), (65e5, 87e5)],
                  [(71e5, 124e5), (60e5, 87e5)],
                  [(17e5, 105e5), (59e5, 82e5)],
                  [(12e5, 27e5), (63e5, 79e5)]]

    kwargs = dict(color='lightsteelblue', alpha=0.6)
    # Tweak where the arrow body ends relative to the arrow tip
    idx_end = [-5, -7, -8, -8, -7]

    for (xystart, xyend), iend in zip(arrow_defs, idx_end):
        # arc
        x_arc, y_arc = calc_arc(xystart, xyend)
        ax.plot(x_arc[:iend], y_arc[:iend], '--', lw=4.5, **kwargs)

        arrow_tail = [x_arc[iend], y_arc[iend]]
        arrow_head = [x_arc[-1], y_arc[-1]]

        p = patches.FancyArrowPatch(arrow_tail, arrow_head, zorder=20,
                                    mutation_scale=20, shrinkA=20, **kwargs)
        ax.add_patch(p)


def plot_logo(lon, lat):
    globe = Basemap(projection='ortho', lon_0=lon, lat_0=lat, resolution='l')

    globe.fillcontinents(color=land_color, lake_color=water_color)
    globe.drawmapboundary(color='w', fill_color=water_color)
    globe.drawparallels(np.arange(-90.,120.,30.), color=line_color)
    globe.drawmeridians(np.arange(0.,420.,60.), color=line_color)

    ax = plt.gca()
    # There's got to be a better way to get the circle patch of the globe.
    circle = [p for p in ax.patches if 'Ellipse' in str(p)][0]

    logo = ScipyLogo(center=circle.center, radius=0.5*circle.width, flip=True)
    logo.plot_snake_curve(ax=ax, color=logo_color, linewidth=6,
                          alpha=logo_alpha)
    logo.circle.plot(color='w', linewidth=3, zorder=10)

    plot_arrows(ax)

    padding = 2e5 * np.array([-1, 1])
    ax.set_xlim(ax.get_xlim() + padding)
    ax.set_ylim(ax.get_ylim() + padding)
    ax.set_clip_on(False)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)


if __name__ == '__main__':
    fig = plt.figure(figsize=(2, 2))
    water_color = np.asarray(colors.hex2color(colors.cnames['royalblue']))
    land_color = 'cornflowerblue'
    line_color = water_color * 0.8
    logo_color = 'white'
    logo_alpha = 0.8

    cities = {
        'austin': {'lat': 30.3, 'lon': -97.7},
        'new_york': {'lat': 40.7, 'lon': -74},
        'paris': {'lat': 48.9, 'lon': 2.3},
        'san_francisco': {'lat': 32.7, 'lon': -117},
        'tokyo': {'lat': 35.7, 'lon': 140},
        'beijing': {'lat': 39.9, 'lon': 116},
        'munich': {'lat': 48.1, 'lon': 11.6},
        'berlin': {'lat': 52.5, 'lon': 13.4},
        'sao_paulo': {'lat': -23.5, 'lon': -46.6},
        'toronto': {'lat': 43.6, 'lon': -79.4},
        'johannesburg': {'lat': -26.2, 'lon': 28.1},
        'moscow': {'lat': 55.8, 'lon': 37.6},
        'mumbai': {'lat': 19, 'lon': 72.8},
        'london': {'lat': 51.5, 'lon': 0.1},
        'seoul': {'lat': 37.6, 'lon': 127},
        'mexico_city': {'lat': 19.1, 'lon': -99.4},
    }

    for city in cities:
        fig.clf()
        lon = cities[city]['lon']
        lat = cities[city]['lat']
        # Rotate so that the city coordinate is at the center of the arrows.
        lat -= 19
        plot_logo(lon, lat)
        plt.savefig('scipy_central_logo_{0}.png'.format(city), dpi=120)

