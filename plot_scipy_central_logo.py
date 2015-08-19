#!/usr/bin/env python
"""
Plot concept for SciPy Central logo.

"""
import argparse
import os

import numpy as np
import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import UnivariateSpline
import bezier


CITIES = {
    'atlantic': {'lat': 30.0, 'lon': -30.0},
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

WATER_COLOR = np.asarray(colors.hex2color(colors.cnames['royalblue']))
LAND_COLOR = 'cornflowerblue'
LINE_COLOR = WATER_COLOR * 0.8
LOGO_COLOR = 'white'
LOGO_ALPHA = 0.8
LINE_WIDTH = 3
OUTLINE_WIDTH = LINE_WIDTH / 1.5
# Instead of centering the target city on the center of the logo, center the
# city the on space surrounded by the top curl in the 'S' of the logo.
CENTER_ON_TOP_OF_S = True


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
                         for t in [(s_start,     -37, 90),
                                   ((144, 312),    7, 20),
                                   ((205, 375),   52, 50),
                                   ((330, 380),  -53, 60),
                                   ((290, 260), -168, 50),
                                   ((217, 245), -168, 50),
                                   ((182, 118),  -50, 60),
                                   ((317, 125),   53, 60),
                                   ((385, 198),   10, 20),
                                   (s_end,       -25, 60)]]
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
        ax.plot(x_arc[:iend], y_arc[:iend], '--', lw=OUTLINE_WIDTH, **kwargs)

        arrow_tail = [x_arc[iend], y_arc[iend]]
        arrow_head = [x_arc[-1], y_arc[-1]]

        p = patches.FancyArrowPatch(arrow_tail, arrow_head, zorder=20,
                                    mutation_scale=20, shrinkA=20, **kwargs)
        ax.add_patch(p)


def plot_logo(ax, city, with_arrows=False):
    location = CITIES[city]
    lon = location['lon']
    lat = location['lat']
    if CENTER_ON_TOP_OF_S:
        lat -= 19

    globe = Basemap(projection='ortho', lon_0=lon, lat_0=lat, resolution='l')

    globe.fillcontinents(color=LAND_COLOR, lake_color=WATER_COLOR)
    globe.drawmapboundary(color='w', fill_color=WATER_COLOR)
    globe.drawparallels(np.arange(-90.0, 120.0, 30.), color=LINE_COLOR)
    globe.drawmeridians(np.arange(0.0, 420.0, 60.), color=LINE_COLOR)

    # There's got to be a better way to get the circle patch of the globe.
    circle = [p for p in ax.patches if 'Ellipse' in str(p)][0]

    logo = ScipyLogo(center=circle.center, radius=0.5*circle.width, flip=True)
    logo.plot_snake_curve(ax=ax, color=LOGO_COLOR, linewidth=LINE_WIDTH,
                          alpha=LOGO_ALPHA)
    logo.circle.plot(color='w', linewidth=OUTLINE_WIDTH, zorder=10)

    if with_arrows:
        plot_arrows(ax)

    padding = 2e5 * np.array([-1, 1])
    ax.set_xlim(ax.get_xlim() + padding)
    ax.set_ylim(ax.get_ylim() + padding)
    ax.set_clip_on(False)


def add_title(ax):
    font_path = os.path.join('~', 'Downloads', 'Roboto', 'Roboto-Light.ttf')
    font_path = os.path.expanduser(font_path)
    kwargs = {}
    if os.path.exists(font_path):
        kwargs['fontproperties'] = fm.FontProperties(fname=font_path)
    else:
        url = 'https://www.google.com/fonts#UsePlace:use/Collection:Roboto'
        msg = "Expected to find: {}\n"
        msg += "Download and unzip Roboto font from {!r}"
        print msg.format(font_path, url)

    ax.text(
        1.1, 0.45,
        'SciPyCentral',
        ha='left', va='center', transform=ax.transAxes,
        alpha=1.0, color=WATER_COLOR, fontsize=50, **kwargs
    )


def save_image(city):
    plt.savefig('scipy_central_logo_{0}.png'.format(city), dpi=120)


def plot_logo_only(city, action='show', **kwargs):
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plot_logo(ax, city, **kwargs)
    if action == 'show':
        plt.show()
    elif action == 'save':
        save_image(city)


def plot_banner(city, action='show', **kwargs):
    fig = plt.figure(figsize=(5, 1))
    ax = fig.add_axes([0.0, 0.075, 0.2, 0.85])
    plot_logo(ax, city, **kwargs)
    add_title(ax)
    if action == 'show':
        plt.show()
    elif action == 'save':
        save_image(city)


def main():
    logo_types = {
        'logo_only': plot_logo_only,
        'banner': plot_banner,
    }
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument('type', choices=logo_types.keys(), default='banner',
                        nargs='?', help="Type of logo to plot")
    parser.add_argument('--city', choices=CITIES.keys(), default='atlantic',
                        help="The city that's the focus of the logo.")
    parser.add_argument('--with-arrows', action='store_true',
                        help="Display arrows spiraling toward to city.")
    parser.add_argument('--action', choices=['show', 'save'], default='show')

    args = parser.parse_args()
    plot = logo_types[args.type]
    plot(args.city, action=args.action, with_arrows=args.with_arrows)


if __name__ == '__main__':
    main()
