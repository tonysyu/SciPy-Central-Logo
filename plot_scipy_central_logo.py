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
            r = self.RADIUS
            def flip_anchor(args):
                xy, theta, length = args
                x, y = xy
                y = 2*yc - y
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
    #ds = np.diff([xystart, xyend], axis=0)[0]
    ds = np.array([xyend[0] - xystart[0], xyend[1] - xystart[1]])
    length = np.sqrt(np.sum(ds**2))

    # Define arc with 3 points: end points, and midpoint offset perpendicularly
    # s = in direction connecting end points, n = normal to s
    s = [0, length/2., length]
    n = [0, -0.2 * length, 0]

    # Using UnivariateSpline is probably unnecessary, but might as well.
    quad = UnivariateSpline(s, n, k=2)

    # interpolated points
    s_pts = np.linspace(s[0], s[-1])
    n_pts = quad(s_pts)

    # rotate arc back to original coordinage system
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
    arrow_defs = [[(93e5, 58e5), (68e5, 81e5)],
                  [(96e5, 98e5), (68e5, 87e5)],
                  [(69e5, 109e5), (63e5, 89e5)],
                  [(23e5, 92e5), (59e5, 86e5)],
                  [(25e5, 54e5), (62e5, 79e5)]]

    kwargs = dict(color='lightsteelblue', alpha=0.7)
    idx_end = [-12, -13, -18, -11, -9]
    for (xystart, xyend), iend in zip(arrow_defs, idx_end):
        # arc
        x_arc, y_arc = calc_arc(xystart, xyend)
        ax.plot(x_arc[4:iend], y_arc[4:iend], '--', lw=8, **kwargs)
        # arrow heads
        arrow_tail = [x_arc[iend], y_arc[iend]]
        arrow_head = [x_arc[-1], y_arc[-1]]
        p = patches.FancyArrowPatch(arrow_tail, arrow_head,
                                    mutation_scale=40, **kwargs)
        ax.add_patch(p)
        # start point
        x, y = xystart
        ax.plot(x, y, 'o', mec='none', markersize=12, **kwargs)


if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 6))
    water_color = np.asarray(colors.hex2color(colors.cnames['royalblue']))
    land_color = 'cornflowerblue'
    line_color = water_color * 0.8
    logo_color = 'white'
    logo_alpha = 0.8

    globe = Basemap(projection='ortho', lon_0=-30, lat_0=20, resolution='l')

    globe.fillcontinents(color=land_color, lake_color=water_color)
    globe.drawmapboundary(color='w', fill_color=water_color)
    globe.drawparallels(np.arange(-90.,120.,30.), color=line_color)
    globe.drawmeridians(np.arange(0.,420.,60.), color=line_color)

    globe.quiver

    # There's got to be a better way to get circle patch globe
    ax = plt.gca()
    circle = ax.patches[319]

    logo = ScipyLogo(center=circle.center, radius=0.5*circle.width, flip=True)
    logo.plot_snake_curve(ax=ax, color=logo_color, linewidth=15,
                          alpha=logo_alpha)
    logo.circle.plot(color='w', linewidth=3, zorder=10)

    plot_arrows(ax)

    padding = 2e5 * np.array([-1, 1])
    ax.set_xlim(ax.get_xlim() + padding)
    ax.set_ylim(ax.get_ylim() + padding)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    plt.savefig('scipy_central_logo.png')
    #plt.show()

