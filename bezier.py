import numpy as np


class QuadraticCurve(object):
    """Anchor point in a parametric curve with symmetric handles

    Parameters
    ----------
    pt : length-2 sequence
        (x, y) coordinates of anchor point
    theta : float
        angle of control handle
    length : float
        half-length of symmetric control handle. Each control point is `length`
        distance away from the anchor point.
    use_degrees : bool
        If True, convert input `theta` from degrees to radians.
    """

    def __init__(self, pt, theta, length, use_degrees=False):
        self.pt = pt
        if use_degrees:
            theta = theta * np.pi / 180
        self.theta = theta
        self.length = length

    def control_points(self):
        """Return control points for symmetric handles

        The first point is in the direction of theta and the second is directly
        opposite. For example, if `theta = 0`, then the first `p1` will be
        directly to the right of the anchor point, and `p2` will be directly
        to the left.
        """
        theta = self.theta
        offset = self.length * np.array([np.cos(theta), np.sin(theta)])
        p1 = self.pt + offset
        p2 = self.pt - offset
        return p1, p2

    def __repr__(self):
        v = (self.pt, self.theta * 180/np.pi, self.length)
        return 'SymmetricAnchorPoint(pt={0}, theta={1}, length={2})'.format(*v)


class SymmetricAnchorPoint(object):
    """Anchor point in a parametric curve with symmetric handles

    Parameters
    ----------
    pt : length-2 sequence
        (x, y) coordinates of anchor point
    theta : float
        angle of control handle
    length : float
        half-length of symmetric control handle. Each control point is `length`
        distance away from the anchor point.
    use_degrees : bool
        If True, convert input `theta` from degrees to radians.
    """

    def __init__(self, pt, theta, length, use_degrees=False):
        self.pt = pt
        if use_degrees:
            theta = theta * np.pi / 180
        self.theta = theta
        self.length = length

    def control_points(self):
        """Return control points for symmetric handles

        The first point is in the direction of theta and the second is directly
        opposite. For example, if `theta = 0`, then the first `p1` will be
        directly to the right of the anchor point, and `p2` will be directly
        to the left.
        """
        theta = self.theta
        offset = self.length * np.array([np.cos(theta), np.sin(theta)])
        p1 = self.pt + offset
        p2 = self.pt - offset
        return p1, p2

    def __repr__(self):
        v = (self.pt, self.theta * 180/np.pi, self.length)
        return 'SymmetricAnchorPoint(pt={0}, theta={1}, length={2})'.format(*v)


def curve_from_anchor_points(anchors):
    """Return curve from a list of SymmetricAnchorPoints"""
    assert len(anchors) > 1
    bezier_pts = []
    for a in anchors:
        c1, c2 = a.control_points()
        bezier_pts.extend([c2, a.pt, c1])
    # clip control points from ends
    bezier_pts = bezier_pts[1:-1]
    x, y = [], []
    # every third point is an anchor point
    for i in range(0, len(bezier_pts)-1, 3):
        xi, yi = cubic_curve(*bezier_pts[i:i+4])
        x.append(xi)
        y.append(yi)
    return np.hstack(x), np.hstack(y)


def quadratic_curve(p0, p1, p2, npts=20):
    """Return points on a quadratic Bezier curve

    Parameters
    ----------
    p0, p2 : length-2 sequences
        end points of curve
    p1 : length-2 sequence
        control point of curve
    npts : int
        number of points to return (including end points)

    Returns
    -------
    x, y : arrays
        points on cubic curve
    """
    t = np.linspace(0, 1, npts)[:, np.newaxis]
    # quadratic bezier curve from http://en.wikipedia.org/wiki/Bezier_curve
    b = (1-t)**2 * p0 + 2*t*(1-t) * p1 + t**2 * p2
    x, y = b.transpose()
    return x, y


def cubic_curve(p0, p1, p2, p3, npts=20):
    """Return points on a cubic Bezier curve

    Parameters
    ----------
    p0, p3 : length-2 sequences
        end points of curve
    p1, p2 : length-2 sequences
        control points of curve
    npts : int
        number of points to return (including end points)

    Returns
    -------
    x, y : arrays
        points on cubic curve
    """
    t = np.linspace(0, 1, npts)[:, np.newaxis]
    # cubic bezier curve from http://en.wikipedia.org/wiki/Bezier_curve
    b = (1-t)**3 * p0 + 3*t*(1-t)**2 * p1 + 3*t**2*(1-t) * p2 + t**3 * p3
    x, y = b.transpose()
    return x, y


