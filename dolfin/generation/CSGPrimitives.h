// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-11
// Last changed: 2012-04-11

#ifndef __CSG_PRIMITIVES_H
#define __CSG_PRIMITIVES_H

namespace dolfin
{

class Cylinder(Primitive):

    # The non-infinite cylinder class is constructed from other
    # primitives but needs to be a class so we can can rotate it. The
    # logic for building the cylinder from other primitives is instead
    # handled in the _write function. An alternative approach is to
    # define a function instead of a class as is done for some of the
    # other geometric objects.

    def __init__(self, x0, x1, x2, y0, y1, y2, r):
        "A cylinder from x to y with radius r"

        self.x = (float(x0), float(x1), float(x2))
        self.y = (float(y0), float(y1), float(y2))
        self.r = r

    def __str__(self):
        return "Cylinder from x = %s to y = %s with radius %s" % \
            (str(self.x), str(self.y), str(self.r))

    def _write(self, prefix):
        x = self.x
        y = self.y
        r = self.r

        # Build cylinder from other primitives
        c = InfiniteCylinder(x[0], x[1], x[2], y[0], y[1], y[2], r)
        p0 = Plane(x[0], x[1], x[2], x[0] - y[0], x[1] - y[1], x[2] - y[2])
        p1 = Plane(y[0], y[1], y[2], y[0] - x[0], y[1] - x[1], y[2] - x[2])

        # Add definitions for primitives
        s = c._write(prefix + "_0")  + "\n" + \
            p0._write(prefix + "_1") + "\n" + \
            p1._write(prefix + "_2")

        # Add definition for composite object
        s += "\nsolid %s = %s_0 and %s_1 and %s_2;" % \
            (prefix, prefix, prefix, prefix)

        return s

    def rotate(self, angle, axis):
        "Rotate given angle / degrees around given axis through center of mass"

        # Compute center of mass
        c = 0.5 * (numpy.array(self.x) + numpy.array(self.y))

        # Rotate points
        self.x = rotate(self.x, c, angle, axis)
        self.y = rotate(self.y, c, angle, axis)

class Sphere(Primitive):

    def __init__(self, x0, x1, x2, r):
        "A sphere at x with radius r"

        Primitive.__init__(self)

        self.x = (float(x0), float(x1), float(x2))
        self.r = r

    def __str__(self):
        return "Sphere at x = %s with radius %s" % \
            (str(self.x), str(self.r))

    def _write(self, prefix):
        x = self.x
        r = self.r
        return "solid %s = sphere(%g, %g, %g; %g);" % \
            (prefix, x[0], x[1], x[2], r)



class Polyhedron(Primitive):

    def __init__(self, points, triangles):
        """A polyhedron defined by a surface. Triangles must be
        oriented counter-clockwise when viewed from the outside."""

        Primitive.__init__(self)

        self.points = points
        self.triangles = triangles

    def __str__(self):
        return "Polyhedron with %d points and %d triangles" % \
            (len(self.points), len(self.triangles))

    def _write(self, prefix):
        points = self.points
        triangles = self.triangles
        return "solid %s = polyhedron(%s;; %s);" % \
            (prefix,
             "; ".join(", ".join(str(pii) for pii in pi) for pi in points),
             "; ".join(", ".join(str(tii + 1) for tii in ti) for ti in triangles))

    def rotate(self, angle, axis):
        "Rotate given angle / degrees around given axis through center of mass"

        # Compute center of mass
        c = sum(numpy.array(p) for p in self.points) / float(len(self.points))

        # Rotate points
        self.points = tuple(rotate(x, c, angle, axis) for x in self.points)

class Brick(Primitive):

    def __init__(self, x0, x1, x2, y0, y1, y2):
        "A brick with opposite corners x and y"

        Primitive.__init__(self)

        self.x = (float(x0), float(x1), float(x2))
        self.y = (float(y0), float(y1), float(y2))

    def __str__(self):
        return "Brick with corners x = %s and y = %s" % \
            (str(self.x), str(self.y))

    def _write(self, prefix):
        x = self.x
        y = self.y
        return "solid %s = orthobrick(%g, %g, %g; %g, %g, %g);" % \
            (prefix, x[0], x[1], x[2], y[0], y[1], y[2])

# Note: This is a function, not a class. See comment in Cylinder class.
def Box(x0, x1, x2, y0, y1, y2):
    """A box with opposite corners x and y. This differs from a
    brick in that it is constructed as a polyhedron and can be
    rotated."""

    # FIXME: Handle boxes where xi < yi is not required
    if x0 >= y0 or x1 >= y1 or x2 >= y2:
        cpp.dolfin_error("netgen.py",
                         "generating a Box",
                         "expected coordinates for box to increase for each axis.")

    # Create triangulation of box surface

    x = numpy.array((x0, x1, x2))
    y = numpy.array((y0, y1, y2))

    dx0 = numpy.array((y[0] - x[0], 0, 0))
    dx1 = numpy.array((0, y[1] - x[1], 0))
    dx2 = numpy.array((0, 0, y[2] - x[2]))

    points = (x, x + dx0, x + dx0 + dx1, x + dx1,
              x + dx2, x + dx0 + dx2, x + dx0 + dx1 + dx2, x + dx1 + dx2)

    triangles = ((0, 1, 5), (0, 5, 4),
                 (1, 2, 6), (1, 6, 5),
                 (2, 3, 7), (2, 7, 6),
                 (3, 0, 4), (3, 4, 7),
                 (3, 2, 1), (3, 1, 0),
                 (4, 5, 6), (4, 6, 7))

    return Polyhedron(points, triangles)

class OpenCone(Primitive):
    "An open cone from x to y with radii rx and ry"

    def __init__(self, x0, x1, x2, y0, y1, y2, rx, ry):

        Primitive.__init__(self)

        self.x = (float(x0), float(x1), float(x2))
        self.y = (float(y0), float(y1), float(y2))
        self.rx = float(rx)
        self.ry = float(ry)

    def __str__(self):
        return "OpenCone from x = %s to y = %s with radii rx = %s and ry = %s" % \
            (str(self.x), str(self.y), str(self.rx), str(self.ry))

    def _write(self, prefix):
        x = self.x
        y = self.y
        rx = self.rx
        ry = self.ry
        return "solid %s = cone(%g, %g, %g; %g; %g, %g, %g; %g);" % \
            (prefix, x[0], x[1], x[2], rx, y[0], y[1], y[2], ry)

# Note: This is a function, not a class. See comment in Cylinder class.
def Cone(x0, x1, x2, y0, y1, y2, rx, ry):
    "A cone from x to y with radii rx and ry"

    cone = OpenCone(x0, x1, x2, y0, y1, y2, rx, ry)
    p0 = Plane(x0, x1, x2, x0 - y0, x1 - y1, x2 - y2)
    p1 = Plane(y0, y1, y2, y0 - x0, y1 - x1, y2 - x2)

    return cone * p0 * p1

# Note: This is a function, not a class. See comment in Cylinder class.
def LEGO(n0, n1, n2=1, x0=0.0, x1=0.0, x2=0.0):
    """A standard LEGO brick starting at the point x with (n0, n1)
    knobs and height n2. The height should be 1 for a thin brick or 3
    for a regular brick."""

    # Standard dimensions for LEGO bricks / m
    P = 8.0 * 0.001
    h = 3.2 * 0.001
    D = 5.0 * 0.001
    b = 1.7 * 0.001
    d = 0.2 * 0.001

    # Create brick
    lego = Brick(x0 + 0.5*d, x1 + 0.5*d, x2,
                 x0 + n0*P - 0.5*d, x1 + n1*P - 0.5*d, x2 + n2*h)

    # Add knobs
    for i in range(n0):
        for j in range(n1):
            x = x0 + (i + 0.5)*P
            y = x1 + (j + 0.5)*P
            z = x2
            knob = Cylinder(x, y, z,
                            x, y, z + n2*h + b,
                            0.5*D)
            lego = lego + knob

    return lego
