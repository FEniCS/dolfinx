# Copyright (C) 2016 Jan Blechta, Martin Alnæs
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
import ufl
import os

exit(0)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
except ImportError:
    print("This demo requires matplotlib! Bye.")
    exit()


rank = MPI.rank(MPI.comm_world)
size = MPI.size(MPI.comm_world)
suffix = "_r%s" % rank if size > 1 else ""


def plot_alongside(*args, **kwargs):
    """Plot supplied functions in single figure with common colorbar.
    It is users responsibility to supply 'range_min' and 'range_max' in kwargs.
    """
    n = len(args)
    plt.figure(figsize=(4*n+2, 4))
    projection = "3d" if kwargs.get("mode") == "warp" else None

    for i in range(n):
        plt.subplot(1, n, i+1, projection=projection)
        p = plot(args[i], **kwargs)

    plt.tight_layout()

    # Create colorbar
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(p, cax=cbar_ax)


def create_interval_mesh(vertices):
    "Given list of vertex coordinate tuples, build and return a mesh of intervals."
    gdim = len(vertices[0])
    mesh = Mesh()
    me = MeshEditor()
    me.open(mesh, "interval", 1, gdim)

    # Add vertices to mesh
    nv = len(vertices)
    me.init_vertices(nv)
    for i, v in enumerate(vertices):
        me.add_vertex(i, *v)

    # Add cells to mesh
    me.init_cells(nv-1)
    for i in range(nv-1):
        c = (i, i+1)
        me.add_cell(i, *c)

    me.close()
    assert mesh.ordered()
    return mesh


def interval_mesh(gdim, n):
    us = [i/float(n-1) for i in range(n)]
    vertices = [(cos(4.0*DOLFIN_PI*u), sin(4.0*DOLFIN_PI*u), 2.0*u)[-gdim:] for u in us]
    return create_interval_mesh(vertices)


def plot_1d_meshes():
    # FIXME: This passes fine in parallel although it's not obvious what does it do
    plt.figure()
    plot(interval_mesh(1, 30))
    plt.savefig("mesh_1d%s.png" % suffix)

    plt.figure()
    plot(interval_mesh(2, 100))
    plt.savefig("mesh_2d%s.png" % suffix)

    plt.figure()
    plot(interval_mesh(3, 100))
    plt.savefig("mesh_3d%s.png" % suffix)


def plot_functions():
    mesh = UnitSquareMesh(8, 8, 'crossed')
    P1 = FunctionSpace(mesh, "Lagrange", 1)
    u = interpolate(Expression("x[0]*x[0] + x[1]*x[1]*(x[1]-0.5)", degree=3), P1)
    v = interpolate(Expression("1.0+x[0]*x[0]", degree=2), P1)

    # Get common range
    r = {"range_min": min(u.vector().min(), v.vector().min()),
         "range_max": max(u.vector().max(), v.vector().max())}

    plot_alongside(u, v, **r)
    plot_alongside(u, v, mode='color', **r)
    plt.savefig("color_plot%s.pdf" % suffix)
    plot_alongside(u, v, mode='warp', **r)
    plt.savefig("warp_plot%s.pdf" % suffix)
    plot_alongside(u, v, v, mode='warp', **r)


def main(argv=None):
    plot_1d_meshes()
    plot_functions()
    if os.environ.get("DOLFIN_NOPLOT", "0") == "0":
        plt.show()


if __name__ == '__main__':
    import sys
    main(sys.argv)
