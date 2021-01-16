# Copyright (C) 2008-2021 Joachim B. Haga, Fredrik Valdmanis, Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple matplotlib plotting functions"""

import warnings

import numpy as np
import typing
import ufl

from dolfinx import cpp, fem
from dolfinx.mesh import create_mesh

__all__ = ["plot"]


def create_boundary_mesh(mesh, comm, orient=False):
    """
    Create a mesh consisting of all exterior facets of a mesh
    Input:
      mesh   - The mesh
      comm   - The MPI communicator
      orient - Boolean flag for reorientation of facets to have
               consistent outwards-pointing normal (default: True)
    Output:
      bmesh - The boundary mesh
      bmesh_to_geometry - Map from cells of the boundary mesh
                          to the geometry of the original mesh
    """
    ext_facets = cpp.mesh.exterior_facet_indices(mesh)
    boundary_geometry = cpp.mesh.entities_to_geometry(
        mesh, mesh.topology.dim - 1, ext_facets, orient)
    facet_type = cpp.mesh.to_string(cpp.mesh.cell_entity_type(
        mesh.topology.cell_type, mesh.topology.dim - 1))
    facet_cell = ufl.Cell(facet_type,
                          geometric_dimension=mesh.geometry.dim)
    degree = mesh.ufl_domain().ufl_coordinate_element().degree()
    ufl_domain = ufl.Mesh(ufl.VectorElement("Lagrange", facet_cell, degree))
    bmesh = create_mesh(comm, boundary_geometry, mesh.geometry.x, ufl_domain)
    return bmesh, boundary_geometry


def _has_matplotlib():
    try:
        import matplotlib  # noqa
    except ImportError:
        return False
    return True


def mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    return tri.Triangulation(xy[:, 0], xy[:, 1], cells)


def mesh2quad(mesh):
    x = mesh.geometry.x[:, :mesh.geometry.dim]
    num_vertices = cpp.mesh.cell_num_vertices(mesh.topology.cell_type)
    vtk_perm = cpp.io.perm_vtk(mesh.topology.cell_type, num_vertices)
    perm = np.zeros(len(vtk_perm), dtype=np.int32)
    for i in range(len(vtk_perm)):
        perm[vtk_perm[i]] = i
    cells = mesh.geometry.dofmap.array.reshape((-1, num_vertices))
    return x, cells[:, perm]


def mplot_mesh(ax, mesh, **kwargs):
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    if gdim == 2 and tdim == 2:
        if mesh.topology.cell_type == cpp.mesh.CellType.triangle:
            color = kwargs.pop("color", '#808080')
            return ax.triplot(mesh2triang(mesh), color=color, **kwargs)
        else:
            from matplotlib.collections import PolyCollection  # noqa
            x, cells = mesh2quad(mesh)
            p = PolyCollection(x[cells], facecolor=(0, 0, 0, 0), edgecolor="k")
            ax.add_collection(p)
            return p
    elif gdim == 3 and tdim == 3:
        # Only plot outer surface of 3D mesh
        bmesh = create_boundary_mesh(mesh, mesh.mpi_comm(), orient=False)
        mplot_mesh(ax, bmesh[0], **kwargs)
    elif gdim == 3 and tdim == 2:
        if mesh.topology.cell_type == cpp.mesh.CellType.triangle:
            xy = mesh.geometry.x
            cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
            return ax.plot_trisurf(*[xy[:, i] for i in range(gdim)], triangles=cells, **kwargs)
        else:
            raise NotImplementedError("Plotting quadrilateral mesh with geometric dimension 3 is not implemented.")

    elif tdim == 1:
        x = [mesh.geometry.x[:, i] for i in range(gdim)]
        if gdim == 1:
            x.append(np.zeros_like(x[0]))
            ax.set_yticks([])
        marker = kwargs.pop('marker', 'o')
        return ax.plot(*x, marker=marker, **kwargs)
    else:
        raise RuntimeError("Mesh with topological dimension {0:d}".format(tdim)
                           + " and geometrical dimension {0:d} cannot be plotted.".format(gdim))


def mplot_function(ax, f, **kwargs):
    mesh = f.function_space.mesh
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim

    # Extract the function vector in a way that also works for
    # subfunctions
    try:
        fvec = f.vector
    except RuntimeError:
        fspace = f.function_space
        try:
            fspace = fspace.collapse()
        except RuntimeError:
            return
        fvec = fem.interpolate(f, fspace).vector

    map_c = mesh.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    if fvec.getSize() == num_cells:
        # DG0 cellwise function
        C = fvec.array
        if (C.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            C = np.real(C)
        # NB! Assuming here dof ordering matching cell numbering
        if gdim == 2 and tdim == 2:
            if mesh.topology.cell_type == cpp.mesh.CellType.triangle:
                return ax.tripcolor(mesh2triang(mesh), C, **kwargs)
            else:
                raise NotImplementedError("Plotting of cellwise constant functions"
                                          + " are not implemented for quadrilaterals.")
        elif gdim == 3 and tdim == 2:  # surface in 3d
            # FIXME: Not tested, probably broken
            xy = mesh.geometry.x
            shade = kwargs.pop("shade", True)
            return ax.plot_trisurf(mesh2triang(mesh), xy[:, 2], C, shade=shade, **kwargs)
        elif gdim == 1 and tdim == 1:
            x = mesh.geometry.x[:, 0]
            nv = len(x)
            # Insert duplicate points to get piecewise constant plot
            xp = np.zeros(2 * nv - 2)
            xp[0] = x[0]
            xp[-1] = x[-1]
            xp[1:2 * nv - 3:2] = x[1:-1]
            xp[2:2 * nv - 2:2] = x[1:-1]
            Cp = np.zeros(len(xp))
            Cp[0:len(Cp) - 1:2] = C
            Cp[1:len(Cp):2] = C
            return ax.plot(xp, Cp, *kwargs)
        # elif tdim == 1:  # FIXME: Plot embedded line
        else:
            raise AttributeError("Plotting of DG0 function with geometric dimension {0: d} ".format(gdim)
                                 + "and topological dimension {0:d} not supported.".format(tdim))

    elif f.function_space.element.value_rank == 0:
        # Scalar function, interpolated to vertices
        # TODO: Handle DG1?
        C = f.compute_point_values()
        if (C.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            C = np.real(C)

        if gdim == 2 and tdim == 2:
            if mesh.topology.cell_type == cpp.mesh.CellType.quadrilateral:
                warnings.warn(
                    "Functions on quadrilateral meshes can only be visualized as a point cloud.\n"
                    + "Please use XDMF for more advanced visualization.")
                x = mesh.geometry.x
                markersize = kwargs.pop("markersize", 25)
                return ax.scatter(x[:, 0], x[:, 1], s=markersize, c=C[:, 0], **kwargs)
            mode = kwargs.pop("mode", "contourf")
            if mode == "contourf":
                levels = kwargs.pop("levels", 40)
                return ax.tricontourf(mesh2triang(mesh), C[:, 0], levels, **kwargs)
            elif mode == "color":
                shading = kwargs.pop("shading", "gouraud")
                return ax.tripcolor(
                    mesh2triang(mesh), C[:, 0], shading=shading, **kwargs)
            elif mode == "warp":
                from matplotlib import cm  # noqa
                cmap = kwargs.pop("cmap", cm.viridis)
                linewidths = kwargs.pop("linewidths", 0)
                return ax.plot_trisurf(mesh2triang(mesh), C[:, 0], cmap=cmap,
                                       linewidths=linewidths, **kwargs)
            elif mode == "wireframe":
                return ax.triplot(mesh2triang(mesh), **kwargs)
            elif mode == "contour":
                return ax.tricontour(mesh2triang(mesh), C[:, 0], **kwargs)
            else:
                raise AttributeError("Invalid plotting mode {0:s} for function".format(mode))
        elif gdim == 3 and tdim == 2:  # surface in 3d
            # FIXME: Not tested
            from matplotlib import cm  # noqa
            cmap = kwargs.pop("cmap", cm.viridis)
            if mesh.topology.cell_type == cpp.mesh.CellType.triangle:
                return ax.plot_trisurf(mesh2triang(mesh), C[:, 0], cmap=cmap, **kwargs)
            else:
                x = mesh.geometry.x
                return ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=C[:, 0])
        elif gdim == 3 and tdim == 3:
            # Volume
            # TODO: Isosurfaces?
            # Vertex point cloud
            X = [mesh.geometry.x[:, i] for i in range(gdim)]
            kwargs.pop("mode", None)
            warnings.warn("Plotting mode ignored for 3D meshes, plotting point cloud.")
            return ax.scatter(*X, c=C.reshape(-1), **kwargs)
        elif gdim == 1 and tdim == 1:
            x = mesh.geometry.x[:, 0]
            ax.set_aspect('auto')
            p = ax.plot(x, C[:, 0], **kwargs)

            # Setting limits for Line2D objects
            # Must be done after generating plot to avoid ignoring function
            # range if no vmin/vmax are supplied
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            ax.set_ylim([vmin, vmax])
            return p
        # elif tdim == 1: # FIXME: Plot embedded line
        else:
            raise AttributeError("Plotting of function with geometric dimension {0: d} ".format(gdim)
                                 + "and topological dimension {0:d} not supported.".format(tdim))

    elif f.function_space.element.value_rank == 1:
        # Vector function, interpolated to vertices
        w0 = f.compute_point_values()
        if (w0.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            w0 = np.real(w0)
        map_v = mesh.topology.index_map(0)
        nv = map_v.size_local + map_v.num_ghosts
        if w0.shape[1] != gdim:
            raise AttributeError('Vector length must match geometric dimension.')
        X = mesh.geometry.x
        X = [X[:, i] for i in range(gdim)]
        U = [x for x in w0.T]

        # Compute magnitude
        C = U[0]**2
        for i in range(1, gdim):
            C += U[i]**2
        C = np.sqrt(C)

        mode = kwargs.pop("mode", "glyphs")
        if mode == "glyphs":
            args = X + U + [C]
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            if not (vmin is None and vmax is None):
                kwargs["clim"] = (vmin, vmax)
            if gdim == 3:
                return ax.quiver(*args, **kwargs)
            else:
                return ax.quiver(*args, **kwargs)
        elif mode == "displacement":
            scale = kwargs.pop("scale", 0.1)
            Xdef = [X[i] + scale * U[i] for i in range(gdim)]
            import matplotlib.tri as tri
            if gdim == 2 and tdim == 2:
                # FIXME: Not tested
                if mesh.topology.cell_type == cpp.mesh.CellType.quadrilateral:
                    raise NotImplementedError("Displacement plot for quadrilaterals is not implemented.")
                cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
                triang = tri.Triangulation(Xdef[0], Xdef[1], cells)
                shading = kwargs.pop("shading", "flat")
                return ax.tripcolor(triang, C, shading=shading, **kwargs)
            else:
                raise AttributeError(
                    "Displacement plotting only supported for meshes with topological and geometrical dimension 2.")
        else:
            raise AttributeError("Unsupported plotting mode {0:s}.".format(mode))


def _plot_matplotlib(obj, mesh, kwargs):
    # Plotting is not working with all ufl cells
    if mesh.ufl_cell().cellname() not in ['interval', 'triangle', "quadrilateral", 'tetrahedron']:
        raise AttributeError("Matplotlib plotting backend doesn't handle {0:s} mesh.\n"
                             "Possible options are saving the output to XDMF file.".format(
                                 mesh.ufl_cell().cellname()))

    # Avoid importing pyplot until used
    try:
        import matplotlib.pyplot as plt
    except Exception:
        cpp.warning("matplotlib.pyplot not available, cannot plot.")
        return

    gdim = mesh.geometry.dim
    if gdim == 3 or kwargs.get("mode") in ("warp", ):
        # Importing this toolkit has side effects enabling 3d support
        from mpl_toolkits.mplot3d import axes3d  # noqa

        # Enabling the 3d toolbox requires some additional arguments
        ax = plt.gca(projection='3d')
    else:
        ax = plt.gca()
        ax.set_aspect('equal')

    title = kwargs.pop("title", None)
    if title is not None:
        ax.set_title(title)

    # Translate range_min/max kwargs supported by VTKPlotter
    vmin = kwargs.pop("range_min", None)
    vmax = kwargs.pop("range_max", None)
    if vmin and "vmin" not in kwargs:
        kwargs["vmin"] = vmin
    if vmax and "vmax" not in kwargs:
        kwargs["vmax"] = vmax

    if isinstance(obj, cpp.fem.Function):
        return mplot_function(ax, obj, **kwargs)
    elif isinstance(obj, cpp.mesh.Mesh):
        return mplot_mesh(ax, obj, **kwargs)


def plot(object: typing.Union[cpp.fem.Function, cpp.mesh.Mesh], *args, **kwargs):
    """
    Plot a Function or mesh using matplotlib.
    For plotting of 3D meshes and functions, it is adviced to save the output to XDMF file."

    *Arguments*
        object
            a :py:class:`Mesh <dolfinx.cpp.mesh.Mesh>`, a :py:class:`Function
            <dolfinx.fem.function.Function>`,

    *Examples of usage*
        In the simplest case, to plot only e.g. a mesh, simply use

        .. code-block:: python

            mesh = UnitSquare(MPI.COMM_WORLD, 4, 4)
            plot(mesh)

        Use the ``title`` argument to specify title of the plot

        .. code-block:: python

            u = dolfinx.Function(V)
            plot(u, title="Finite element function")

        For a scalar valued function the function can alternatively
        be warped by scalar to produce a 3D plot.

        ... code-block:: python

            plot(u, mode="warp")

        For a vector valued function the default mode is to visualize
        the values using glyphs.

        Alternatively
        visualized on the mesh displaced by the function,
        using the displacement mode

        .. code-block:: python

            plot(u, mode = "displacement")

        A more advanced example

        .. code-block:: python

            plot(u,
                 scale = 1.5,                   # Scale the warping/glyphs
                 mode="displacement",           # Set mode to displacement
                 title = "Fancy plot",          # Set your own title
                 vmin=0,                        # Set minimum range for colorbar
                 vmax=1                         # Set maximum range for colorbar
                 )

    """

    # Return if Matplotlib is not available
    if not _has_matplotlib():
        raise ImportError("Matplotlib is required to plot from Python.")

    # For dolfinx.fem.Function, extract cpp_object
    if hasattr(object, "_cpp_object"):
        object = object._cpp_object

    mesh = None
    if isinstance(object, cpp.mesh.Mesh):
        mesh = object
    elif isinstance(object, cpp.fem.Function):
        mesh = object.function_space.mesh
    else:
        raise ValueError("Don't know how to plot type {0:s}.".format(type(object)))

    return _plot_matplotlib(object, mesh, kwargs)
