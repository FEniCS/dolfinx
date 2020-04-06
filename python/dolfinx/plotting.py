# Copyright (C) 2008-2012 Joachim B. Haga and Fredrik Valdmanis
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import warnings

import numpy as np

from dolfinx import cpp, fem, function

__all__ = ["plot"]

_matplotlib_plottable_types = (cpp.function.Function,
                               cpp.mesh.Mesh,
                               cpp.fem.DirichletBC)
_all_plottable_types = tuple(set.union(set(_matplotlib_plottable_types)))


def _has_matplotlib():
    try:
        import matplotlib  # noqa
    except ImportError:
        return False
    return True


def mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.geometry.x
    cells = mesh.geometry.dofmap().array().reshape((-1, mesh.topology.dim + 1))
    return tri.Triangulation(xy[:, 0], xy[:, 1], cells)


def mplot_mesh(ax, mesh, **kwargs):
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    if gdim == 2 and tdim == 2:
        color = kwargs.pop("color", '#808080')
        return ax.triplot(mesh2triang(mesh), color=color, **kwargs)
    elif gdim == 3 and tdim == 3:
        bmesh = cpp.mesh.BoundaryMesh(mesh, "exterior", order=False)
        mplot_mesh(ax, bmesh, **kwargs)
    elif gdim == 3 and tdim == 2:
        xy = mesh.geometry.x
        cells = mesh.geometry.dofmap().array().reshape((-1, mesh.topology.dim + 1))
        return ax.plot_trisurf(
            *[xy[:, i] for i in range(gdim)], triangles=cells, **kwargs)
    elif tdim == 1:
        x = [mesh.geometry.x[:, i] for i in range(gdim)]
        if gdim == 1:
            x.append(np.zeros_like(x[0]))
            ax.set_yticks([])
        marker = kwargs.pop('marker', 'o')
        return ax.plot(*x, marker=marker, **kwargs)
    else:
        assert False, "this code should not be reached"


# TODO: This is duplicated somewhere else
def create_cg1_function_space(mesh, sh):
    r = len(sh)
    if r == 0:
        V = function.FunctionSpace(mesh, ("CG", 1))
    elif r == 1:
        V = function.VectorFunctionSpace(mesh, ("CG", 1), dim=sh[0])
    else:
        V = function.TensorFunctionSpace(mesh, ("CG", 1), shape=sh)
    return V


def mplot_expression(ax, f, mesh, **kwargs):
    # TODO: Can probably avoid creating the function space here by
    # restructuring mplot_function a bit so it can handle Expression
    # natively
    V = create_cg1_function_space(mesh, f.value_shape)
    g = fem.interpolate(f, V)
    return mplot_function(ax, g, **kwargs)


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

    if fvec.getSize() == mesh.num_entities(tdim):
        # DG0 cellwise function
        C = fvec.get_local()
        if (C.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            C = np.real(C)
        # NB! Assuming here dof ordering matching cell numbering
        if gdim == 2 and tdim == 2:
            return ax.tripcolor(mesh2triang(mesh), C, **kwargs)
        elif gdim == 3 and tdim == 2:  # surface in 3d
            # FIXME: Not tested, probably broken
            xy = mesh.geometry.x
            shade = kwargs.pop("shade", True)
            return ax.plot_trisurf(
                mesh2triang(mesh), xy[:, 2], C, shade=shade, **kwargs)
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
            raise AttributeError(
                'Matplotlib plotting backend only supports 2D mesh for scalar functions.'
            )

    elif f.value_rank == 0:
        # Scalar function, interpolated to vertices
        # TODO: Handle DG1?
        C = f.compute_point_values()
        if (C.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            C = np.real(C)

        if gdim == 2 and tdim == 2:
            mode = kwargs.pop("mode", "contourf")
            if mode == "contourf":
                levels = kwargs.pop("levels", 40)
                return ax.tricontourf(
                    mesh2triang(mesh), C[:, 0], levels, **kwargs)
            elif mode == "color":
                shading = kwargs.pop("shading", "gouraud")
                return ax.tripcolor(
                    mesh2triang(mesh), C[:, 0], shading=shading, **kwargs)
            elif mode == "warp":
                from matplotlib import cm
                cmap = kwargs.pop("cmap", cm.jet)
                linewidths = kwargs.pop("linewidths", 0)
                return ax.plot_trisurf(
                    mesh2triang(mesh),
                    C[:, 0],
                    cmap=cmap,
                    linewidths=linewidths,
                    **kwargs)
            elif mode == "wireframe":
                return ax.triplot(mesh2triang(mesh), **kwargs)
            elif mode == "contour":
                return ax.tricontour(mesh2triang(mesh), C[:, 0], **kwargs)
        elif gdim == 3 and tdim == 2:  # surface in 3d
            # FIXME: Not tested
            from matplotlib import cm
            cmap = kwargs.pop("cmap", cm.jet)
            return ax.plot_trisurf(
                mesh2triang(mesh), C[:, 0], cmap=cmap, **kwargs)
        elif gdim == 3 and tdim == 3:
            # Volume
            # TODO: Isosurfaces?
            # Vertex point cloud
            X = [mesh.geometrycoordinates[:, i] for i in range(gdim)]
            return ax.scatter(*X, c=C, **kwargs)
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
            raise AttributeError(
                'Matplotlib plotting backend only supports 2D mesh for scalar functions.'
            )

    elif f.value_rank == 1:
        # Vector function, interpolated to vertices
        w0 = f.compute_point_values()
        if (w0.dtype.type is np.complex128):
            warnings.warn("Plotting real part of complex data")
            w0 = np.real(w0)
        nv = mesh.num_entities(0)
        if w0.shape[1] != gdim:
            raise AttributeError(
                'Vector length must match geometric dimension.')
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
            if gdim == 3:
                length = kwargs.pop("length", 0.1)
                return ax.quiver(*args, length=length, **kwargs)
            else:
                return ax.quiver(*args, **kwargs)
        elif mode == "displacement":
            Xdef = [X[i] + U[i] for i in range(gdim)]
            import matplotlib.tri as tri
            if gdim == 2 and tdim == 2:
                # FIXME: Not tested
                cells = mesh.geometry.dofmap().array().reshape((-1, mesh.topology.dim + 1))
                triang = tri.Triangulation(Xdef[0], Xdef[1], cells)
                shading = kwargs.pop("shading", "flat")
                return ax.tripcolor(triang, C, shading=shading, **kwargs)
            else:
                # Return gracefully to make regression test pass without vtk
                warnings.warn(
                    'Plotting does not support displacement for {} in {}}. Continuing without plot.'.
                    format(tdim, gdim))
                return


def mplot_dirichletbc(ax, obj, **kwargs):
    raise AttributeError(
        "Matplotlib plotting backend doesn't handle DirichletBC.")


def _plot_matplotlib(obj, mesh, kwargs):
    if not isinstance(obj, _matplotlib_plottable_types):
        print("Don't know how to plot type %s." % type(obj))
        return

    # Plotting is not working with all ufl cells
    if mesh.ufl_cell().cellname() not in [
            'interval', 'triangle', 'tetrahedron'
    ]:
        raise AttributeError(
            ("Matplotlib plotting backend doesn't handle %s mesh.\n"
             "Possible options are saving the output to XDMF file.") %
            mesh.ufl_cell().cellname())

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

    # Drop unsupported kwargs and inform user
    _unsupported_kwargs = ["rescale", "wireframe"]
    for kw in _unsupported_kwargs:
        if kwargs.pop(kw, None):
            cpp.warning("Matplotlib backend does not support '%s' kwarg yet. "
                        "Ignoring it..." % kw)

    if isinstance(obj, cpp.function.Function):
        return mplot_function(ax, obj, **kwargs)
    # elif isinstance(obj, cpp.function.Expression):
    #     return mplot_expression(ax, obj, mesh, **kwargs)
    elif isinstance(obj, cpp.mesh.Mesh):
        return mplot_mesh(ax, obj, **kwargs)
    elif isinstance(obj, cpp.fem.DirichletBC):
        return mplot_dirichletbc(ax, obj, **kwargs)
    else:
        raise AttributeError('Failed to plot %s' % type(obj))


def plot(object, *args, **kwargs):
    """
    Plot given object.

    *Arguments*
        object
            a :py:class:`Mesh <dolfinx.cpp.Mesh>`, a :py:class:`Function
            <dolfinx.functions.function.Function>`, a :py:class:`Expression`
            <dolfinx.Expression>, a :py:class:`DirichletBC`
            <dolfinx.cpp.DirichletBC>, a :py:class:`FiniteElement
            <ufl.FiniteElement>`.

    *Examples of usage*
        In the simplest case, to plot only e.g. a mesh, simply use

        .. code-block:: python

            mesh = UnitSquare(4, 4)
            plot(mesh)

        Use the ``title`` argument to specify title of the plot

        .. code-block:: python

            plot(mesh, tite="Finite element mesh")

        It is also possible to plot an element

        .. code-block:: python

            element = FiniteElement("BDM", tetrahedron, 3)
            plot(element)

        Vector valued functions can be visualized with an alternative mode

        .. code-block:: python

            plot(u, mode = "glyphs")

        A more advanced example

        .. code-block:: python

            plot(u,
                 wireframe = True,              # use wireframe rendering
                 interactive = False,           # do not hold plot on screen
                 scalarbar = False,             # hide the color mapping bar
                 hardcopy_prefix = "myplot",    # default plotfile name
                 scale = 2.0,                   # scale the warping/glyphs
                 title = "Fancy plot",          # set your own title
                 )

    """

    # Return if plotting is disabled
    if os.environ.get("DOLFIN_NOPLOT", "0") != "0":
        return

    # Return if Matplotlib is not available
    if not _has_matplotlib():
        cpp.log.info("Matplotlib is required to plot from Python.")
        return

    # For dolfinx.function.Function, extract cpp_object
    if hasattr(object, "_cpp_object"):
        object = object._cpp_object

    # Get mesh from explicit mesh kwarg, only positional arg, or via
    # object
    mesh = kwargs.pop('mesh', None)
    if isinstance(object, cpp.mesh.Mesh):
        if mesh is not None and mesh.id() != object.id():
            raise RuntimeError(
                "Got different mesh in plot object and keyword argument")
        mesh = object

    if mesh is None:
        if isinstance(object, cpp.function.Function):
            mesh = object.function_space.mesh
        elif hasattr(object, "mesh"):
            mesh = object.mesh

    # Expressions do not carry their own mesh
    # if isinstance(object, cpp.function.Expression) and mesh is None:
    #     raise RuntimeError("Expecting a mesh as keyword argument")

    backend = kwargs.pop("backend", "matplotlib")
    if backend not in ("matplotlib"):
        raise RuntimeError("Plotting backend %s not recognised" % backend)

    # Try to project if object is not a standard plottable type
    if not isinstance(object, _all_plottable_types):
        raise RuntimeError("Cannot plot object.")

    # Plot
    if backend == "matplotlib":
        return _plot_matplotlib(object, mesh, kwargs)
    else:
        assert False, "This code should not be reached."
