"""Visualization of the mesh and subdomains with
[PyVista](https://docs.pyvista.org/)"""

from pathlib import Path

import numpy as np

from dolfinx import fem, plot


def load_pyvista():
    try:
        import pyvista

        return pyvista
    except ModuleNotFoundError:
        print("pyvista and pyvistaqt are required to visualise the solution")
        return None


def load_vtx_writer():
    try:
        from dolfinx.io import VTXWriter

        return VTXWriter
    except ImportError:
        print("This demo requires DOLFINx to be configured with adios2.")
        raise SystemExit(0)


def make_output_folder(path="output_pml"):
    out_folder = Path(path)
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder


def visualize_mesh(mesh_data, out_folder, pyvista):
    if pyvista is None:
        return

    tdim = mesh_data.mesh.topology.dim
    topology, cell_types, geometry = plot.vtk_mesh(mesh_data.mesh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()

    num_local_cells = mesh_data.mesh.topology.index_map(tdim).size_local
    grid.cell_data["Marker"] = mesh_data.cell_tags.values[
        mesh_data.cell_tags.indices < num_local_cells
    ]
    grid.set_active_scalars("Marker")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()

    if not pyvista.OFF_SCREEN:
        plotter.show(interactive=True)
    else:
        plotter.screenshot(out_folder / "wire_mesh_pml.png", window_size=[800, 800])


def interpolate_to_dg(V, field, degree: int):
    gdim = V.mesh.geometry.dim
    V_dg = fem.functionspace(V.mesh, ("DG", degree, (gdim,)))
    field_dg = fem.Function(V_dg)
    field_dg.interpolate(field)
    return V_dg, field_dg


def write_bp(mesh_obj, field_dg, filename):
    VTXWriter = load_vtx_writer()
    with VTXWriter(mesh_obj.comm, filename, field_dg) as vtx:
        vtx.write(0.0)


def visualize_vector_field(V_dg, field_dg, out_folder, image_name, pyvista):
    if pyvista is None:
        return

    tdim = V_dg.mesh.topology.dim
    V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
    V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)

    field_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
    field_values[:, :tdim] = field_dg.x.array.reshape(V_x.shape[0], tdim).real
    V_grid.point_data["u"] = field_values

    plotter = pyvista.Plotter()
    plotter.add_text("magnitude", font_size=12, color="black")
    plotter.add_mesh(V_grid.copy(), show_edges=True)
    plotter.view_xy()
    plotter.link_views()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(out_folder / image_name, window_size=[800, 800])


def save_field(mesh_obj, V, field, degree, out_folder, filename, pyvista=None, image_name=None):
    V_dg, field_dg = interpolate_to_dg(V, field, degree)
    write_bp(mesh_obj, field_dg, out_folder / filename)

    if image_name is not None:
        visualize_vector_field(V_dg, field_dg, out_folder, image_name, pyvista)

    return V_dg, field_dg
