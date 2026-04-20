# Copyright (C) 2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest


@pytest.mark.parametrize(
    "marker_mode",
    [
        pytest.param(0, marks=pytest.mark.xfail(raises=RuntimeError)),
        pytest.param(1, marks=pytest.mark.xfail(raises=RuntimeError)),
        2,
        pytest.param(3, marks=pytest.mark.xfail(raises=RuntimeError)),
    ],
)
def test_physical_tags(marker_mode):
    """Test that we catch partially tagged meshes and not tagged
    meshes as errors.
    """
    gmsh = pytest.importorskip("gmsh")

    from dolfinx.io import gmsh as gmshio

    gmsh.initialize()

    def gmsh_tet_model(order):
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model()
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            model.add("Sphere minus box")
            model.setCurrent("Sphere minus box")
            model.occ.addSphere(0, 0, 0, 1)
            model.occ.addSphere(2, 2, 2, 0.3)
            model.occ.synchronize()
            volume_entities = [model[1] for model in model.getEntities(3)]
            volume_entities = volume_entities[:marker_mode]
            for i, entity in enumerate(volume_entities):
                model.addPhysicalGroup(3, [entity], tag=i)
            if marker_mode == 3:  # Check duplicate marker error
                model.addPhysicalGroup(3, [entity], tag=10)
            model.mesh.generate(3)
            gmsh.option.setNumber("General.Terminal", 1)
            model.mesh.setOrder(order)
            gmsh.option.setNumber("General.Terminal", 0)

        mesh_data = gmshio.model_to_mesh(model, comm, 0)
        return mesh_data.mesh, mesh_data.cell_tags

    msh, cell_tags = gmsh_tet_model(1)
    gdim = msh.geometry.dim
    assert msh.geometry.cmap.degree == 1
    assert msh.geometry.dim == gdim
    local_values = np.unique(cell_tags.values)
    all_values = np.unique(np.hstack(msh.comm.allgather(local_values)))
    assert len(all_values) == 2

    gmsh.finalize()
