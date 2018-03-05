// Copyright (C) 2018 Chris N. Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/mesh/Mesh.h>
#include <dolfin/refinement/refine.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers {

void refinement(py::module &m) {
  // dolfin::refinement::refine
  m.def("refine", [](dolfin::mesh::Mesh &mesh, bool redistribute) {
    auto new_mesh = std::make_unique<dolfin::mesh::Mesh>(mesh.mpi_comm());
    dolfin::refinement::refine(*new_mesh, mesh, redistribute);
    return new_mesh;
  });
}

} // namespace dolfin_wrappers
