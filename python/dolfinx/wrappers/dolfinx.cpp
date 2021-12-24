// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void common(py::module& m);
void mpi(py::module& m);

void log(py::module& m);
void fem(py::module& m);
void geometry(py::module& m);
void graph(py::module& m);
void io(py::module& m);
void la(py::module& m);
void mesh(py::module& m);
void nls(py::module& m);
void refinement(py::module& m);
} // namespace dolfinx_wrappers

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINx Python interface";
  m.attr("__version__") = DOLFINX_VERSION;

  // Create common submodule [common]
  py::module common = m.def_submodule("common", "Common module");
  dolfinx_wrappers::common(common);

  // Create common submodule [log]
  py::module log = m.def_submodule("log", "Logging module");
  dolfinx_wrappers::log(log);

  // Create mesh submodule [mesh]
  py::module mesh = m.def_submodule("mesh", "Mesh library module");
  dolfinx_wrappers::mesh(mesh);

  // Create graph submodule [graph]
  py::module graph = m.def_submodule("graph", "Graph module");
  dolfinx_wrappers::graph(graph);

  // Create fem submodule [fem]
  py::module fem = m.def_submodule("fem", "FEM module");
  dolfinx_wrappers::fem(fem);

  // Create geometry submodule
  py::module geometry = m.def_submodule("geometry", "Geometry module");
  dolfinx_wrappers::geometry(geometry);

  // Create io submodule
  py::module io = m.def_submodule("io", "I/O module");
  dolfinx_wrappers::io(io);

  // Create la submodule
  py::module la = m.def_submodule("la", "Linear algebra module");
  dolfinx_wrappers::la(la);

  // Create nls submodule
  py::module nls = m.def_submodule("nls", "Nonlinear solver module");
  dolfinx_wrappers::nls(nls);

  // Create refinement submodule
  py::module refinement = m.def_submodule("refinement", "Refinement module");
  dolfinx_wrappers::refinement(refinement);
}
