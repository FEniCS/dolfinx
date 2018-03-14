// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/log/log.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfin_wrappers {
// common
void common(py::module &m);
void mpi(py::module &m);

// log
void log(py::module &m);

void function(py::module &m);
void fem(py::module &m);

void generation(py::module &m);
void geometry(py::module &m);
void graph(py::module &m);
void io(py::module &m);
void la(py::module &m);
void math(py::module &m);
void mesh(py::module &m);
void nls(py::module &m);
void parameter(py::module &m);
void refinement(py::module &m);
} // namespace dolfin_wrappers

PYBIND11_MODULE(cpp, m) {
  // Create module for C++ wrappers
  m.doc() = "DOLFIN Python interface";
  m.attr("__version__") = DOLFIN_VERSION;

  // Create common submodule [common]
  py::module common = m.def_submodule("common", "Common module");
  dolfin_wrappers::common(common);

  // Create MPI class [common]
  dolfin_wrappers::mpi(m);

  // Create common submodule [log]
  py::module log = m.def_submodule("log", "Logging module");
  dolfin_wrappers::log(log);

  // Create function submodule [function]
  py::module function = m.def_submodule("function", "Function module");
  dolfin_wrappers::function(function);

  // Create math submodule [math]
  py::module math = m.def_submodule("math", "Math library module");
  dolfin_wrappers::math(math);

  // Create mesh submodule [mesh]
  py::module mesh = m.def_submodule("mesh", "Mesh library module");
  dolfin_wrappers::mesh(mesh);

  // Create graph submodule [graph]
  py::module graph = m.def_submodule("graph", "Graph module");
  dolfin_wrappers::graph(graph);

  // Create fem submodule [fem]
  py::module fem = m.def_submodule("fem", "FEM module");
  dolfin_wrappers::fem(fem);

  // Create generation submodule [generation]
  py::module generation =
      m.def_submodule("generation", "Mesh generation module");
  dolfin_wrappers::generation(generation);

  // Create geometry submodule
  py::module geometry = m.def_submodule("geometry", "Geometry module");
  dolfin_wrappers::geometry(geometry);

  // Create io submodule
  py::module io = m.def_submodule("io", "I/O module");
  dolfin_wrappers::io(io);

  // Create la submodule
  py::module la = m.def_submodule("la", "Linear algebra module");
  dolfin_wrappers::la(la);

  // Create nls submodule
  py::module nls = m.def_submodule("nls", "Nonlinear solver module");
  dolfin_wrappers::nls(nls);

  // Create parameter submodule
  py::module parameter = m.def_submodule("parameter", "Parameter module");
  dolfin_wrappers::parameter(parameter);

  // Create refinement submodule
  py::module refinement = m.def_submodule("refinement", "Refinement module");
  dolfin_wrappers::refinement(refinement);

  // FIXME: these are just for the transition
  m.def("warning", [](std::string message) { dolfin::log::warning(message); });
}
