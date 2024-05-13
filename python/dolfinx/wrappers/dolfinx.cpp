// Copyright (C) 2017-2023 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void common(nb::module_& m);
void mpi(nb::module_& m);

void petsc(nb::module_& m_fem, nb::module_& m_la, nb::module_& m_nls);

void log(nb::module_& m);
void assemble(nb::module_& m);
void fem(nb::module_& m);
void geometry(nb::module_& m);
void graph(nb::module_& m);
void la(nb::module_& m);
void mesh(nb::module_& m);
void nls(nb::module_& m);
void refinement(nb::module_& m);
} // namespace dolfinx_wrappers

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINx Python interface";
  m.attr("__version__") = DOLFINX_VERSION;

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // Create common submodule [common]
  nb::module_ common = m.def_submodule("common", "Common module");
  dolfinx_wrappers::common(common);

  // Create common submodule [log]
  nb::module_ log = m.def_submodule("log", "Logging module");
  dolfinx_wrappers::log(log);

  // Create mesh submodule [mesh]
  nb::module_ mesh = m.def_submodule("mesh", "Mesh library module");
  dolfinx_wrappers::mesh(mesh);

  // Create graph submodule [graph]
  nb::module_ graph = m.def_submodule("graph", "Graph module");
  dolfinx_wrappers::graph(graph);

  // Create fem submodule [fem]
  nb::module_ fem = m.def_submodule("fem", "FEM module");
  dolfinx_wrappers::assemble(fem);
  dolfinx_wrappers::fem(fem);

  // Create geometry submodule
  nb::module_ geometry = m.def_submodule("geometry", "Geometry module");
  dolfinx_wrappers::geometry(geometry);

  // Create io submodule
  nb::module_ io = m.def_submodule("io", "I/O module");
  dolfinx_wrappers::io(io);

  // Create la submodule
  nb::module_ la = m.def_submodule("la", "Linear algebra module");
  dolfinx_wrappers::la(la);

  // Create refinement submodule
  nb::module_ refinement = m.def_submodule("refinement", "Refinement module");
  dolfinx_wrappers::refinement(refinement);

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)
  // PETSc-specific wrappers
  nb::module_ nls = m.def_submodule("nls", "Nonlinear solver module");
  dolfinx_wrappers::petsc(fem, la, nls);
#endif
}
