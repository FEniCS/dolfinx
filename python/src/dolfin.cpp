// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <iostream>

//#ifdef HAS_PETSC4PY
#include <petsc4py/petsc4py.h>
//#endif
#include <pybind11/pybind11.h>
#include <dolfin/log/log.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
  // common
  void common(py::module& m);
  void mpi(py::module& m);

  // log
  void log(py::module& m);

  void adaptivity(py::module& m);
  void ale(py::module& m);

  void function(py::module& m);
  void fem(py::module& m);

  void generation(py::module& m);
  void geometry(py::module& m);
  void graph(py::module& m);
  void io(py::module& m);
  void la(py::module& m);
  void math(py::module& m);
  void mesh(py::module& m);
  void multistage(py::module& m);
  void nls(py::module& m);
  void parameter(py::module& m);
  void refinement(py::module& m);
}


PYBIND11_MODULE(cpp, m)
{
  //#ifdef HAS_PETSC4PY
  //int ierr = import_petsc4py();
  //std::cout << "*************" << std::endl;
  //if (ierr != 0)
  //  throw std::runtime_error("Error importing petsc4py from pybind11 layer.");
  //#endif

  // Create module for C++ wrappers
  m.doc() ="DOLFIN Python interface";

  // Create common submodule [common]
  py::module common = m.def_submodule("common", "Common module");
  dolfin_wrappers::common(common);

  // Create MPI class [common]
  dolfin_wrappers::mpi(m);

  // Create common submodule [log]
  py::module log = m.def_submodule("log", "Logging module");
  dolfin_wrappers::log(log);

  // Create function submodule [function]
  py::module function = m.def_submodule("function",
                                        "Function module");
  dolfin_wrappers::function(function);


  // Create ale submodule [ale]
  py::module ale = m.def_submodule("ale", "ALE (mesh movement) module");
  dolfin_wrappers::ale(ale);

  // Create math submodule [math]
  py::module math = m.def_submodule("math", "Math library module");
  dolfin_wrappers::math(math);

  // Create mesh submodule [mesh]
  py::module mesh = m.def_submodule("mesh", "Mesh library module");
  dolfin_wrappers::mesh(mesh);

  // Create multistage submodule [multistage]
  py::module multistage = m.def_submodule("multistage", "Multistage integrator library module");
  dolfin_wrappers::multistage(multistage);

  // Create graph submodule [graph]
  py::module graph = m.def_submodule("graph", "Graph module");
  dolfin_wrappers::graph(graph);

  // Create fem submodule [fem]
  py::module fem = m.def_submodule("fem", "FEM module");
  dolfin_wrappers::fem(fem);

  // Create generation submodule [generation]
  py::module generation = m.def_submodule("generation",
                                          "Mesh generation module");
  dolfin_wrappers::generation(generation);

  // Create geometry submodule
  py::module geometry = m.def_submodule("geometry",
                                        "Geometry module");
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
  py::module refinement = m.def_submodule("refinement", "Mesh refinement module");
  dolfin_wrappers::refinement(refinement);

  // Create adaptivity submodule [adaptivity]
  py::module adaptivity = m.def_submodule("adaptivity", "Adaptivity module");
  dolfin_wrappers::adaptivity(adaptivity);

  // FIXME: these are just for the transition
  m.def("warning", [](std::string message) { dolfin::warning(message); });

}
