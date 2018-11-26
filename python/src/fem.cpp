// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>

#ifdef HAS_PYBIND11_PETSC4PY
#include <petsc4py/petsc4py.h>
#endif

#include "casters.h"
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DiscreteOperators.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/fem/PETScDMCollection.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/fem/assembler.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <ufc.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
void fem(py::module& m)
{
  // UFC objects
  py::class_<ufc_finite_element, std::shared_ptr<ufc_finite_element>>(
      m, "ufc_finite_element", "UFC finite element object");
  py::class_<ufc_dofmap, std::shared_ptr<ufc_dofmap>>(m, "ufc_dofmap",
                                                      "UFC dofmap object");
  py::class_<ufc_form, std::shared_ptr<ufc_form>>(m, "ufc_form",
                                                  "UFC form object");
  py::class_<ufc_coordinate_mapping, std::shared_ptr<ufc_coordinate_mapping>>(
      m, "ufc_coordinate_mapping", "UFC coordinate_mapping object");

  // Function to convert pointers (from JIT usually) to UFC objects
  m.def("make_ufc_finite_element",
        [](std::uintptr_t e) {
          ufc_finite_element* p = reinterpret_cast<ufc_finite_element*>(e);
          return std::shared_ptr<const ufc_finite_element>(p);
        },
        "Create a ufc_finite_element object from a pointer.");

  m.def("make_ufc_dofmap",
        [](std::uintptr_t e) {
          ufc_dofmap* p = reinterpret_cast<ufc_dofmap*>(e);
          return std::shared_ptr<const ufc_dofmap>(p);
        },
        "Create a ufc_dofmap object from a pointer.");

  m.def("make_ufc_form",
        [](std::uintptr_t e) {
          ufc_form* p = reinterpret_cast<ufc_form*>(e);
          return std::shared_ptr<const ufc_form>(p);
        },
        "Create a ufc_form object from a pointer.");

  m.def("make_ufc_coordinate_mapping",
        [](std::uintptr_t e) {
          ufc_coordinate_mapping* p
              = reinterpret_cast<ufc_coordinate_mapping*>(e);
          return std::shared_ptr<const ufc_coordinate_mapping>(p);
        },
        "Create a ufc_coordinate_mapping object from a pointer.");

  // dolfin::fem::FiniteElement
  py::class_<dolfin::fem::FiniteElement,
             std::shared_ptr<dolfin::fem::FiniteElement>>(
      m, "FiniteElement", "DOLFIN FiniteElement object")
      .def(py::init<std::shared_ptr<const ufc_finite_element>>())
      .def("num_sub_elements", &dolfin::fem::FiniteElement::num_sub_elements)
      .def("dof_reference_coordinates",
           &dolfin::fem::FiniteElement::dof_reference_coordinates)
      // TODO: Update for change to Eigen::Tensor
      //   .def("tabulate_dof_coordinates",
      //        [](const dolfin::fem::FiniteElement &self,
      //           const dolfin::mesh::Cell &cell) {
      //          // Get cell vertex coordinates
      //          std::vector<double> coordinate_dofs;
      //          cell.get_coordinate_dofs(coordinate_dofs);

      //          // Tabulate the coordinates
      //          boost::multi_array<double, 2> _dof_coords;
      //          self.tabulate_dof_coordinates(_dof_coords, coordinate_dofs,
      //          cell);

      //          // Copy data and return
      //          typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
      //                               Eigen::RowMajor>
      //              EigenArray;
      //          EigenArray dof_coords = Eigen::Map<EigenArray>(
      //              _dof_coords.data(), _dof_coords.shape()[0],
      //              _dof_coords.shape()[1]);
      //          return dof_coords;
      //        },
      //        "Tabulate coordinates of dofs on cell")
      .def("space_dimension", &dolfin::fem::FiniteElement::space_dimension)
      .def("topological_dimension",
           &dolfin::fem::FiniteElement::topological_dimension)
      .def("value_dimension", &dolfin::fem::FiniteElement::value_dimension)
      .def("signature", &dolfin::fem::FiniteElement::signature);

  // dolfin::fem::GenericDofMap
  py::class_<dolfin::fem::GenericDofMap,
             std::shared_ptr<dolfin::fem::GenericDofMap>,
             dolfin::common::Variable>(m, "GenericDofMap",
                                       "DOLFIN DofMap object")
      .def("global_dimension", &dolfin::fem::GenericDofMap::global_dimension,
           "The dimension of the global finite element function space")
      .def("index_map", &dolfin::fem::GenericDofMap::index_map)
      .def("neighbours", &dolfin::fem::GenericDofMap::neighbours)
      .def("shared_nodes", &dolfin::fem::GenericDofMap::shared_nodes)
      .def("cell_dofs", &dolfin::fem::GenericDofMap::cell_dofs)
      .def("dofs", (Eigen::Array<PetscInt, Eigen::Dynamic, 1>(
                       dolfin::fem::GenericDofMap::*)() const)
                       & dolfin::fem::GenericDofMap::dofs)
      .def("dofs", (Eigen::Array<PetscInt, Eigen::Dynamic, 1>(
                       dolfin::fem::GenericDofMap::*)(const dolfin::mesh::Mesh&,
                                                      std::size_t) const)
                       & dolfin::fem::GenericDofMap::dofs)
      .def("entity_dofs", (Eigen::Array<PetscInt, Eigen::Dynamic, 1>(
                              dolfin::fem::GenericDofMap::*)(
                              const dolfin::mesh::Mesh&, std::size_t) const)
                              & dolfin::fem::GenericDofMap::entity_dofs)
      .def("entity_dofs", (Eigen::Array<PetscInt, Eigen::Dynamic, 1>(
                              dolfin::fem::GenericDofMap::*)(
                              const dolfin::mesh::Mesh&, std::size_t,
                              const std::vector<std::size_t>&) const)
                              & dolfin::fem::GenericDofMap::entity_dofs)
      .def("num_entity_dofs", &dolfin::fem::GenericDofMap::num_entity_dofs)
      .def("tabulate_local_to_global_dofs",
           &dolfin::fem::GenericDofMap::tabulate_local_to_global_dofs)
      .def("tabulate_entity_dofs",
           &dolfin::fem::GenericDofMap::tabulate_entity_dofs)
      .def("block_size", &dolfin::fem::GenericDofMap::block_size)
      .def("set", &dolfin::fem::GenericDofMap::set);

  // dolfin::fem::DofMap
  py::class_<dolfin::fem::DofMap, std::shared_ptr<dolfin::fem::DofMap>,
             dolfin::fem::GenericDofMap>(m, "DofMap", "DofMap object")
      .def(py::init<std::shared_ptr<const ufc_dofmap>,
                    const dolfin::mesh::Mesh&>())
      .def("ownership_range", &dolfin::fem::DofMap::ownership_range)
      .def("cell_dofs", &dolfin::fem::DofMap::cell_dofs);

  py::class_<dolfin::fem::CoordinateMapping,
             std::shared_ptr<dolfin::fem::CoordinateMapping>>(
      m, "CoordinateMapping", "Coordinate mapping object")
      .def(py::init<std::shared_ptr<const ufc_coordinate_mapping>>());

  // dolfin::fem::SparsityPatternBuilder
  py::class_<dolfin::fem::SparsityPatternBuilder>(m, "SparsityPatternBuilder")
      .def_static(
          "build",
          [](const MPICommWrapper comm, const dolfin::mesh::Mesh& mesh,
             const std::array<const dolfin::fem::GenericDofMap*, 2> dofmaps,
             bool cells, bool interior_facets, bool exterior_facets,
             bool vertices, bool diagonal, bool finalize) {
            return dolfin::fem::SparsityPatternBuilder::build(
                comm.get(), mesh, dofmaps, cells, interior_facets,
                exterior_facets, vertices, diagonal, finalize);
          },
          py::arg("mpi_comm"), py::arg("mesh"), py::arg("dofmaps"),
          py::arg("cells"), py::arg("interior_facets"),
          py::arg("exterior_facets"), py::arg("vertices"), py::arg("diagonal"),
          py::arg("finalize") = true,
          "Create SparsityPattern from pair of dofmaps");

  // dolfin::fem::DirichletBC
  py::class_<dolfin::fem::DirichletBC,
             std::shared_ptr<dolfin::fem::DirichletBC>,
             dolfin::common::Variable>
      dirichletbc(
          m, "DirichletBC",
          "Object for representing Dirichlet (essential) boundary conditions");

  // dolfin::fem::DirichletBC  enum
  py::enum_<dolfin::fem::DirichletBC::Method>(dirichletbc, "Method")
      .value("topological", dolfin::fem::DirichletBC::Method::topological)
      .value("geometric", dolfin::fem::DirichletBC::Method::geometric)
      .value("pointwise", dolfin::fem::DirichletBC::Method::pointwise);

  dirichletbc
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>,
                    std::shared_ptr<const dolfin::function::Function>,
                    std::shared_ptr<const dolfin::mesh::SubDomain>,
                    dolfin::fem::DirichletBC::Method, bool>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domain"), py::arg("method"),
           py::arg("check_midpoint"))
      .def(
          py::init<std::shared_ptr<const dolfin::function::FunctionSpace>,
                   std::shared_ptr<const dolfin::function::Function>,
                   std::pair<std::shared_ptr<
                                 const dolfin::mesh::MeshFunction<std::size_t>>,
                             std::size_t>,
                   dolfin::fem::DirichletBC::Method>(),
          py::arg("V"), py::arg("g"), py::arg("sub_domain"), py::arg("method"))
      .def("function_space", &dolfin::fem::DirichletBC::function_space)
      .def("method", &dolfin::fem::DirichletBC::method)
      .def("get_boundary_values",
           [](const dolfin::fem::DirichletBC& instance) {
             dolfin::fem::DirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("user_subdomain", &dolfin::fem::DirichletBC::user_sub_domain)
      .def("set_value", &dolfin::fem::DirichletBC::set_value);

  py::enum_<dolfin::fem::BlockType>(
      m, "BlockType",
      "Enum for matrix/vector assembly type for nested problems")
      //   .value("monolithic", dolfin::fem::BlockType::monolithic,
      //          "Use monolithic linear algebra data structures for block
      //          forms")
      //   .value("nested", dolfin::fem::BlockType::nested,
      //          "Use nested linear algebra data structures for block forms");
      .value("monolithic", dolfin::fem::BlockType::monolithic)
      .value("nested", dolfin::fem::BlockType::nested);

  // dolfin::fem::assemble
  m.def("assemble",
        py::overload_cast<const dolfin::fem::Form&>(&dolfin::fem::assemble),
        "Assemble form over mesh");
  m.def("assemble_blocked_vector",
        py::overload_cast<
            std::vector<const dolfin::fem::Form*>,
            const std::vector<
                std::vector<std::shared_ptr<const dolfin::fem::Form>>>,
            std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
            const dolfin::la::PETScVector*, dolfin::fem::BlockType, double>(
            &dolfin::fem::assemble),
        py::arg("L"), py::arg("a"), py::arg("bcs"), py::arg("x0"),
        py::arg("block_type"), py::arg("scale") = 1.0,
        "Assemble linear forms over mesh into blocked vector");
  m.def("reassemble_blocked_vector",
        py::overload_cast<
            dolfin::la::PETScVector&, std::vector<const dolfin::fem::Form*>,
            const std::vector<
                std::vector<std::shared_ptr<const dolfin::fem::Form>>>,
            std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
            const dolfin::la::PETScVector*, double>(&dolfin::fem::assemble),
        py::arg("b"), py::arg("L"), py::arg("a"), py::arg("bcs"),
        py::arg("x0"), py::arg("scale") = 1.0,
        "Re-assemble linear forms over mesh into blocked vector");

  m.def("assemble_blocked_matrix",
        py::overload_cast<
            const std::vector<std::vector<const dolfin::fem::Form*>>,
            std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
            dolfin::fem::BlockType, double>(&dolfin::fem::assemble),
        py::arg("a"), py::arg("bcs"), py::arg("block_type"),
        py::arg("diagonal"),
        "Assemble bilinear forms over mesh into blocked matrix");
  m.def(
      "reassemble_blocked_matrix",
      py::overload_cast<
          dolfin::la::PETScMatrix&,
          const std::vector<std::vector<const dolfin::fem::Form*>>,
          std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>, double>(
          &dolfin::fem::assemble),
      py::arg("A"), py::arg("a"), py::arg("bcs"), py::arg("diagonal"),
      "Re-assemble bilinear forms over mesh into blocked matrix");
  m.def("set_bc", &dolfin::fem::set_bc,
        "Insert boundary condition values into vector");

  // dolfin::fem::AssemblerBase
  py::class_<dolfin::fem::AssemblerBase,
             std::shared_ptr<dolfin::fem::AssemblerBase>>(m, "AssemblerBase")
      .def_readwrite("add_values", &dolfin::fem::AssemblerBase::add_values)
      .def_readwrite("keep_diagonal",
                     &dolfin::fem::AssemblerBase::keep_diagonal)
      .def_readwrite("finalize_tensor",
                     &dolfin::fem::AssemblerBase::finalize_tensor);

  // dolfin::fem::SystemAssembler
  py::class_<dolfin::fem::SystemAssembler,
             std::shared_ptr<dolfin::fem::SystemAssembler>,
             dolfin::fem::AssemblerBase>(m, "SystemAssembler",
                                         "DOLFIN SystemAssembler object")
      .def(py::init<
           std::shared_ptr<const dolfin::fem::Form>,
           std::shared_ptr<const dolfin::fem::Form>,
           std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>>())
      .def("assemble", (void (dolfin::fem::SystemAssembler::*)(
                           dolfin::la::PETScMatrix&, dolfin::la::PETScVector&))
                           & dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(dolfin::la::PETScMatrix&))
               & dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(dolfin::la::PETScVector&))
               & dolfin::fem::SystemAssembler::assemble)
      .def("assemble", (void (dolfin::fem::SystemAssembler::*)(
                           dolfin::la::PETScMatrix&, dolfin::la::PETScVector&,
                           const dolfin::la::PETScVector&))
                           & dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(
               dolfin::la::PETScVector&, const dolfin::la::PETScVector&))
               & dolfin::fem::SystemAssembler::assemble);

  // dolfin::fem::DiscreteOperators
  py::class_<dolfin::fem::DiscreteOperators>(m, "DiscreteOperators")
      .def_static("build_gradient",
                  &dolfin::fem::DiscreteOperators::build_gradient);

  // dolfin::fem::Form
  py::class_<dolfin::fem::Form, std::shared_ptr<dolfin::fem::Form>>(
      m, "Form", "DOLFIN Form object")
      .def(py::init<std::shared_ptr<const ufc_form>,
                    std::vector<std::shared_ptr<
                        const dolfin::function::FunctionSpace>>>())
      .def(py::init<std::vector<
               std::shared_ptr<const dolfin::function::FunctionSpace>>>())
      .def("num_coefficients",
           [](const dolfin::fem::Form& self) { return self.coeffs().size(); },
           "Return number of coefficients in form")
      .def("original_coefficient_position",
           &dolfin::fem::Form::original_coefficient_position)
      .def("set_coefficient",
           [](dolfin::fem::Form& self, std::size_t i,
              std::shared_ptr<const dolfin::function::Function> f) {
             self.coeffs().set(i, f);
           })
      .def("set_mesh", &dolfin::fem::Form::set_mesh)
      .def("set_cell_domains", &dolfin::fem::Form::set_cell_domains)
      .def("set_exterior_facet_domains",
           &dolfin::fem::Form::set_exterior_facet_domains)
      .def("set_interior_facet_domains",
           &dolfin::fem::Form::set_interior_facet_domains)
      .def("set_vertex_domains", &dolfin::fem::Form::set_vertex_domains)
      .def("set_cell_tabulate",
           [](dolfin::fem::Form& self, unsigned int i, std::size_t addr) {
             auto tabulate_tensor_ptr = (void (*)(
                 PetscScalar*, const PetscScalar*, const double*, int))addr;
             self.integrals().set_cell_tabulate_tensor(i, tabulate_tensor_ptr);
           })
      .def("rank", &dolfin::fem::Form::rank)
      .def("mesh", &dolfin::fem::Form::mesh)
      .def("coordinate_mapping", &dolfin::fem::Form::coordinate_mapping);

  // dolfin::fem::NonlinearVariationalProblem
  py::class_<dolfin::fem::NonlinearVariationalProblem,
             std::shared_ptr<dolfin::fem::NonlinearVariationalProblem>>(
      m, "NonlinearVariationalProblem")
      .def(
          py::init<std::shared_ptr<const dolfin::fem::Form>,
                   std::shared_ptr<dolfin::function::Function>,
                   std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
                   std::shared_ptr<const dolfin::fem::Form>>())
      .def("set_bounds",
           py::overload_cast<std::shared_ptr<const dolfin::la::PETScVector>,
                             std::shared_ptr<const dolfin::la::PETScVector>>(
               &dolfin::fem::NonlinearVariationalProblem::set_bounds))
      .def("set_bounds",
           py::overload_cast<const dolfin::function::Function&,
                             const dolfin::function::Function&>(
               &dolfin::fem::NonlinearVariationalProblem::set_bounds));

  // dolfin::fem::PETScDMCollection
  py::class_<dolfin::fem::PETScDMCollection,
             std::shared_ptr<dolfin::fem::PETScDMCollection>>(
      m, "PETScDMCollection")
      .def(py::init<std::vector<
               std::shared_ptr<const dolfin::function::FunctionSpace>>>())
      .def_static("create_transfer_matrix",
                  &dolfin::fem::PETScDMCollection::create_transfer_matrix)
      .def("check_ref_count", &dolfin::fem::PETScDMCollection::check_ref_count)
      .def("get_dm", &dolfin::fem::PETScDMCollection::get_dm);
}
} // namespace dolfin_wrappers
