// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <Eigen/Dense>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DiscreteOperators.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/PETScDMCollection.h>
#include <dolfin/fem/assembler.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
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

  // Functions to convert pointers (from JIT usually) to UFC objects
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

  m.def("make_coordinate_mapping",
        [](std::uintptr_t e) {
          ufc_coordinate_mapping* p
              = reinterpret_cast<ufc_coordinate_mapping*>(e);
          return dolfin::fem::get_cmap_from_ufc_cmap(*p);
        },
        "Create a CoordinateMapping object from a pointer to a "
        "ufc_coordinate_map.");

  // utils
  m.def("create_vector", // TODO: change name to create_vector_block
        [](const std::vector<const dolfin::fem::Form*> L) {
          dolfin::la::PETScVector x = dolfin::fem::create_vector_block(L);
          Vec _x = x.vec();
          PetscObjectReference((PetscObject)_x);
          return _x;
        },
        py::return_value_policy::take_ownership,
        "Initialise monolithic vector for multiple (stacked) linear forms.");
  m.def("create_vector_nest",
        [](const std::vector<const dolfin::fem::Form*> L) {
          auto x = dolfin::fem::create_vector_nest(L);
          Vec _x = x.vec();
          PetscObjectReference((PetscObject)_x);
          return _x;
        },
        py::return_value_policy::take_ownership,
        "Initialise nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix",
        [](const dolfin::fem::Form& a) {
          auto A = dolfin::fem::create_matrix(a);
          Mat _A = A.mat();
          PetscObjectReference((PetscObject)_A);
          return _A;
        },
        py::return_value_policy::take_ownership,
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        [](std::vector<std::vector<const dolfin::fem::Form*>> a) {
          auto A = dolfin::fem::create_matrix_block(a);
          Mat _A = A.mat();
          PetscObjectReference((PetscObject)_A);
          return _A;
        },
        py::return_value_policy::take_ownership,
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        [](const std::vector<std::vector<const dolfin::fem::Form*>> a) {
          auto A = dolfin::fem::create_matrix_nest(a);
          Mat _A = A.mat();
          PetscObjectReference((PetscObject)_A);
          return _A;
        },
        py::return_value_policy::take_ownership,
        "Create nested sparse matrix for bilinear forms.");

  // dolfin::fem::FiniteElement
  py::class_<dolfin::fem::FiniteElement,
             std::shared_ptr<dolfin::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init<const ufc_finite_element&>())
      .def("num_sub_elements", &dolfin::fem::FiniteElement::num_sub_elements)
      .def("dof_reference_coordinates",
           &dolfin::fem::FiniteElement::dof_reference_coordinates)
      .def("space_dimension", &dolfin::fem::FiniteElement::space_dimension)
      .def("topological_dimension",
           &dolfin::fem::FiniteElement::topological_dimension)
      .def("value_dimension", &dolfin::fem::FiniteElement::value_dimension)
      .def("signature", &dolfin::fem::FiniteElement::signature);

  // dolfin::fem::GenericDofMap
  py::class_<dolfin::fem::GenericDofMap,
             std::shared_ptr<dolfin::fem::GenericDofMap>>(m, "GenericDofMap",
                                                          "DofMap object")
      .def_property_readonly(
          "global_dimension", &dolfin::fem::GenericDofMap::global_dimension,
          "The dimension of the global finite element function space")
      .def_property_readonly("index_map",
                             &dolfin::fem::GenericDofMap::index_map)
      .def("cell_dofs", &dolfin::fem::GenericDofMap::cell_dofs)
      .def("dofs", &dolfin::fem::GenericDofMap::dofs)
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
      .def("set", &dolfin::fem::GenericDofMap::set)
      .def("dof_array", &dolfin::fem::GenericDofMap::dof_array);

  // dolfin::fem::DofMap
  py::class_<dolfin::fem::DofMap, std::shared_ptr<dolfin::fem::DofMap>,
             dolfin::fem::GenericDofMap>(m, "DofMap", "DofMap object")
      .def(py::init<const ufc_dofmap&, const dolfin::mesh::Mesh&>());

  // dolfin::fem::CoordinateMapping
  py::class_<dolfin::fem::CoordinateMapping,
             std::shared_ptr<dolfin::fem::CoordinateMapping>>(
      m, "CoordinateMapping", "Coordinate mapping object");

  // dolfin::fem::DirichletBC
  py::class_<dolfin::fem::DirichletBC,
             std::shared_ptr<dolfin::fem::DirichletBC>>
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
                    const dolfin::mesh::SubDomain&,
                    dolfin::fem::DirichletBC::Method, bool>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domain"), py::arg("method"),
           py::arg("check_midpoint"))
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>,
                    std::shared_ptr<const dolfin::function::Function>,
                    const std::vector<std::int32_t>&,
                    dolfin::fem::DirichletBC::Method>(),
           py::arg("V"), py::arg("g"), py::arg("facets"), py::arg("method"))
      .def("function_space", &dolfin::fem::DirichletBC::function_space);

  // dolfin::fem::assemble
  m.def("assemble_scalar", &dolfin::fem::assemble_scalar,
        "Assemble functional over mesh");
  // Vectors (single)
  m.def("assemble_vector",
        py::overload_cast<Vec, const dolfin::fem::Form&>(
            &dolfin::fem::assemble_vector),
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing vector");
  // Block/nest vectors
  m.def("assemble_vector",
        py::overload_cast<
            Vec, std::vector<const dolfin::fem::Form*>,
            const std::vector<
                std::vector<std::shared_ptr<const dolfin::fem::Form>>>,
            std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
            const Vec, double>(&dolfin::fem::assemble_vector),
        "Re-assemble linear forms over mesh into blocked/nested vector");
  // Matrices
  m.def(
      "assemble_matrix",
      py::overload_cast<
          Mat, const dolfin::fem::Form&,
          std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>, double>(
          &dolfin::fem::assemble_matrix),
      py::arg("A"), py::arg("a"), py::arg("bcs"), py::arg("diagonal"),
      "Assemble bilinear form over mesh into matrix");
  m.def("assemble_blocked_matrix",
        py::overload_cast<
            Mat, const std::vector<std::vector<const dolfin::fem::Form*>>,
            std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
            double, bool>(&dolfin::fem::assemble_matrix),
        py::arg("A"), py::arg("a"), py::arg("bcs"), py::arg("diagonal"),
        py::arg("use_nest_extract") = true,
        "Re-assemble bilinear forms over mesh into blocked matrix");
  // BC modifiers
  m.def("apply_lifting", &dolfin::fem::apply_lifting,
        "Modify vector for lifted boundary conditions");
  m.def("set_bc", &dolfin::fem::set_bc,
        "Insert boundary condition values into vector");

  // dolfin::fem::DiscreteOperators
  py::class_<dolfin::fem::DiscreteOperators>(m, "DiscreteOperators")
      .def_static("build_gradient",
                  [](const dolfin::function::FunctionSpace& V0,
                     const dolfin::function::FunctionSpace& V1) {
                    dolfin::la::PETScMatrix A
                        = dolfin::fem::DiscreteOperators::build_gradient(V0,
                                                                         V1);
                    Mat _A = A.mat();
                    PetscObjectReference((PetscObject)_A);
                    return _A;
                  },
                  py::return_value_policy::take_ownership);

  // dolfin::fem::Form
  py::class_<dolfin::fem::Form, std::shared_ptr<dolfin::fem::Form>>(
      m, "Form", "Variational form object")
      .def(py::init<const ufc_form&,
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
      .def("set_tabulate_cell",
           [](dolfin::fem::Form& self, int i, std::intptr_t addr) {
             auto tabulate_tensor_ptr = (void (*)(
                 PetscScalar*, const PetscScalar*, const double*, int))addr;
             self.register_tabulate_tensor_cell(i, tabulate_tensor_ptr);
           })
      .def_property_readonly("rank", &dolfin::fem::Form::rank)
      .def("mesh", &dolfin::fem::Form::mesh)
      .def("function_space", &dolfin::fem::Form::function_space)
      .def("coordinate_mapping", &dolfin::fem::Form::coordinate_mapping);

  // dolfin::fem::PETScDMCollection
  py::class_<dolfin::fem::PETScDMCollection,
             std::shared_ptr<dolfin::fem::PETScDMCollection>>(
      m, "PETScDMCollection")
      .def(py::init<std::vector<
               std::shared_ptr<const dolfin::function::FunctionSpace>>>())
      .def_static(
          "create_transfer_matrix",
          [](const dolfin::function::FunctionSpace& V0,
             const dolfin::function::FunctionSpace& V1) {
            auto A = dolfin::fem::PETScDMCollection::create_transfer_matrix(V0,
                                                                            V1);
            Mat _A = A.mat();
            PetscObjectReference((PetscObject)_A);
            return _A;
          },
          py::return_value_policy::take_ownership)
      .def("check_ref_count", &dolfin::fem::PETScDMCollection::check_ref_count)
      .def("get_dm", &dolfin::fem::PETScDMCollection::get_dm);
} // namespace dolfin_wrappers
} // namespace dolfin_wrappers
