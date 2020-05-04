// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DiscreteOperators.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
#include <ufc.h>

namespace py = pybind11;

namespace
{
// Copy a vector-of-vectors into an Eigen::Array for dolfinx::fem::Form*
Eigen::Array<const dolfinx::fem::Form*, Eigen::Dynamic, Eigen::Dynamic,
             Eigen::RowMajor>
forms_vector_to_array(
    const std::vector<std::vector<const dolfinx::fem::Form*>>& a)
{
  if (a.empty())
  {
    return Eigen::Array<const dolfinx::fem::Form*, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::RowMajor>();
  }
  Eigen::Array<const dolfinx::fem::Form*, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>
      _a(a.size(), a[0].size());
  _a = nullptr;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    if (a[i].size() != a[0].size())
      throw std::runtime_error("Array of forms is not rectangular.");
    for (std::size_t j = 0; j < a[i].size(); ++j)
      _a(i, j) = a[i][j];
  }
  return _a;
}

} // namespace

namespace dolfinx_wrappers
{
void fem(py::module& m)
{
  // utils
  m.def("block_function_spaces",
        [](const std::vector<std::vector<const dolfinx::fem::Form*>>& a) {
          return dolfinx::fem::block_function_spaces(forms_vector_to_array(a));
        });
  m.def(
      "create_vector_block",
      [](const std::vector<
          std::reference_wrapper<const dolfinx::common::IndexMap>>& maps) {
        dolfinx::la::PETScVector x = dolfinx::fem::create_vector_block(maps);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def(
      "create_vector_nest",
      [](const std::vector<const dolfinx::common::IndexMap*>& maps) {
        auto x = dolfinx::fem::create_vector_nest(maps);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create nested vector for multiple (stacked) linear forms.");

  m.def("create_sparsity_pattern", &dolfinx::fem::create_sparsity_pattern,
        "Create a sparsity pattern for bilinear form.");
  m.def("pack_coefficients", &dolfinx::fem::pack_coefficients,
        "Pack coefficients for a UFL form.");
  m.def("pack_constants", &dolfinx::fem::pack_constants,
        "Pack constants for a UFL form.");
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form& a) {
        auto A = dolfinx::fem::create_matrix(a);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat for bilinear form.");
  m.def(
      "create_matrix_block",
      [](const std::vector<std::vector<const dolfinx::fem::Form*>>& a) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_block(forms_vector_to_array(a));
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def(
      "create_matrix_nest",
      [](const std::vector<std::vector<const dolfinx::fem::Form*>>& a) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_nest(forms_vector_to_array(a));
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create nested sparse matrix for bilinear forms.");
  m.def("create_element_dof_layout", &dolfinx::fem::create_element_dof_layout,
        "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "create_dofmap",
      [](const MPICommWrapper comm, const std::uintptr_t dofmap,
         dolfinx::mesh::Topology& topology) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_dofmap(comm.get(), *p, topology);
      },
      "Create DofMap object from a pointer to ufc_dofmap.");
  m.def(
      "create_form",
      [](const std::uintptr_t form,
         const std::vector<
             std::shared_ptr<const dolfinx::function::FunctionSpace>>& spaces) {
        const ufc_form* p = reinterpret_cast<const ufc_form*>(form);
        return dolfinx::fem::create_form(*p, spaces);
      },
      "Create Form from a pointer to ufc_form.");
  m.def(
      "create_coordinate_map",
      [](std::uintptr_t cmap) {
        const ufc_coordinate_mapping* p
            = reinterpret_cast<const ufc_coordinate_mapping*>(cmap);
        return dolfinx::fem::create_coordinate_map(*p);
      },
      "Create CoordinateElement from a pointer to ufc_coordinate_map.");
  m.def(
      "build_dofmap",
      [](const dolfinx::mesh::Mesh& mesh,
         std::shared_ptr<const dolfinx::fem::ElementDofLayout>
             element_dof_layout) {
        return dolfinx::fem::DofMapBuilder::build(
            mesh.mpi_comm(), mesh.topology(), element_dof_layout);
      },
      "Build and dofmap on a mesh.");
  m.def(
      "build_dofmap",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& element_dof_layout, int bs) {
        // See https://github.com/pybind/pybind11/issues/1138 on why we need
        // to convert from a std::unique_ptr to a std::shard_ptr
        auto [map, dofmap] = dolfinx::fem::DofMapBuilder::build(
            comm.get(), topology, element_dof_layout, bs);
        return std::pair(
            std::shared_ptr<const dolfinx::common::IndexMap>(std::move(map)),
            std::move(dofmap));
      },
      "Build and dofmap on a mesh.");

  // dolfinx::fem::FiniteElement
  py::class_<dolfinx::fem::FiniteElement,
             std::shared_ptr<dolfinx::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init([](const std::uintptr_t ufc_element) {
        const ufc_finite_element* p
            = reinterpret_cast<const ufc_finite_element*>(ufc_element);
        return std::make_unique<dolfinx::fem::FiniteElement>(*p);
      }))
      .def("num_sub_elements", &dolfinx::fem::FiniteElement::num_sub_elements)
      .def("dof_reference_coordinates",
           &dolfinx::fem::FiniteElement::dof_reference_coordinates)
      .def("space_dimension", &dolfinx::fem::FiniteElement::space_dimension)
      .def("value_dimension", &dolfinx::fem::FiniteElement::value_dimension)
      .def("signature", &dolfinx::fem::FiniteElement::signature);

  // dolfinx::fem::ElementDofLayout
  py::class_<dolfinx::fem::ElementDofLayout,
             std::shared_ptr<dolfinx::fem::ElementDofLayout>>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(py::init<int, const std::vector<std::vector<std::set<int>>>&,
                    const std::vector<int>&,
                    const std::vector<
                        std::shared_ptr<const dolfinx::fem::ElementDofLayout>>,
                    const dolfinx::mesh::CellType,
                    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&>())
      .def_property_readonly("num_dofs",
                             &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs)
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs)
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs)
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs);

  // dolfinx::fem::DofMap
  py::class_<dolfinx::fem::DofMap, std::shared_ptr<dolfinx::fem::DofMap>>(
      m, "DofMap", "DofMap object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::ElementDofLayout>,
                    std::shared_ptr<const dolfinx::common::IndexMap>,
                    dolfinx::graph::AdjacencyList<std::int32_t>&>(),
           py::arg("element_dof_layout"), py::arg("index_map"),
           py::arg("dofmap"))
      .def_readonly("index_map", &dolfinx::fem::DofMap::index_map)
      .def_readonly("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def("cell_dofs", &dolfinx::fem::DofMap::cell_dofs)
      .def("list", &dolfinx::fem::DofMap::list);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def_property_readonly("dof_layout",
                             &dolfinx::fem::CoordinateElement::dof_layout)
      .def("push_forward", &dolfinx::fem::CoordinateElement::push_forward);

  // dolfinx::fem::DirichletBC
  py::class_<dolfinx::fem::DirichletBC,
             std::shared_ptr<dolfinx::fem::DirichletBC>>
      dirichletbc(
          m, "DirichletBC",
          "Object for representing Dirichlet (essential) boundary conditions");

  dirichletbc
      .def(py::init<std::shared_ptr<const dolfinx::function::Function>,
                    const Eigen::Ref<
                        const Eigen::Array<std::int32_t, Eigen::Dynamic, 2>>&,
                    std::shared_ptr<const dolfinx::function::FunctionSpace>>(),
           py::arg("V"), py::arg("g"), py::arg("V_g_dofs"))
      .def(
          py::init<std::shared_ptr<const dolfinx::function::Function>,
                   const Eigen::Ref<
                       const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&>(),
          py::arg("g"), py::arg("dofs"))
      .def_property_readonly("dof_indices", &dolfinx::fem::DirichletBC::dofs)
      .def_property_readonly("function_space",
                             &dolfinx::fem::DirichletBC::function_space)
      .def_property_readonly("value", &dolfinx::fem::DirichletBC::value);

  // dolfinx::fem::assemble
  m.def("assemble_scalar", &dolfinx::fem::assemble_scalar,
        "Assemble functional over mesh");
  // Vectors (single)
  m.def("assemble_vector",
        py::overload_cast<Vec, const dolfinx::fem::Form&>(
            &dolfinx::fem::assemble_vector),
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing vector");
  m.def("assemble_vector",
        py::overload_cast<
            Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>,
            const dolfinx::fem::Form&>(&dolfinx::fem::assemble_vector),
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing Eigen vector");
  // Matrices
  m.def(
      "assemble_matrix",
      py::overload_cast<
          Mat, const dolfinx::fem::Form&,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&>(
          &dolfinx::fem::assemble_matrix));
  m.def("assemble_matrix",
        py::overload_cast<Mat, const dolfinx::fem::Form&,
                          const std::vector<bool>&, const std::vector<bool>&>(
            &dolfinx::fem::assemble_matrix));
  m.def(
      "add_diagonal",
      py::overload_cast<
          Mat, const dolfinx::function::FunctionSpace&,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&,
          PetscScalar>(&dolfinx::fem::add_diagonal));

  m.def("assemble_scalar", &dolfinx::fem::assemble_scalar,
        "Assemble functional over mesh");
  // Vectors (single)
  m.def("assemble_vector",
        py::overload_cast<Vec, const dolfinx::fem::Form&>(
            &dolfinx::fem::assemble_vector),
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing vector");
  m.def("assemble_vector",
        py::overload_cast<
            Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>,
            const dolfinx::fem::Form&>(&dolfinx::fem::assemble_vector),
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing Eigen vector");
  // Matrices
  m.def(
      "assemble_matrix",
      py::overload_cast<
          Mat, const dolfinx::fem::Form&,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&>(
          &dolfinx::fem::assemble_matrix));
  m.def("assemble_matrix",
        py::overload_cast<Mat, const dolfinx::fem::Form&,
                          const std::vector<bool>&, const std::vector<bool>&>(
            &dolfinx::fem::assemble_matrix));
  m.def(
      "add_diagonal",
      py::overload_cast<
          Mat, const dolfinx::function::FunctionSpace&,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&,
          PetscScalar>(&dolfinx::fem::add_diagonal));

  m.def("assemble_matrix_eigen", &dolfinx::fem::assemble_matrix_eigen);

  // BC modifiers
  m.def("apply_lifting",
        py::overload_cast<
            Vec, const std::vector<std::shared_ptr<const dolfinx::fem::Form>>&,
            const std::vector<
                std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>>&,
            const std::vector<Vec>&, double>(&dolfinx::fem::apply_lifting),
        "Modify vector for lifted boundary conditions");
  m.def(
      "apply_lifting",
      py::overload_cast<
          Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>,
          const std::vector<std::shared_ptr<const dolfinx::fem::Form>>&,
          const std::vector<
              std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>>&,
          const std::vector<
              Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>&,
          double>(&dolfinx::fem::apply_lifting),
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      py::overload_cast<
          Vec,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&,
          const Vec, double>(&dolfinx::fem::set_bc),
      "Insert boundary condition values into vector");
  m.def(
      "set_bc",
      [](Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>&
             bcs,
         const py::array_t<PetscScalar>& x0, double scale) {
        if (x0.ndim() == 0)
          dolfinx::fem::set_bc(b, bcs, scale);
        else if (x0.ndim() == 1)
        {
          Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x0(
              x0.data(), x0.shape(0));
          dolfinx::fem::set_bc(b, bcs, _x0, scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
  // Tools
  m.def("bcs_rows", &dolfinx::fem::bcs_rows);
  m.def("bcs_cols", &dolfinx::fem::bcs_cols);

  //   // dolfinx::fem::DiscreteOperators
  //   py::class_<dolfinx::fem::DiscreteOperators>(m, "DiscreteOperators")
  //       .def_static(
  //           "build_gradient",
  //           [](const dolfinx::function::FunctionSpace& V0,
  //              const dolfinx::function::FunctionSpace& V1) {
  //             dolfinx::la::PETScMatrix A
  //                 = dolfinx::fem::DiscreteOperators::build_gradient(V0, V1);
  //             Mat _A = A.mat();
  //             PetscObjectReference((PetscObject)_A);
  //             return _A;
  //           },
  //           py::return_value_policy::take_ownership);

  // dolfinx::fem::FormIntegrals
  py::class_<dolfinx::fem::FormIntegrals,
             std::shared_ptr<dolfinx::fem::FormIntegrals>>
      formintegrals(m, "FormIntegrals",
                    "Holder for integral kernels and domains");
  formintegrals.def("integral_ids", &dolfinx::fem::FormIntegrals::integral_ids)
      .def(
          "integral_domains",
          [](dolfinx::fem::FormIntegrals& self,
             dolfinx::fem::FormIntegrals::Type type, int i) {
            const std::vector<std::int32_t>& domains
                = self.integral_domains(type, i);

            return py::array_t<std::int32_t>(domains.size(), domains.data(),
                                             py::none());
          },
          py::return_value_policy::reference_internal,
          "Return active domains for given integral");

  py::enum_<dolfinx::fem::FormIntegrals::Type>(formintegrals, "Type")
      .value("cell", dolfinx::fem::FormIntegrals::Type::cell)
      .value("exterior_facet",
             dolfinx::fem::FormIntegrals::Type::exterior_facet)
      .value("interior_facet",
             dolfinx::fem::FormIntegrals::Type::interior_facet);

  // dolfinx::fem::Form
  py::class_<dolfinx::fem::Form, std::shared_ptr<dolfinx::fem::Form>>(
      m, "Form", "Variational form object")
      .def(py::init<std::vector<
               std::shared_ptr<const dolfinx::function::FunctionSpace>>>())
      .def("integrals", &dolfinx::fem::Form::integrals)
      .def(
          "num_coefficients",
          [](const dolfinx::fem::Form& self) {
            return self.coefficients().size();
          },
          "Return number of coefficients in form")
      .def("original_coefficient_position",
           &dolfinx::fem::Form::original_coefficient_position)
      .def("set_coefficient",
           [](dolfinx::fem::Form& self, std::size_t i,
              std::shared_ptr<const dolfinx::function::Function> f) {
             self.coefficients().set(i, f);
           })
      .def("set_constants",
           py::overload_cast<
               std::vector<std::shared_ptr<const dolfinx::function::Constant>>>(
               &dolfinx::fem::Form::set_constants))
      .def("set_mesh", &dolfinx::fem::Form::set_mesh)
      .def("set_cell_domains", &dolfinx::fem::Form::set_cell_domains)
      .def("set_exterior_facet_domains",
           &dolfinx::fem::Form::set_exterior_facet_domains)
      .def("set_interior_facet_domains",
           &dolfinx::fem::Form::set_interior_facet_domains)
      .def("set_vertex_domains", &dolfinx::fem::Form::set_vertex_domains)
      .def("set_tabulate_tensor",
           [](dolfinx::fem::Form& self, dolfinx::fem::FormIntegrals::Type type,
              int i, py::object addr) {
             auto tabulate_tensor_ptr = (void (*)(
                 PetscScalar*, const PetscScalar*, const PetscScalar*,
                 const double*, const int*, const std::uint8_t*,
                 const std::uint32_t))addr.cast<std::uintptr_t>();
             self.set_tabulate_tensor(type, i, tabulate_tensor_ptr);
           })
      .def_property_readonly("rank", &dolfinx::fem::Form::rank)
      .def("mesh", &dolfinx::fem::Form::mesh)
      .def("function_space", &dolfinx::fem::Form::function_space);

  m.def("locate_dofs_topological", &dolfinx::fem::locate_dofs_topological,
        py::arg("V"), py::arg("dim"), py::arg("entities"),
        py::arg("remote") = true);
  m.def("locate_dofs_geometrical", &dolfinx::fem::locate_dofs_geometrical);
} // namespace dolfinx_wrappers
} // namespace dolfinx_wrappers
