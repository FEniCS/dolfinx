// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#ifdef HAS_PYBIND11_PETSC4PY
#include <petsc4py/petsc4py.h>
#endif

#include "casters.h"
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DiscreteOperators.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/fem/PETScDMCollection.h>
#include <dolfin/fem/PointSource.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <ufc.h>

namespace py = pybind11;

namespace dolfin_wrappers {
void fem(py::module &m) {
  // UFC objects
  py::class_<ufc::finite_element, std::shared_ptr<ufc::finite_element>>(
      m, "ufc_finite_element", "UFC finite element object");
  py::class_<ufc::dofmap, std::shared_ptr<ufc::dofmap>>(m, "ufc_dofmap",
                                                        "UFC dofmap object");
  py::class_<ufc::form, std::shared_ptr<ufc::form>>(m, "ufc_form",
                                                    "UFC form object");

  // Function to convert pointers (from JIT usually) to UFC objects
  m.def("make_ufc_finite_element", [](std::uintptr_t e) {
    ufc::finite_element *p = reinterpret_cast<ufc::finite_element *>(e);
    return std::shared_ptr<const ufc::finite_element>(p);
  });

  m.def("make_ufc_dofmap", [](std::uintptr_t e) {
    ufc::dofmap *p = reinterpret_cast<ufc::dofmap *>(e);
    return std::shared_ptr<const ufc::dofmap>(p);
  });

  m.def("make_ufc_form", [](std::uintptr_t e) {
    ufc::form *p = reinterpret_cast<ufc::form *>(e);
    return std::shared_ptr<const ufc::form>(p);
  });

  // dolfin::fem::FiniteElement
  py::class_<dolfin::fem::FiniteElement,
             std::shared_ptr<dolfin::fem::FiniteElement>>(
      m, "FiniteElement", "DOLFIN FiniteElement object")
      .def(py::init<std::shared_ptr<const ufc::finite_element>>())
      .def("num_sub_elements", &dolfin::fem::FiniteElement::num_sub_elements)
      .def("tabulate_dof_coordinates",
           [](const dolfin::fem::FiniteElement &self,
              const dolfin::mesh::Cell &cell) {
             // Get cell vertex coordinates
             std::vector<double> coordinate_dofs;
             cell.get_coordinate_dofs(coordinate_dofs);

             // Tabulate the coordinates
             boost::multi_array<double, 2> _dof_coords;
             self.tabulate_dof_coordinates(_dof_coords, coordinate_dofs, cell);

             // Copy data and return
             typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>
                 EigenArray;
             EigenArray dof_coords = Eigen::Map<EigenArray>(
                 _dof_coords.data(), _dof_coords.shape()[0],
                 _dof_coords.shape()[1]);
             return dof_coords;
           },
           "Tabulate coordinates of dofs on cell")
      .def("evaluate_basis",
           [](const dolfin::fem::FiniteElement &self, int i,
              const py::array_t<double> x,
              const py::array_t<double> coordinate_dofs, int cell_orientation) {
             auto ufc_element = self.ufc_element();
             const std::size_t size = ufc_element->value_size();
             py::array_t<double, py::array::c_style> values(size);
             self.evaluate_basis(i, values.mutable_data(), x.data(),
                                 coordinate_dofs.data(), cell_orientation);
             return values;
           })
      .def("evaluate_basis_derivatives",
           [](const dolfin::fem::FiniteElement &self, int i, int order,
              const py::array_t<double> x,
              const py::array_t<double> coordinate_dofs, int cell_orientation) {
             auto ufc_element = self.ufc_element();

             const std::size_t gdim = self.geometric_dimension();
             const std::size_t num_derivs = pow(gdim, order);
             const std::size_t size = ufc_element->value_size() * num_derivs;
             py::array_t<double, py::array::c_style> values(size);
             self.evaluate_basis_derivatives(i, order, values.mutable_data(),
                                             x.data(), coordinate_dofs.data(),
                                             cell_orientation);
             return values;
           })
      .def("space_dimension", &dolfin::fem::FiniteElement::space_dimension)
      .def("geometric_dimension",
           &dolfin::fem::FiniteElement::geometric_dimension)
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
      .def("off_process_owner", &dolfin::fem::GenericDofMap::off_process_owner)
      .def("shared_nodes", &dolfin::fem::GenericDofMap::shared_nodes)
      .def("cell_dofs", &dolfin::fem::GenericDofMap::cell_dofs)
      .def("dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)()
                const) &
               dolfin::fem::GenericDofMap::dofs)
      .def("dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)(
               const dolfin::mesh::Mesh &, std::size_t) const) &
               dolfin::fem::GenericDofMap::dofs)
      .def("entity_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)(
               const dolfin::mesh::Mesh &, std::size_t) const) &
               dolfin::fem::GenericDofMap::entity_dofs)
      .def("entity_closure_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)(
               const dolfin::mesh::Mesh &, std::size_t) const) &
               dolfin::fem::GenericDofMap::entity_closure_dofs)
      .def("entity_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)(
               const dolfin::mesh::Mesh &, std::size_t,
               const std::vector<std::size_t> &) const) &
               dolfin::fem::GenericDofMap::entity_dofs)
      .def("entity_closure_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::fem::GenericDofMap::*)(
               const dolfin::mesh::Mesh &, std::size_t,
               const std::vector<std::size_t> &) const) &
               dolfin::fem::GenericDofMap::entity_closure_dofs)
      .def("num_entity_dofs", &dolfin::fem::GenericDofMap::num_entity_dofs)
      .def("tabulate_local_to_global_dofs",
           &dolfin::fem::GenericDofMap::tabulate_local_to_global_dofs)
      .def("clear_sub_map_data",
           &dolfin::fem::GenericDofMap::clear_sub_map_data)
      .def("tabulate_entity_dofs",
           [](const dolfin::fem::GenericDofMap &instance,
              std::size_t entity_dim, std::size_t cell_entity_index) {
             std::vector<std::size_t> dofs(
                 instance.num_entity_dofs(entity_dim));
             instance.tabulate_entity_dofs(dofs, entity_dim, cell_entity_index);
             return py::array_t<std::size_t>(dofs.size(), dofs.data());
           })
      .def("block_size", &dolfin::fem::GenericDofMap::block_size)
      .def("tabulate_local_to_global_dofs",
           [](const dolfin::fem::GenericDofMap &instance) {
             std::vector<std::size_t> dofs;
             instance.tabulate_local_to_global_dofs(dofs);
             return py::array_t<std::size_t>(dofs.size(), dofs.data());
           })
      .def("set", &dolfin::fem::GenericDofMap::set)
      .def_readonly("constrained_domain",
                    &dolfin::fem::GenericDofMap::constrained_domain);

  // dolfin::fem::DofMap
  py::class_<dolfin::fem::DofMap, std::shared_ptr<dolfin::fem::DofMap>,
             dolfin::fem::GenericDofMap>(m, "DofMap", "DOLFIN DofMap object")
      .def(py::init<std::shared_ptr<const ufc::dofmap>,
                    const dolfin::mesh::Mesh &>())
      .def(py::init<std::shared_ptr<const ufc::dofmap>,
                    const dolfin::mesh::Mesh &,
                    std::shared_ptr<const dolfin::mesh::SubDomain>>())
      .def("ownership_range", &dolfin::fem::DofMap::ownership_range)
      .def("cell_dofs", &dolfin::fem::DofMap::cell_dofs);

  // dolfin::fem::SparsityPatternBuilder
  py::class_<dolfin::fem::SparsityPatternBuilder>(m, "SparsityPatternBuilder")
      .def_static("build", &dolfin::fem::SparsityPatternBuilder::build,
                  py::arg("sparsity_pattern"), py::arg("mesh"),
                  py::arg("dofmaps"), py::arg("cells"),
                  py::arg("interior_facets"), py::arg("exterior_facets"),
                  py::arg("vertices"), py::arg("diagonal"),
                  py::arg("init") = true, py::arg("finalize") = true);

  // dolfin::fem::DirichletBC
  py::class_<dolfin::fem::DirichletBC,
             std::shared_ptr<dolfin::fem::DirichletBC>,
             dolfin::common::Variable>
      dirichletbc(m, "DirichletBC", "DirichletBC object");

  // dolfin::fem::DirichletBC  enum
  py::enum_<dolfin::fem::DirichletBC::Method>(dirichletbc, "Method")
      .value("topological", dolfin::fem::DirichletBC::Method::topological)
      .value("geometric", dolfin::fem::DirichletBC::Method::geometric)
      .value("pointwise", dolfin::fem::DirichletBC::Method::pointwise);

  dirichletbc.def(py::init<const dolfin::fem::DirichletBC &>())
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>,
                    std::shared_ptr<const dolfin::function::GenericFunction>,
                    std::shared_ptr<const dolfin::mesh::SubDomain>,
                    dolfin::fem::DirichletBC::Method, bool>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domain"),
           py::arg("method") = dolfin::fem::DirichletBC::Method::topological,
           py::arg("check_midpoint") = true)
      .def(py::init<
               std::shared_ptr<const dolfin::function::FunctionSpace>,
               std::shared_ptr<const dolfin::function::GenericFunction>,
               std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>,
               std::size_t, dolfin::fem::DirichletBC::Method>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domains"),
           py::arg("sub_domain"),
           py::arg("method") = dolfin::fem::DirichletBC::Method::topological)
      .def("function_space", &dolfin::fem::DirichletBC::function_space)
      .def("homogenize", &dolfin::fem::DirichletBC::homogenize)
      .def("method", &dolfin::fem::DirichletBC::method)
      .def("get_boundary_values",
           [](const dolfin::fem::DirichletBC &instance) {
             dolfin::fem::DirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("user_subdomain", &dolfin::fem::DirichletBC::user_sub_domain)
      .def("set_value", &dolfin::fem::DirichletBC::set_value)
      .def("set_value", [](dolfin::fem::DirichletBC &self, py::object value) {
        auto _u =
            value.attr("_cpp_object")
                .cast<
                    std::shared_ptr<const dolfin::function::GenericFunction>>();
        self.set_value(_u);
      });

  // dolfin::fem::Assembler
  py::class_<dolfin::fem::Assembler, std::shared_ptr<dolfin::fem::Assembler>>
      assembler(
          m, "Assembler",
          "Assembler object for assembling forms into matrices and vectors");

  // dolfin::fem::Assembler::BlockType enum
  py::enum_<dolfin::fem::Assembler::BlockType>(assembler, "BlockType")
      .value("nested", dolfin::fem::Assembler::BlockType::nested)
      .value("monolithic", dolfin::fem::Assembler::BlockType::monolithic);

  // dolfin::fem::Assembler
  assembler
      .def(py::init<
           std::vector<std::vector<std::shared_ptr<const dolfin::fem::Form>>>,
           std::vector<std::shared_ptr<const dolfin::fem::Form>>,
           std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>>())
      .def("assemble", py::overload_cast<dolfin::la::PETScMatrix &,
                                         dolfin::la::PETScVector &>(
                           &dolfin::fem::Assembler::assemble))
      .def("assemble", py::overload_cast<dolfin::la::PETScMatrix &,
                                         dolfin::fem::Assembler::BlockType>(
                           &dolfin::fem::Assembler::assemble))
      .def("assemble", py::overload_cast<dolfin::la::PETScVector &>(
                           &dolfin::fem::Assembler::assemble));

  // dolfin::fem::AssemblerBase
  py::class_<dolfin::fem::AssemblerBase,
             std::shared_ptr<dolfin::fem::AssemblerBase>>(m, "AssemblerBase")
      //.def("init_global_tensor",
      //&dolfin::fem::AssemblerBase::init_global_tensor)
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
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(dolfin::la::PETScMatrix &,
                                                   dolfin::la::PETScVector &)) &
               dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(dolfin::la::PETScMatrix &)) &
               dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(dolfin::la::PETScVector &)) &
               dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(
               dolfin::la::PETScMatrix &, dolfin::la::PETScVector &,
               const dolfin::la::PETScVector &)) &
               dolfin::fem::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::fem::SystemAssembler::*)(
               dolfin::la::PETScVector &, const dolfin::la::PETScVector &)) &
               dolfin::fem::SystemAssembler::assemble);

  // dolfin::fem::DiscreteOperators
  py::class_<dolfin::fem::DiscreteOperators>(m, "DiscreteOperators")
      .def_static("build_gradient",
                  &dolfin::fem::DiscreteOperators::build_gradient)
      .def_static("build_gradient", [](py::object V0, py::object V1) {
        auto _V0 =
            V0.attr("_cpp_object").cast<dolfin::function::FunctionSpace *>();
        auto _V1 =
            V1.attr("_cpp_object").cast<dolfin::function::FunctionSpace *>();
        return dolfin::fem::DiscreteOperators::build_gradient(*_V0, *_V1);
      });

  // dolfin::fem::Form
  py::class_<dolfin::fem::Form, std::shared_ptr<dolfin::fem::Form>>(
      m, "Form", "DOLFIN Form object")
      .def(py::init<std::shared_ptr<const ufc::form>,
                    std::vector<std::shared_ptr<
                        const dolfin::function::FunctionSpace>>>())
      .def("num_coefficients",
           [](const dolfin::fem::Form &self) { return self.coeffs().size(); },
           "Return number of coefficients in form")
      .def("original_coefficient_position",
           &dolfin::fem::Form::original_coefficient_position)
      .def("set_coefficient",
           [](dolfin::fem::Form &self, std::size_t i,
              std::shared_ptr<const dolfin::function::GenericFunction> f) {
             self.coeffs().set(i, f);
           })
      .def("set_mesh", &dolfin::fem::Form::set_mesh)
      .def("set_cell_domains", &dolfin::fem::Form::set_cell_domains)
      .def("set_exterior_facet_domains",
           &dolfin::fem::Form::set_exterior_facet_domains)
      .def("set_interior_facet_domains",
           &dolfin::fem::Form::set_interior_facet_domains)
      .def("set_vertex_domains", &dolfin::fem::Form::set_vertex_domains)
      .def("rank", &dolfin::fem::Form::rank)
      .def("mesh", &dolfin::fem::Form::mesh);

  // dolfin::fem::PointSource
  py::class_<dolfin::fem::PointSource,
             std::shared_ptr<dolfin::fem::PointSource>>(m, "PointSource")
      // FIXME: consolidate down to one intialiser when switching from
      // SWIG to pybind11
      .def(
          py::init([](
              py::object V,
              const std::vector<std::pair<dolfin::geometry::Point, double>>
                  values) {
            std::shared_ptr<const dolfin::function::FunctionSpace> _V;
            if (py::hasattr(V, "_cpp_object"))
              _V =
                  V.attr("_cpp_object")
                      .cast<std::shared_ptr<dolfin::function::FunctionSpace>>();
            else
              _V = V.cast<std::shared_ptr<dolfin::function::FunctionSpace>>();

            return std::make_unique<dolfin::fem::PointSource>(_V, values);
          }),
          py::arg("V"), py::arg("values"))
      .def(
          py::init([](
              py::object V0, py::object V1,
              const std::vector<std::pair<dolfin::geometry::Point, double>>
                  values) {
            std::shared_ptr<const dolfin::function::FunctionSpace> _V0, _V1;
            if (py::hasattr(V0, "_cpp_object"))
              _V0 =
                  V0.attr("_cpp_object")
                      .cast<std::shared_ptr<dolfin::function::FunctionSpace>>();
            else
              _V0 = V0.cast<std::shared_ptr<dolfin::function::FunctionSpace>>();

            if (py::hasattr(V1, "_cpp_object"))
              _V1 =
                  V1.attr("_cpp_object")
                      .cast<std::shared_ptr<dolfin::function::FunctionSpace>>();
            else
              _V1 = V1.cast<std::shared_ptr<dolfin::function::FunctionSpace>>();

            return std::make_unique<dolfin::fem::PointSource>(_V0, _V1, values);
          }),
          py::arg("V0"), py::arg("V1"), py::arg("values"))
      .def("apply",
           (void (dolfin::fem::PointSource::*)(dolfin::la::PETScVector &)) &
               dolfin::fem::PointSource::apply)
      .def("apply",
           (void (dolfin::fem::PointSource::*)(dolfin::la::PETScMatrix &)) &
               dolfin::fem::PointSource::apply);

  // dolfin::fem::NonlinearVariationalProblem
  py::class_<dolfin::fem::NonlinearVariationalProblem,
             std::shared_ptr<dolfin::fem::NonlinearVariationalProblem>>(
      m, "NonlinearVariationalProblem")
      .def(
          py::init<std::shared_ptr<const dolfin::fem::Form>,
                   std::shared_ptr<dolfin::function::Function>,
                   std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>,
                   std::shared_ptr<const dolfin::fem::Form>>())
      // FIXME: is there a better way to handle the casting
      .def("set_bounds",
           (void (dolfin::fem::NonlinearVariationalProblem::*)(
               std::shared_ptr<const dolfin::la::PETScVector>,
               std::shared_ptr<const dolfin::la::PETScVector>)) &
               dolfin::fem::NonlinearVariationalProblem::set_bounds)
      .def("set_bounds",
           (void (dolfin::fem::NonlinearVariationalProblem::*)(
               const dolfin::function::Function &,
               const dolfin::function::Function &)) &
               dolfin::fem::NonlinearVariationalProblem::set_bounds)
      .def("set_bounds", [](dolfin::fem::NonlinearVariationalProblem &self,
                            py::object lb, py::object ub) {
        auto &_lb = lb.attr("_cpp_object").cast<dolfin::function::Function &>();
        auto &_ub = ub.attr("_cpp_object").cast<dolfin::function::Function &>();
        self.set_bounds(_lb, _ub);
      });

#ifdef HAS_PETSC
  // dolfin::fem::PETScDMCollection
  py::class_<dolfin::fem::PETScDMCollection,
             std::shared_ptr<dolfin::fem::PETScDMCollection>>(
      m, "PETScDMCollection")
      .def(py::init<std::vector<
               std::shared_ptr<const dolfin::function::FunctionSpace>>>())
      .def(py::init([](py::list V) {
        std::vector<std::shared_ptr<const dolfin::function::FunctionSpace>> _V;
        for (auto space : V) {
          auto _space =
              space.attr("_cpp_object")
                  .cast<
                      std::shared_ptr<const dolfin::function::FunctionSpace>>();
          _V.push_back(_space);
        }
        return dolfin::fem::PETScDMCollection(_V);
      }))
      .def_static("create_transfer_matrix",
                  &dolfin::fem::PETScDMCollection::create_transfer_matrix)
      .def_static(
          "create_transfer_matrix",
          [](py::object V_coarse, py::object V_fine) {
            auto _V0 = V_coarse.attr("_cpp_object")
                           .cast<dolfin::function::FunctionSpace *>();
            auto _V1 = V_fine.attr("_cpp_object")
                           .cast<dolfin::function::FunctionSpace *>();
            return dolfin::fem::PETScDMCollection::create_transfer_matrix(*_V0,
                                                                          *_V1);
          })
      .def("check_ref_count", &dolfin::fem::PETScDMCollection::check_ref_count)
      .def("get_dm", &dolfin::fem::PETScDMCollection::get_dm);
#endif

  // FEM utils free functions
  // m.def("create_mesh", dolfin::fem::create_mesh);
  // m.def("create_mesh", [](const py::object u) {
  //  auto _u = u.attr("_cpp_object").cast<dolfin::function::Function*>();
  //  return dolfin::fem::create_mesh(*_u);
  //});

  m.def("set_coordinates", &dolfin::fem::set_coordinates);
  m.def("set_coordinates", [](dolfin::mesh::MeshGeometry &geometry,
                              const py::object u) {
    auto _u = u.attr("_cpp_object").cast<const dolfin::function::Function *>();
    dolfin::fem::set_coordinates(geometry, *_u);
  });

  m.def("get_coordinates", &dolfin::fem::get_coordinates);
  m.def("get_coordinates",
        [](py::object u, const dolfin::mesh::MeshGeometry &geometry) {
          auto _u = u.attr("_cpp_object").cast<dolfin::function::Function *>();
          return dolfin::fem::get_coordinates(*_u, geometry);
        });

  m.def("vertex_to_dof_map", [](const dolfin::function::FunctionSpace &V) {
    const auto _v2d = dolfin::fem::vertex_to_dof_map(V);
    return py::array_t<dolfin::la_index_t>(_v2d.size(), _v2d.data());
  });

  m.def("vertex_to_dof_map", [](py::object V) {
    auto _V = V.attr("_cpp_object").cast<dolfin::function::FunctionSpace *>();
    const auto _v2d = dolfin::fem::vertex_to_dof_map(*_V);
    return py::array_t<dolfin::la_index_t>(_v2d.size(), _v2d.data());
  });
  m.def("dof_to_vertex_map", &dolfin::fem::dof_to_vertex_map);
  m.def("dof_to_vertex_map", [](py::object V) {
    auto _V = V.attr("_cpp_object").cast<dolfin::function::FunctionSpace *>();
    const auto _d2v = dolfin::fem::dof_to_vertex_map(*_V);
    return py::array_t<std::size_t>(_d2v.size(), _d2v.data());
  });
}
} // namespace dolfin_wrappers
