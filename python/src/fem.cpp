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
  py::class_<ufc::finite_element, std::shared_ptr<ufc::finite_element>>(
      m, "ufc_finite_element", "UFC finite element object");
  py::class_<ufc::dofmap, std::shared_ptr<ufc::dofmap>>(m, "ufc_dofmap",
                                                        "UFC dofmap object");
  py::class_<ufc::form, std::shared_ptr<ufc::form>>(m, "ufc_form",
                                                    "UFC form object");

  // Function to convert pointers (from JIT usually) to UFC objects
  m.def("make_ufc_finite_element", [](std::uintptr_t e) {
    ufc::finite_element* p = reinterpret_cast<ufc::finite_element*>(e);
    return std::shared_ptr<const ufc::finite_element>(p);
  });

  m.def("make_ufc_dofmap", [](std::uintptr_t e) {
    ufc::dofmap* p = reinterpret_cast<ufc::dofmap*>(e);
    return std::shared_ptr<const ufc::dofmap>(p);
  });

  m.def("make_ufc_form", [](std::uintptr_t e) {
    ufc::form* p = reinterpret_cast<ufc::form*>(e);
    return std::shared_ptr<const ufc::form>(p);
  });

  // dolfin::FiniteElement
  py::class_<dolfin::FiniteElement, std::shared_ptr<dolfin::FiniteElement>>(
      m, "FiniteElement", "DOLFIN FiniteElement object")
      .def(py::init<std::shared_ptr<const ufc::finite_element>>())
      .def("num_sub_elements", &dolfin::FiniteElement::num_sub_elements)
      .def("tabulate_dof_coordinates",
           [](const dolfin::FiniteElement& self, const dolfin::Cell& cell) {
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
           [](const dolfin::FiniteElement& self, int i,
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
           [](const dolfin::FiniteElement& self, int i, int order,
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
      .def("space_dimension", &dolfin::FiniteElement::space_dimension)
      .def("geometric_dimension", &dolfin::FiniteElement::geometric_dimension)
      .def("value_dimension", &dolfin::FiniteElement::value_dimension)
      .def("signature", &dolfin::FiniteElement::signature);

  // dolfin::GenericDofMap
  py::class_<dolfin::GenericDofMap, std::shared_ptr<dolfin::GenericDofMap>,
             dolfin::Variable>(m, "GenericDofMap", "DOLFIN DofMap object")
      .def("global_dimension", &dolfin::GenericDofMap::global_dimension,
           "The dimension of the global finite element function space")
      .def("index_map", &dolfin::GenericDofMap::index_map)
      .def("neighbours", &dolfin::GenericDofMap::neighbours)
      .def("off_process_owner", &dolfin::GenericDofMap::off_process_owner)
      .def("shared_nodes", &dolfin::GenericDofMap::shared_nodes)
      .def("cell_dofs", &dolfin::GenericDofMap::cell_dofs)
      .def("dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)() const)
               & dolfin::GenericDofMap::dofs)
      .def("dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)(
               const dolfin::Mesh&, std::size_t) const)
               & dolfin::GenericDofMap::dofs)
      .def("entity_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)(
               const dolfin::Mesh&, std::size_t) const)
               & dolfin::GenericDofMap::entity_dofs)
      .def("entity_closure_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)(
               const dolfin::Mesh&, std::size_t) const)
               & dolfin::GenericDofMap::entity_closure_dofs)
      .def("entity_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)(
               const dolfin::Mesh&, std::size_t,
               const std::vector<std::size_t>&) const)
               & dolfin::GenericDofMap::entity_dofs)
      .def("entity_closure_dofs",
           (std::vector<dolfin::la_index_t>(dolfin::GenericDofMap::*)(
               const dolfin::Mesh&, std::size_t,
               const std::vector<std::size_t>&) const)
               & dolfin::GenericDofMap::entity_closure_dofs)
      .def("num_entity_dofs", &dolfin::GenericDofMap::num_entity_dofs)
      .def("tabulate_local_to_global_dofs",
           &dolfin::GenericDofMap::tabulate_local_to_global_dofs)
      .def("clear_sub_map_data", &dolfin::GenericDofMap::clear_sub_map_data)
      .def("tabulate_entity_dofs",
           [](const dolfin::GenericDofMap& instance, std::size_t entity_dim,
              std::size_t cell_entity_index) {
             std::vector<std::size_t> dofs(
                 instance.num_entity_dofs(entity_dim));
             instance.tabulate_entity_dofs(dofs, entity_dim, cell_entity_index);
             return py::array_t<std::size_t>(dofs.size(), dofs.data());
           })
      .def("block_size", &dolfin::GenericDofMap::block_size)
      .def("tabulate_local_to_global_dofs",
           [](const dolfin::GenericDofMap& instance) {
             std::vector<std::size_t> dofs;
             instance.tabulate_local_to_global_dofs(dofs);
             return py::array_t<std::size_t>(dofs.size(), dofs.data());
           })
      .def("set", &dolfin::GenericDofMap::set)
      .def_readonly("constrained_domain",
                    &dolfin::GenericDofMap::constrained_domain);

  // dolfin::DofMap
  py::class_<dolfin::DofMap, std::shared_ptr<dolfin::DofMap>,
             dolfin::GenericDofMap>(m, "DofMap", "DOLFIN DofMap object")
      .def(py::init<std::shared_ptr<const ufc::dofmap>, const dolfin::Mesh&>())
      .def(py::init<std::shared_ptr<const ufc::dofmap>, const dolfin::Mesh&,
                    std::shared_ptr<const dolfin::SubDomain>>())
      .def("ownership_range", &dolfin::DofMap::ownership_range)
      .def("cell_dofs", &dolfin::DofMap::cell_dofs);

  // dolfin::SparsityPatternBuilder
  py::class_<dolfin::SparsityPatternBuilder>(m, "SparsityPatternBuilder")
      .def_static("build", &dolfin::SparsityPatternBuilder::build,
                  py::arg("sparsity_pattern"), py::arg("mesh"),
                  py::arg("dofmaps"), py::arg("cells"),
                  py::arg("interior_facets"), py::arg("exterior_facets"),
                  py::arg("vertices"), py::arg("diagonal"),
                  py::arg("init") = true, py::arg("finalize") = true);

  // dolfin::DirichletBC
  py::class_<dolfin::DirichletBC, std::shared_ptr<dolfin::DirichletBC>,
             dolfin::Variable>(m, "DirichletBC", "DOLFIN DirichletBC object")
      .def(py::init<const dolfin::DirichletBC&>())
      .def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
                    std::shared_ptr<const dolfin::GenericFunction>,
                    std::shared_ptr<const dolfin::SubDomain>, std::string,
                    bool>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domain"),
           py::arg("method") = "topological", py::arg("check_midpoint") = true)
      .def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
                    std::shared_ptr<const dolfin::GenericFunction>,
                    std::shared_ptr<const dolfin::MeshFunction<std::size_t>>,
                    std::size_t, std::string>(),
           py::arg("V"), py::arg("g"), py::arg("sub_domains"),
           py::arg("sub_domain"), py::arg("method") = "topological")
      .def("function_space", &dolfin::DirichletBC::function_space)
      .def("homogenize", &dolfin::DirichletBC::homogenize)
      .def("method", &dolfin::DirichletBC::method)
      .def("get_boundary_values",
           [](const dolfin::DirichletBC& instance) {
             dolfin::DirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("user_subdomain", &dolfin::DirichletBC::user_sub_domain)
      .def("set_value", &dolfin::DirichletBC::set_value)
      .def("set_value", [](dolfin::DirichletBC& self, py::object value) {
        auto _u = value.attr("_cpp_object")
                      .cast<std::shared_ptr<const dolfin::GenericFunction>>();
        self.set_value(_u);
      });

  // dolfin::fem::Assembler
  py::class_<dolfin::fem::Assembler, std::shared_ptr<dolfin::fem::Assembler>>(
      m, "Assembler",
      "Assembler object for assembling forms into matrices and vectors")
      .def(py::init<std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>());

  // dolfin::AssemblerBase
  py::class_<dolfin::AssemblerBase, std::shared_ptr<dolfin::AssemblerBase>>(
      m, "AssemblerBase")
      //.def("init_global_tensor", &dolfin::AssemblerBase::init_global_tensor)
      .def_readwrite("add_values", &dolfin::AssemblerBase::add_values)
      .def_readwrite("keep_diagonal", &dolfin::AssemblerBase::keep_diagonal)
      .def_readwrite("finalize_tensor",
                     &dolfin::AssemblerBase::finalize_tensor);

  // dolfin::SystemAssembler
  py::class_<dolfin::SystemAssembler, std::shared_ptr<dolfin::SystemAssembler>,
             dolfin::AssemblerBase>(m, "SystemAssembler",
                                    "DOLFIN SystemAssembler object")
      .def(py::init<std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def("assemble",
           (void (dolfin::SystemAssembler::*)(dolfin::PETScMatrix&,
                                              dolfin::PETScVector&))
               & dolfin::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::SystemAssembler::*)(dolfin::PETScMatrix&))
               & dolfin::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::SystemAssembler::*)(dolfin::PETScVector&))
               & dolfin::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::SystemAssembler::*)(dolfin::PETScMatrix&,
                                              dolfin::PETScVector&,
                                              const dolfin::PETScVector&))
               & dolfin::SystemAssembler::assemble)
      .def("assemble",
           (void (dolfin::SystemAssembler::*)(dolfin::PETScVector&,
                                              const dolfin::PETScVector&))
               & dolfin::SystemAssembler::assemble);

  // dolfin::DiscreteOperators
  py::class_<dolfin::DiscreteOperators>(m, "DiscreteOperators")
      .def_static("build_gradient", &dolfin::DiscreteOperators::build_gradient)
      .def_static("build_gradient", [](py::object V0, py::object V1) {
        auto _V0 = V0.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
        auto _V1 = V1.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
        return dolfin::DiscreteOperators::build_gradient(*_V0, *_V1);
      });

  // dolfin::Form
  py::class_<dolfin::Form, std::shared_ptr<dolfin::Form>>(m, "Form",
                                                          "DOLFIN Form object")
      .def(
          py::init<std::shared_ptr<const ufc::form>,
                   std::vector<std::shared_ptr<const dolfin::FunctionSpace>>>())
      .def("num_coefficients", &dolfin::Form::num_coefficients,
           "Return number of coefficients in form")
      .def("original_coefficient_position",
           &dolfin::Form::original_coefficient_position)
      .def("set_coefficient",
           (void (dolfin::Form::*)(
               std::size_t, std::shared_ptr<const dolfin::GenericFunction>))
               & dolfin::Form::set_coefficient,
           "Doc")
      .def("set_coefficient",
           (void (dolfin::Form::*)(
               std::string, std::shared_ptr<const dolfin::GenericFunction>))
               & dolfin::Form::set_coefficient,
           "Doc")
      .def("set_mesh", &dolfin::Form::set_mesh)
      .def("set_cell_domains", &dolfin::Form::set_cell_domains)
      .def("set_exterior_facet_domains",
           &dolfin::Form::set_exterior_facet_domains)
      .def("set_interior_facet_domains",
           &dolfin::Form::set_interior_facet_domains)
      .def("set_vertex_domains", &dolfin::Form::set_vertex_domains)
      .def("rank", &dolfin::Form::rank)
      .def("mesh", &dolfin::Form::mesh);

  // dolfin::PointSource
  py::class_<dolfin::PointSource, std::shared_ptr<dolfin::PointSource>>(
      m, "PointSource")
      // FIXME: consolidate down to one intialiser when switching from
      // SWIG to pybind11
      .def(py::init(
               [](py::object V,
                  const std::vector<std::pair<dolfin::Point, double>> values) {
                 std::shared_ptr<const dolfin::FunctionSpace> _V;
                 if (py::hasattr(V, "_cpp_object"))
                   _V = V.attr("_cpp_object")
                            .cast<std::shared_ptr<dolfin::FunctionSpace>>();
                 else
                   _V = V.cast<std::shared_ptr<dolfin::FunctionSpace>>();

                 return dolfin::PointSource(_V, values);
               }),
           py::arg("V"), py::arg("values"))
      .def(py::init(
               [](py::object V0, py::object V1,
                  const std::vector<std::pair<dolfin::Point, double>> values) {
                 std::shared_ptr<const dolfin::FunctionSpace> _V0, _V1;
                 if (py::hasattr(V0, "_cpp_object"))
                   _V0 = V0.attr("_cpp_object")
                             .cast<std::shared_ptr<dolfin::FunctionSpace>>();
                 else
                   _V0 = V0.cast<std::shared_ptr<dolfin::FunctionSpace>>();

                 if (py::hasattr(V1, "_cpp_object"))
                   _V1 = V1.attr("_cpp_object")
                             .cast<std::shared_ptr<dolfin::FunctionSpace>>();
                 else
                   _V1 = V1.cast<std::shared_ptr<dolfin::FunctionSpace>>();

                 return dolfin::PointSource(_V0, _V1, values);
               }),
           py::arg("V0"), py::arg("V1"), py::arg("values"))
      //
      //.def(py::init<std::shared_ptr<const dolfin::FunctionSpace>, const
      // dolfin::Point&, double>(),
      //     py::arg("V"), py::arg("p"), py::arg("value"))
      //.def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
      // std::shared_ptr<const dolfin::FunctionSpace>, const dolfin::Point&,
      // double>(),
      //     py::arg("V0"), py::arg("V1"), py::arg("p"), py::arg("value"))
      //.def(py::init<std::shared_ptr<const dolfin::FunctionSpace>, const
      // std::vector<std::pair<const dolfin::Point*, double>>>())
      //.def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
      // std::shared_ptr<const dolfin::FunctionSpace>,
      //     const std::vector<std::pair<const dolfin::Point*, double>>>())
      .def("apply",
           (void (dolfin::PointSource::*)(dolfin::PETScVector&))
               & dolfin::PointSource::apply)
      .def("apply",
           (void (dolfin::PointSource::*)(dolfin::PETScMatrix&))
               & dolfin::PointSource::apply);

  // dolfin::NonlinearVariationalProblem
  py::class_<dolfin::NonlinearVariationalProblem,
             std::shared_ptr<dolfin::NonlinearVariationalProblem>>(
      m, "NonlinearVariationalProblem")
      .def(py::init<std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<dolfin::Function>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>,
                    std::shared_ptr<const dolfin::Form>>())
      // FIXME: is there a better way to handle the casting
      .def("set_bounds",
           (void (dolfin::NonlinearVariationalProblem::*)(
               std::shared_ptr<const dolfin::PETScVector>,
               std::shared_ptr<const dolfin::PETScVector>))
               & dolfin::NonlinearVariationalProblem::set_bounds)
      .def("set_bounds",
           (void (dolfin::NonlinearVariationalProblem::*)(
               const dolfin::Function&, const dolfin::Function&))
               & dolfin::NonlinearVariationalProblem::set_bounds)
      .def("set_bounds", [](dolfin::NonlinearVariationalProblem& self,
                            py::object lb, py::object ub) {
        auto& _lb = lb.attr("_cpp_object").cast<dolfin::Function&>();
        auto& _ub = ub.attr("_cpp_object").cast<dolfin::Function&>();
        self.set_bounds(_lb, _ub);
      });

#ifdef HAS_PETSC
  // dolfin::PETScDMCollection
  py::class_<dolfin::PETScDMCollection,
             std::shared_ptr<dolfin::PETScDMCollection>>(m, "PETScDMCollection")
      .def(
          py::init<std::vector<std::shared_ptr<const dolfin::FunctionSpace>>>())
      .def(py::init([](py::list V) {
        std::vector<std::shared_ptr<const dolfin::FunctionSpace>> _V;
        for (auto space : V)
        {
          auto _space
              = space.attr("_cpp_object")
                    .cast<std::shared_ptr<const dolfin::FunctionSpace>>();
          _V.push_back(_space);
        }
        return dolfin::PETScDMCollection(_V);
      }))
      .def_static("create_transfer_matrix",
                  &dolfin::PETScDMCollection::create_transfer_matrix)
      .def_static(
          "create_transfer_matrix",
          [](py::object V_coarse, py::object V_fine) {
            auto _V0
                = V_coarse.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
            auto _V1
                = V_fine.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
            return dolfin::PETScDMCollection::create_transfer_matrix(*_V0,
                                                                     *_V1);
          })
      .def("check_ref_count", &dolfin::PETScDMCollection::check_ref_count)
      .def("get_dm", &dolfin::PETScDMCollection::get_dm);
#endif

  // FEM utils free functions
  // m.def("create_mesh", dolfin::fem::create_mesh);
  // m.def("create_mesh", [](const py::object u) {
  //  auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
  //  return dolfin::fem::create_mesh(*_u);
  //});

  m.def("set_coordinates", &dolfin::fem::set_coordinates);
  m.def("set_coordinates",
        [](dolfin::MeshGeometry& geometry, const py::object u) {
          auto _u = u.attr("_cpp_object").cast<const dolfin::Function*>();
          dolfin::fem::set_coordinates(geometry, *_u);
        });

  m.def("get_coordinates", &dolfin::fem::get_coordinates);
  m.def("get_coordinates",
        [](py::object u, const dolfin::MeshGeometry& geometry) {
          auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
          return dolfin::fem::get_coordinates(*_u, geometry);
        });

  m.def("vertex_to_dof_map", [](const dolfin::FunctionSpace& V) {
    const auto _v2d = dolfin::fem::vertex_to_dof_map(V);
    return py::array_t<dolfin::la_index_t>(_v2d.size(), _v2d.data());
  });

  m.def("vertex_to_dof_map", [](py::object V) {
    auto _V = V.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
    const auto _v2d = dolfin::fem::vertex_to_dof_map(*_V);
    return py::array_t<dolfin::la_index_t>(_v2d.size(), _v2d.data());
  });
  m.def("dof_to_vertex_map", &dolfin::fem::dof_to_vertex_map);
  m.def("dof_to_vertex_map", [](py::object V) {
    auto _V = V.attr("_cpp_object").cast<dolfin::FunctionSpace*>();
    const auto _d2v = dolfin::fem::dof_to_vertex_map(*_V);
    return py::array_t<std::size_t>(_d2v.size(), _d2v.data());
  });
}
}
