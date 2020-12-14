// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DiscreteOperators.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
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
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
#include <ufc.h>

namespace py = pybind11;

namespace
{
// Copy a vector-of-vectors into an Eigen::Array for dolfinx::fem::Form*
Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
             Eigen::Dynamic, Eigen::RowMajor>
forms_vector_to_array(
    const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>& a)
{
  if (a.empty())
  {
    return Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::RowMajor>();
  }
  Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
               Eigen::Dynamic, Eigen::RowMajor>
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
  m.def(
      "create_vector_block",
      [](const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps) {
        dolfinx::la::PETScVector x = dolfinx::fem::create_vector_block(maps);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def(
      "create_vector_nest",
      [](const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps) {
        auto x = dolfinx::fem::create_vector_nest(maps);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create nested vector for multiple (stacked) linear forms.");

  m.def("create_sparsity_pattern",
        &dolfinx::fem::create_sparsity_pattern<PetscScalar>,
        "Create a sparsity pattern for bilinear form.");
  m.def("pack_coefficients",
        &dolfinx::fem::pack_coefficients<dolfinx::fem::Form<PetscScalar>>,
        "Pack coefficients for a UFL form.");
  m.def("pack_coefficients",
        &dolfinx::fem::pack_coefficients<dolfinx::fem::Form<PetscScalar>>,
        "Pack coefficients for a UFL expression.");
  m.def("pack_constants",
        &dolfinx::fem::pack_constants<dolfinx::fem::Form<PetscScalar>>,
        "Pack constants for a UFL form.");
  m.def("pack_constants",
        &dolfinx::fem::pack_constants<dolfinx::fem::Expression<PetscScalar>>,
        "Pack constants for a UFL expression.");
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<PetscScalar>& a, const std::string& type) {
        dolfinx::la::PETScMatrix A = dolfinx::fem::create_matrix(a, type);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership, py::arg("a"),
      py::arg("type") = std::string(), "Create a PETSc Mat for bilinear form.");
  m.def(
      "create_matrix_block",
      [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>&
             a,
         const std::string& type) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_block(forms_vector_to_array(a), type);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership, py::arg("a"),
      py::arg("type") = std::string(),
      "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def(
      "create_matrix_nest",
      [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>&
             a,
         const std::vector<std::vector<std::string>>& types) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_nest(forms_vector_to_array(a), types);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership, py::arg("a"),
      py::arg("types") = std::vector<std::vector<std::string>>(),
      "Create nested sparse matrix for bilinear forms.");
  m.def(
      "create_element_dof_layout",
      [](const std::uintptr_t dofmap, const dolfinx::mesh::CellType cell_type,
         const std::vector<int>& parent_map) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_element_dof_layout(*p, cell_type,
                                                       parent_map);
      },
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
         const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace>>&
             spaces,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Function<PetscScalar>>>& coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Constant<PetscScalar>>>& constants,
         const std::map<dolfinx::fem::IntegralType,
                        const dolfinx::mesh::MeshTags<int>*>& subdomains,
         const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
        const ufc_form* p = reinterpret_cast<const ufc_form*>(form);
        return dolfinx::fem::create_form<PetscScalar>(
            *p, spaces, coefficients, constants, subdomains, mesh);
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
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& element_dof_layout) {
        auto [map, bs, dofmap] = dolfinx::fem::DofMapBuilder::build(
            comm.get(), topology, element_dof_layout);
        return std::tuple(map, bs, std::move(dofmap));
      },
      "Build and dofmap on a mesh.");
  m.def("transpose_dofmap", &dolfinx::fem::transpose_dofmap,
        "Build the index to (cell, local index) map from a "
        "dofmap ((cell, local index ) -> index).");

  // dolfinx::fem::FiniteElement
  py::class_<dolfinx::fem::FiniteElement,
             std::shared_ptr<dolfinx::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init([](const std::uintptr_t ufc_element) {
        const ufc_finite_element* p
            = reinterpret_cast<const ufc_finite_element*>(ufc_element);
        return dolfinx::fem::FiniteElement(*p);
      }))
      .def("num_sub_elements", &dolfinx::fem::FiniteElement::num_sub_elements)
      .def("dof_reference_coordinates",
           &dolfinx::fem::FiniteElement::dof_reference_coordinates)
      .def_property_readonly("value_rank",
                             &dolfinx::fem::FiniteElement::value_rank)
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
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs)
      .def("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  py::class_<dolfinx::fem::DofMap, std::shared_ptr<dolfinx::fem::DofMap>>(
      m, "DofMap", "DofMap object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::ElementDofLayout>,
                    std::shared_ptr<const dolfinx::common::IndexMap>, int,
                    dolfinx::graph::AdjacencyList<std::int32_t>&, int>(),
           py::arg("element_dof_layout"), py::arg("index_map"),
           py::arg("index_map_bs"), py::arg("dofmap"), py::arg("bs"))
      .def_readonly("index_map", &dolfinx::fem::DofMap::index_map)
      .def_property_readonly("index_map_bs",
                             &dolfinx::fem::DofMap::index_map_bs)
      .def_readonly("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def("cell_dofs", &dolfinx::fem::DofMap::cell_dofs)
      .def_property_readonly("bs", &dolfinx::fem::DofMap::bs)
      .def("list", &dolfinx::fem::DofMap::list);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def_property_readonly("dof_layout",
                             &dolfinx::fem::CoordinateElement::dof_layout)
      .def("push_forward", &dolfinx::fem::CoordinateElement::push_forward)
      .def_readwrite("non_affine_atol",
                     &dolfinx::fem::CoordinateElement::non_affine_atol)
      .def_readwrite("non_affine_max_its",
                     &dolfinx::fem::CoordinateElement::non_affine_max_its);

  // dolfinx::fem::DirichletBC
  py::class_<dolfinx::fem::DirichletBC<PetscScalar>,
             std::shared_ptr<dolfinx::fem::DirichletBC<PetscScalar>>>
      dirichletbc(
          m, "DirichletBC",
          "Object for representing Dirichlet (essential) boundary conditions");

  dirichletbc
      .def(py::init<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>,
                    const std::array<
                        Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>&,
                    std::shared_ptr<const dolfinx::fem::FunctionSpace>>(),
           py::arg("V"), py::arg("g"), py::arg("V_g_dofs"))
      .def(
          py::init<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>,
                   const Eigen::Ref<
                       const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&>(),
          py::arg("g"), py::arg("dofs"))
      .def("dof_indices", &dolfinx::fem::DirichletBC<PetscScalar>::dofs_owned)
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::DirichletBC<PetscScalar>::function_space)
      .def_property_readonly("value",
                             &dolfinx::fem::DirichletBC<PetscScalar>::value);

  // dolfinx::fem::assemble
  // Functional
  m.def("assemble_scalar", &dolfinx::fem::assemble_scalar<PetscScalar>,
        "Assemble functional over mesh");
  // Vector
  m.def("assemble_vector", &dolfinx::fem::assemble_vector<PetscScalar>,
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing Eigen vector");
  // Matrices
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::add_block_fn(A), a, bcs);
        });
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<bool>& rows0, const std::vector<bool>& rows1) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::add_block_fn(A), a, rows0, rows1);
        });
  m.def("assemble_matrix_petsc_unrolled",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::add_block_expand_fn(
                  A, a.function_spaces()[0]->dofmap()->bs(),
                  a.function_spaces()[1]->dofmap()->bs()),
              a, bcs);
        });
  m.def("assemble_matrix_petsc_unrolled",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<bool>& rows0, const std::vector<bool>& rows1) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::add_block_expand_fn(
                  A, a.function_spaces()[0]->dofmap()->bs(),
                  a.function_spaces()[1]->dofmap()->bs()),
              a, rows0, rows1);
        });
  m.def("add_diagonal",
        [](Mat A, const dolfinx::fem::FunctionSpace& V,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           PetscScalar diagonal) {
          dolfinx::fem::add_diagonal(dolfinx::la::PETScMatrix::add_fn(A), V,
                                     bcs, diagonal);
        });
  m.def("assemble_matrix",
        py::overload_cast<const std::function<int(
                              std::int32_t, const std::int32_t*, std::int32_t,
                              const std::int32_t*, const PetscScalar*)>&,
                          const dolfinx::fem::Form<PetscScalar>&,
                          const std::vector<std::shared_ptr<
                              const dolfinx::fem::DirichletBC<PetscScalar>>>&>(
            &dolfinx::fem::assemble_matrix<PetscScalar>));

  // BC modifiers
  m.def("apply_lifting", &dolfinx::fem::apply_lifting<PetscScalar>,
        "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
         const py::array_t<PetscScalar>& x0, double scale) {
        if (x0.ndim() == 0)
          dolfinx::fem::set_bc<PetscScalar>(b, bcs, scale);
        else if (x0.ndim() == 1)
        {
          Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x0(
              x0.data(), x0.shape(0));
          dolfinx::fem::set_bc<PetscScalar>(b, bcs, _x0, scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
  // Tools
  m.def("bcs_rows", &dolfinx::fem::bcs_rows<PetscScalar>);
  m.def("bcs_cols", &dolfinx::fem::bcs_cols<PetscScalar>);

  //   // dolfinx::fem::DiscreteOperators
  //   py::class_<dolfinx::fem::DiscreteOperators>(m, "DiscreteOperators")
  //       .def_static(
  //           "build_gradient",
  //           [](const dolfinx::fem::FunctionSpace& V0,
  //              const dolfinx::fem::FunctionSpace& V1) {
  //             dolfinx::la::PETScMatrix A
  //                 = dolfinx::fem::DiscreteOperators::build_gradient(V0, V1);
  //             Mat _A = A.mat();
  //             PetscObjectReference((PetscObject)_A);
  //             return _A;
  //           },
  //           py::return_value_policy::take_ownership);

  py::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell)
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet)
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet)
      .value("vertex", dolfinx::fem::IntegralType::vertex);

  // dolfinx::fem::Form
  py::class_<dolfinx::fem::Form<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Form<PetscScalar>>>(
      m, "Form", "Variational form object")
      .def(
          py::init(
              [](const std::vector<std::shared_ptr<
                     const dolfinx::fem::FunctionSpace>>& spaces,
                 const std::map<
                     dolfinx::fem::IntegralType,
                     std::pair<std::vector<std::pair<int, py::object>>,
                               const dolfinx::mesh::MeshTags<int>*>>& integrals,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Function<PetscScalar>>>& coefficients,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Constant<PetscScalar>>>& constants,
                 bool needs_permutation_data,
                 const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
                using kern = std::function<void(
                    PetscScalar*, const PetscScalar*, const PetscScalar*,
                    const double*, const int*, const std::uint8_t*,
                    const std::uint32_t)>;
                std::map<dolfinx::fem::IntegralType,
                         std::pair<std::vector<std::pair<int, kern>>,
                                   const dolfinx::mesh::MeshTags<int>*>>
                    _integrals;

                // Loop over kernel for each entity type
                for (auto& kernel_type : integrals)
                {
                  // Set subdomain markers
                  _integrals[kernel_type.first].second = nullptr;

                  // Loop over each domain kernel
                  for (auto& kernel : kernel_type.second.first)
                  {
                    auto tabulate_tensor_ptr
                        = (void (*)(PetscScalar*, const PetscScalar*,
                                    const PetscScalar*, const double*,
                                    const int*, const std::uint8_t*,
                                    const std::uint32_t))
                              kernel.second.cast<std::uintptr_t>();
                    _integrals[kernel_type.first].first.push_back(
                        {kernel.first, tabulate_tensor_ptr});
                  }
                }
                return dolfinx::fem::Form<PetscScalar>(
                    spaces, _integrals, coefficients, constants,
                    needs_permutation_data, mesh);
              }),
          py::arg("spaces"), py::arg("integrals"), py::arg("coefficients"),
          py::arg("constants"), py::arg("need_permutation_data"),
          py::arg("mesh") = py::none())
      .def_property_readonly("coefficients",
                             &dolfinx::fem::Form<PetscScalar>::coefficients)
      .def_property_readonly("rank", &dolfinx::fem::Form<PetscScalar>::rank)
      .def_property_readonly("mesh", &dolfinx::fem::Form<PetscScalar>::mesh)
      .def_property_readonly("function_spaces",
                             &dolfinx::fem::Form<PetscScalar>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<PetscScalar>::integral_ids)
      .def("domains", &dolfinx::fem::Form<PetscScalar>::domains);

  m.def(
      "locate_dofs_topological",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
         bool remote) {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        return dolfinx::fem::locate_dofs_topological({V[0], V[1]}, dim,
                                                     entities, remote);
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def("locate_dofs_topological",
        py::overload_cast<const dolfinx::fem::FunctionSpace&, const int,
                          const Eigen::Ref<const Eigen::ArrayXi>&, bool>(
            &dolfinx::fem::locate_dofs_topological),
        py::arg("V"), py::arg("dim"), py::arg("entities"),
        py::arg("remote") = true);

  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
             const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                 Eigen::RowMajor>>&)>& marker) {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        return dolfinx::fem::locate_dofs_geometrical({V[0], V[1]}, marker);
      },
      py::arg("V"), py::arg("marker"));
  m.def("locate_dofs_geometrical",
        py::overload_cast<
            const dolfinx::fem::FunctionSpace&,
            const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
                const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                    Eigen::RowMajor>>&)>&>(
            &dolfinx::fem::locate_dofs_geometrical),
        py::arg("V"), py::arg("marker"));

  // dolfinx::fem::Function
  py::class_<dolfinx::fem::Function<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::fem::FunctionSpace>,
                    std::shared_ptr<dolfinx::la::Vector<PetscScalar>>>())
      .def_readwrite("name", &dolfinx::fem::Function<PetscScalar>::name)
      .def_property_readonly("id", &dolfinx::fem::Function<PetscScalar>::id)
      .def("sub", &dolfinx::fem::Function<PetscScalar>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<PetscScalar>::collapse,
           "Collapse sub-function view")
      .def("interpolate",
           py::overload_cast<const std::function<Eigen::Array<
               PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
               const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                   Eigen::RowMajor>>&)>&>(
               &dolfinx::fem::Function<PetscScalar>::interpolate),
           py::arg("f"), "Interpolate an expression")
      .def("interpolate",
           py::overload_cast<const dolfinx::fem::Function<PetscScalar>&>(
               &dolfinx::fem::Function<PetscScalar>::interpolate),
           py::arg("u"), "Interpolate a finite element function")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<PetscScalar>& self, std::uintptr_t addr) {
            const std::function<void(PetscScalar*, int, int, const double*)> f
                = reinterpret_cast<void (*)(PetscScalar*, int, int,
                                            const double*)>(addr);
            auto _f
                = [&f](Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>>
                           values,
                       const Eigen::Ref<const Eigen::Array<
                           double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x) {
                    f(values.data(), values.rows(), values.cols(), x.data());
                  };
            dolfinx::fem::interpolate_c<PetscScalar>(self, _f);
          },
          "Interpolate using a pointer to an expression with a C signature")
      .def_property_readonly(
          "vector",
          [](const dolfinx::fem::Function<PetscScalar>&
                 self) { return self.vector(); },
          "Return the vector associated with the finite element Function")
      .def_property_readonly(
          "x", py::overload_cast<>(&dolfinx::fem::Function<PetscScalar>::x),
          "Return the vector associated with the finite element Function")
      .def("eval", &dolfinx::fem::Function<PetscScalar>::eval, py::arg("x"),
           py::arg("cells"), py::arg("values"), "Evaluate Function")
      .def("compute_point_values",
           &dolfinx::fem::Function<PetscScalar>::compute_point_values,
           "Compute values at all mesh points")
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::Function<PetscScalar>::function_space);

  // dolfinx::fem::FunctionSpace
  py::class_<dolfinx::fem::FunctionSpace,
             std::shared_ptr<dolfinx::fem::FunctionSpace>>(m, "FunctionSpace",
                                                           py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>())
      .def_property_readonly("id", &dolfinx::fem::FunctionSpace::id)
      .def("__hash__", &dolfinx::fem::FunctionSpace::id)
      .def("__eq__", &dolfinx::fem::FunctionSpace::operator==)
      .def_property_readonly("dim", &dolfinx::fem::FunctionSpace::dim)
      .def("collapse", &dolfinx::fem::FunctionSpace::collapse)
      .def("component", &dolfinx::fem::FunctionSpace::component)
      .def("contains", &dolfinx::fem::FunctionSpace::contains)
      .def_property_readonly("element", &dolfinx::fem::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfinx::fem::FunctionSpace::mesh)
      .def_property_readonly("dofmap", &dolfinx::fem::FunctionSpace::dofmap)
      .def("sub", &dolfinx::fem::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           &dolfinx::fem::FunctionSpace::tabulate_dof_coordinates);

  // dolfinx::fem::Constant
  py::class_<dolfinx::fem::Constant<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Constant<PetscScalar>>>(
      m, "Constant", "A value constant with respect to integration domain")
      .def(py::init<std::vector<int>, std::vector<PetscScalar>>(),
           "Create a constant from a scalar value array")
      .def(
          "value",
          [](dolfinx::fem::Constant<PetscScalar>& self) {
            return py::array(self.shape, self.value.data(), py::none());
          },
          py::return_value_policy::reference_internal);

  // dolfinx::fem::Expression
  py::class_<dolfinx::fem::Expression<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Expression<PetscScalar>>>(
      m, "Expression", "An Expression")
      .def(py::init(
               [](const std::vector<std::shared_ptr<
                      const dolfinx::fem::Function<PetscScalar>>>& coefficients,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Constant<PetscScalar>>>& constants,
                  const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
                  const Eigen::Ref<const Eigen::Array<
                      double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&
                      x,
                  py::object addr, const std::size_t value_size) {
                 auto tabulate_expression_ptr = (void (*)(
                     PetscScalar*, const PetscScalar*, const PetscScalar*,
                     const double*))addr.cast<std::uintptr_t>();
                 return dolfinx::fem::Expression<PetscScalar>(
                     coefficients, constants, mesh, x, tabulate_expression_ptr,
                     value_size);
               }),
           py::arg("coefficients"), py::arg("constants"), py::arg("mesh"),
           py::arg("x"), py::arg("fn"), py::arg("value_size"))
      .def("eval", &dolfinx::fem::Expression<PetscScalar>::eval)
      .def_property_readonly("mesh",
                             &dolfinx::fem::Expression<PetscScalar>::mesh,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("num_points",
                             &dolfinx::fem::Expression<PetscScalar>::num_points,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("value_size",
                             &dolfinx::fem::Expression<PetscScalar>::value_size,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("x", &dolfinx::fem::Expression<PetscScalar>::x,
                             py::return_value_policy::reference_internal);
}
} // namespace dolfinx_wrappers
