// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_petsc.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <span>
#include <string>
#include <utility>

namespace py = pybind11;

namespace
{

template <typename T>
std::map<std::pair<dolfinx::fem::IntegralType, int>,
         std::pair<std::span<const T>, int>>
py_to_cpp_coeffs(const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                py::array_t<T, py::array::c_style>>& coeffs)
{
  using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key_t, std::pair<std::span<const T>, int>> c;
  std::transform(coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
                 [](auto& e) -> typename decltype(c)::value_type
                 {
                   return {e.first,
                           {std::span(e.second.data(), e.second.size()),
                            e.second.shape(1)}};
                 });
  return c;
}

// Declare assembler function that have multiple scalar types
template <typename T>
void declare_assembly_functions(py::module& m)
{
  // Coefficient/constant packing
  m.def(
      "pack_coefficients",
      [](const dolfinx::fem::Form<T, double>& form)
      {
        using Key_t = typename std::pair<dolfinx::fem::IntegralType, int>;

        // Pack coefficients
        std::map<Key_t, std::pair<std::vector<T>, int>> coeffs
            = dolfinx::fem::allocate_coefficient_storage(form);
        dolfinx::fem::pack_coefficients(form, coeffs);

        // Move into NumPy data structures
        std::map<Key_t, py::array_t<T, py::array::c_style>> c;
        std::transform(
            coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              int num_ents = e.second.first.empty()
                                 ? 0
                                 : e.second.first.size() / e.second.second;
              return {e.first, dolfinx_wrappers::as_pyarray(
                                   std::move(e.second.first),
                                   std::array{num_ents, e.second.second})};
            });

        return c;
      },
      py::arg("form"), "Pack coefficients for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<T, double>& form) {
        return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(form));
      },
      py::arg("form"), "Pack constants for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Expression<T, double>& e)
      { return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(e)); },
      py::arg("e"), "Pack constants for an Expression.");

  // Functional
  m.def(
      "assemble_scalar",
      [](const dolfinx::fem::Form<T, double>& M,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        return dolfinx::fem::assemble_scalar<T>(
            M, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      py::arg("M"), py::arg("constants"), py::arg("coefficients"),
      "Assemble functional over mesh with provided constants and "
      "coefficients");
  // Vector
  m.def(
      "assemble_vector",
      [](py::array_t<T, py::array::c_style> b,
         const dolfinx::fem::Form<T, double>& L,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        dolfinx::fem::assemble_vector<T>(
            std::span(b.mutable_data(), b.size()), L,
            std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      py::arg("b"), py::arg("L"), py::arg("constants"), py::arg("coeffs"),
      "Assemble linear form into an existing vector with pre-packed constants "
      "and coefficients");
  // MatrixCSR
  m.def(
      "assemble_matrix",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::Form<T, double>& a,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, double>>>& bcs)
      {
        if (a.function_spaces()[0]->dofmap()->bs() != 1
            or a.function_spaces()[0]->dofmap()->bs() != 1)
        {
          throw std::runtime_error("Assembly with block size > 1 not yet "
                                   "supported with la::MatrixCSR.");
        }
        dolfinx::fem::assemble_matrix(
            A.mat_add_values(), a,
            std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients), bcs);
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs"), "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A,
         const dolfinx::fem::FunctionSpace<double>& V,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, double>>>& bcs,
         T diagonal)
      { dolfinx::fem::set_diagonal(A.mat_set_values(), V, bcs, diagonal); },
      py::arg("A"), py::arg("V"), py::arg("bcs"), py::arg("diagonal"),
      "Experimental.");
  m.def(
      "assemble_matrix",
      [](const std::function<int(const py::array_t<std::int32_t>&,
                                 const py::array_t<std::int32_t>&,
                                 const py::array_t<T>&)>& fin,
         const dolfinx::fem::Form<T, double>& form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, double>>>& bcs)
      {
        auto f = [&fin](const std::span<const std::int32_t>& rows,
                        const std::span<const std::int32_t>& cols,
                        const std::span<const T>& data)
        {
          return fin(py::array(rows.size(), rows.data()),
                     py::array(cols.size(), cols.data()),
                     py::array(data.size(), data.data()));
        };
        dolfinx::fem::assemble_matrix(f, form, bcs);
      },
      py::arg("fin"), py::arg("form"), py::arg("bcs"),
      "Experimental assembly with Python insertion function. This will be "
      "slow. Use for testing only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<T, double>>>& a,
         const std::vector<py::array_t<T, py::array::c_style>>& constants,
         const std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    py::array_t<T, py::array::c_style>>>&
             coeffs,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<T, double>>>>& bcs1,
         const std::vector<py::array_t<T, py::array::c_style>>& x0,
         double scale)
      {
        std::vector<std::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        std::vector<std::span<const T>> _constants;
        std::transform(constants.begin(), constants.end(),
                       std::back_inserter(_constants),
                       [](auto& c) { return std::span(c.data(), c.size()); });

        std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                             std::pair<std::span<const T>, int>>>
            _coeffs;
        std::transform(coeffs.begin(), coeffs.end(),
                       std::back_inserter(_coeffs),
                       [](auto& c) { return py_to_cpp_coeffs(c); });

        dolfinx::fem::apply_lifting<T>(std::span(b.mutable_data(), b.size()), a,
                                       _constants, _coeffs, bcs1, _x0, scale);
      },
      py::arg("b"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs1"), py::arg("x0"), py::arg("scale"),
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, double>>>& bcs,
         const py::array_t<T, py::array::c_style>& x0, double scale)
      {
        if (x0.ndim() == 0)
        {
          dolfinx::fem::set_bc<T>(std::span(b.mutable_data(), b.size()), bcs,
                                  scale);
        }
        else if (x0.ndim() == 1)
        {
          dolfinx::fem::set_bc<T>(std::span(b.mutable_data(), b.size()), bcs,
                                  std::span(x0.data(), x0.shape(0)), scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
}

void petsc_module(py::module& m)
{
  // Create PETSc vectors and matrices
  m.def("create_vector_block", &dolfinx::fem::petsc::create_vector_block,
        py::return_value_policy::take_ownership, py::arg("maps"),
        "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def("create_vector_nest", &dolfinx::fem::petsc::create_vector_nest,
        py::return_value_policy::take_ownership, py::arg("maps"),
        "Create nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix", dolfinx::fem::petsc::create_matrix<double>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        &dolfinx::fem::petsc::create_matrix_block<double>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest", &dolfinx::fem::petsc::create_matrix_nest<double>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, double>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, double>>>& bcs,
         bool unrolled)
      {
        if (unrolled)
        {
          auto set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
          dolfinx::fem::assemble_matrix(
              set_fn, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else
        {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
              std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs"), py::arg("unrolled") = false,
      "Assemble bilinear form into an existing PETSc matrix");
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, double>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const py::array_t<std::int8_t, py::array::c_style>& rows0,
         const py::array_t<std::int8_t, py::array::c_style>& rows1,
         bool unrolled)
      {
        if (rows0.ndim() != 1 or rows1.ndim())
        {
          throw std::runtime_error(
              "Expected 1D arrays for boundary condition rows/columns");
        }

        std::function<int(const std::span<const std::int32_t>&,
                          const std::span<const std::int32_t>&,
                          const std::span<const PetscScalar>&)>
            set_fn;
        if (unrolled)
        {
          set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
        }
        else
          set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);

        dolfinx::fem::assemble_matrix(
            set_fn, a, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients),
            std::span(rows0.data(), rows0.size()),
            std::span(rows1.data(), rows1.size()));
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("rows0"), py::arg("rows1"), py::arg("unrolled") = false);
  m.def(
      "insert_diagonal",
      [](Mat A, const dolfinx::fem::FunctionSpace<double>& V,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, double>>>& bcs,
         PetscScalar diagonal)
      {
        dolfinx::fem::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, bcs,
            diagonal);
      },
      py::arg("A"), py::arg("V"), py::arg("bcs"), py::arg("diagonal"));

  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<double>& V0,
         const dolfinx::fem::FunctionSpace<double>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
        assert(dofmap0);
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});
        dolfinx::fem::sparsitybuild::cells(sp, mesh->topology(),
                                           {*dofmap1, *dofmap0});
        sp.assemble();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::discrete_gradient<PetscScalar>(
            V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
            {*V1.element(), *V1.dofmap()},
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        // dolfinx::fem::discrete_gradient<PetscScalar>(
        //     V0, V1, dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership, py::arg("V0"), py::arg("V1"));
  m.def(
      "interpolation_matrix",
      [](const dolfinx::fem::FunctionSpace<double>& V0,
         const dolfinx::fem::FunctionSpace<double>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
        assert(dofmap0);
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});
        dolfinx::fem::sparsitybuild::cells(sp, mesh->topology(),
                                           {*dofmap1, *dofmap0});
        sp.assemble();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::interpolation_matrix<PetscScalar>(
            V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership, py::arg("V0"), py::arg("V1"));
}

} // namespace

namespace dolfinx_wrappers
{

void assemble(py::module& m)
{
  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_module(petsc_mod);

  // dolfinx::fem::assemble
  declare_assembly_functions<float>(m);
  declare_assembly_functions<double>(m);
  declare_assembly_functions<std::complex<float>>(m);
  declare_assembly_functions<std::complex<double>>(m);
}
} // namespace dolfinx_wrappers
