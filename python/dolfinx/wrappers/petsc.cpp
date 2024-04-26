// Copyright (C) 2017-2023 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include "pycoeff.h"
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
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/nls/NewtonSolver.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscis.h>

namespace
{
// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_petsc_discrete_operators(nb::module_& m)
{
  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, {c, c}, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::discrete_gradient<T, U>(
            *V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
            {*V1.element(), *V1.dofmap()},
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        return A;
      },
      nb::rv_policy::take_ownership, nb::arg("V0"), nb::arg("V1"));
  m.def(
      "interpolation_matrix",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, {c, c}, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::interpolation_matrix<T, U>(
            V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A, INSERT_VALUES));
        return A;
      },
      nb::rv_policy::take_ownership, nb::arg("V0"), nb::arg("V1"));
}

void petsc_la_module(nb::module_& m)
{
  import_petsc4py();

  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p, const std::string& type)
      {
        Mat A = dolfinx::la::petsc::create_matrix(comm.get(), p, type);
        PyObject* obj = PyPetscMat_New(A);
        PetscObjectDereference((PetscObject)A);
        return nb::borrow(obj);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");

  m.def(
      "create_index_sets",
      [](const std::vector<std::pair<const common::IndexMap*, int>>& maps)
      {
        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            _maps;
        for (auto m : maps)
          _maps.push_back({*m.first, m.second});
        std::vector<IS> index_sets
            = dolfinx::la::petsc::create_index_sets(_maps);

        std::vector<nb::object> py_index_sets;
        for (auto is : index_sets)
        {
          PyObject* obj = PyPetscIS_New(is);
          PetscObjectDereference((PetscObject)is);
          py_index_sets.push_back(nb::steal(obj));
        }
        return py_index_sets;
      },
      nb::arg("maps"));

  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<
             nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig>>& x_b,
         const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        std::vector<std::span<const PetscScalar>> _x_b;
        std::vector<std::pair<
            std::reference_wrapper<const dolfinx::common::IndexMap>, int>>
            _maps;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        for (auto q : maps)
          _maps.push_back({*q.first, q.second});

        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, _maps);
      },
      nb::arg("x"), nb::arg("x_b"), nb::arg("maps"),
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");

  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        std::vector<std::pair<
            std::reference_wrapper<const dolfinx::common::IndexMap>, int>>
            _maps;
        for (auto m : maps)
          _maps.push_back({*m.first, m.second});
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::petsc::get_local_vectors(x, _maps);
        std::vector<nb::ndarray<PetscScalar, nb::numpy>> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(dolfinx_wrappers::as_nbarray(std::move(v)));
        return ret;
      },
      nb::arg("x"), nb::arg("maps"),
      "Gather an (ordered) list of sub vectors from a block vector.");
}

void petsc_fem_module(nb::module_& m)
{
  // Create PETSc vectors and matrices
  m.def(
      "create_vector_block",
      [](const std::vector<
          std::pair<std::shared_ptr<const common::IndexMap>, int>>& maps)
      {
        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            _maps;
        for (auto q : maps)
          _maps.push_back({*q.first, q.second});

        return dolfinx::fem::petsc::create_vector_block(_maps);
      },
      nb::rv_policy::take_ownership, nb::arg("maps"),
      "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def(
      "create_vector_nest",
      [](const std::vector<
          std::pair<std::shared_ptr<const common::IndexMap>, int>>& maps)
      {
        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            _maps;
        for (auto m : maps)
          _maps.push_back({*m.first, m.second});
        return dolfinx::fem::petsc::create_vector_nest(_maps);
      },
      nb::rv_policy::take_ownership, nb::arg("maps"),
      "Create nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix", dolfinx::fem::petsc::create_matrix<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"),
        nb::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        &dolfinx::fem::petsc::create_matrix_block<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"),
        nb::arg("type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        &dolfinx::fem::petsc::create_matrix_nest<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"),
        nb::arg("types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const PetscScalar, nb::ndim<2>,
                                    nb::c_contig>>& coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, PetscReal>>>& bcs,
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
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("bcs"), nb::arg("unrolled") = false,
      "Assemble bilinear form into an existing PETSc matrix");
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const PetscScalar, nb::ndim<2>,
                                    nb::c_contig>>& coefficients,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> rows0,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> rows1,
         bool unrolled)
      {
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          std::span<const PetscScalar>)>
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
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("rows0"), nb::arg("rows1"), nb::arg("unrolled") = false);
  m.def(
      "insert_diagonal",
      [](Mat A, const dolfinx::fem::FunctionSpace<PetscReal>& V,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, PetscReal>>>& bcs,
         PetscScalar diagonal)
      {
        dolfinx::fem::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, bcs,
            diagonal);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("bcs"), nb::arg("diagonal"));

  declare_petsc_discrete_operators<PetscScalar, PetscReal>(m);
}

void petsc_nls_module(nb::module_& m)
{
  // dolfinx::NewtonSolver
  nb::class_<dolfinx::nls::petsc::NewtonSolver>(m, "NewtonSolver")
      .def(
          "__init__",
          [](dolfinx::nls::petsc::NewtonSolver* ns,
             const dolfinx_wrappers::MPICommWrapper comm)
          { new (ns) dolfinx::nls::petsc::NewtonSolver(comm.get()); },
          nb::arg("comm"))
      .def_prop_ro("krylov_solver",
                   [](const dolfinx::nls::petsc::NewtonSolver& self)
                   {
                     KSP ksp = self.get_krylov_solver().ksp();
                     PyObject* obj = PyPetscKSP_New(ksp);
                     PetscObjectDereference((PetscObject)ksp);
                     return nb::borrow(obj);
                   })
      .def("setF", &dolfinx::nls::petsc::NewtonSolver::setF, nb::arg("F"),
           nb::arg("b"))
      .def("setJ", &dolfinx::nls::petsc::NewtonSolver::setJ, nb::arg("J"),
           nb::arg("Jmat"))
      .def("setP", &dolfinx::nls::petsc::NewtonSolver::setP, nb::arg("P"),
           nb::arg("Pmat"))
      .def(
          "set_update",
          [](dolfinx::nls::petsc::NewtonSolver& self,
             std::function<void(const dolfinx::nls::petsc::NewtonSolver* solver,
                                const Vec, Vec)>
                 update)
          {
            // See https://github.com/wjakob/nanobind/discussions/361 on below
            self.set_update(
                [update](const dolfinx::nls::petsc::NewtonSolver& solver,
                         const Vec dx, Vec x) { update(&solver, dx, x); });
          },
          nb::arg("update"))
      .def("set_form", &dolfinx::nls::petsc::NewtonSolver::set_form,
           nb::arg("form"))
      .def("solve", &dolfinx::nls::petsc::NewtonSolver::solve, nb::arg("x"))
      .def_rw("atol", &dolfinx::nls::petsc::NewtonSolver::atol,
              "Absolute tolerance")
      .def_rw("rtol", &dolfinx::nls::petsc::NewtonSolver::rtol,
              "Relative tolerance")
      .def_rw("error_on_nonconvergence",
              &dolfinx::nls::petsc::NewtonSolver::error_on_nonconvergence)
      .def_rw("report", &dolfinx::nls::petsc::NewtonSolver::report)
      .def_rw("relaxation_parameter",
              &dolfinx::nls::petsc::NewtonSolver::relaxation_parameter,
              "Relaxation parameter")
      .def_rw("max_it", &dolfinx::nls::petsc::NewtonSolver::max_it,
              "Maximum number of iterations")
      .def_rw("convergence_criterion",
              &dolfinx::nls::petsc::NewtonSolver::convergence_criterion,
              "Convergence criterion, either 'residual' (default) or "
              "'incremental'");
}

} // namespace

namespace dolfinx_wrappers
{
void petsc(nb::module_& m_fem, nb::module_& m_la, nb::module_& m_nls)
{
  nb::module_ petsc_fem_mod
      = m_fem.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_fem_module(petsc_fem_mod);

  nb::module_ petsc_la_mod
      = m_la.def_submodule("petsc", "PETSc-specific linear algebra module");
  petsc_la_module(petsc_la_mod);

  nb::module_ petsc_nls_mod
      = m_nls.def_submodule("petsc", "PETSc-specific nonlinear solvers");
  petsc_nls_module(petsc_nls_mod);
}
} // namespace dolfinx_wrappers
#endif
