// Copyright (C) 2017-2025 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include "pycoeff.h"
#include <concepts>
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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscis.h>

namespace dolfinx_wrappers
{

namespace nb = nanobind;

// Declare assembler function that have multiple scalar types
template <typename T, std::floating_point U>
void declare_petsc_discrete_operators(nb::module_& m)
{
  m.def(
      "discrete_curl",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        MPI_Comm comm = mesh->comm();
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        auto c = std::ranges::views::iota(0, map->size_local());
        dolfinx::fem::sparsitybuild::cells(sp, std::pair{c, c},
                                           {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::discrete_curl<U, T>(
            V0, V1, dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        return A;
      },
      nb::rv_policy::take_ownership, nb::arg("V0"), nb::arg("V1"));

  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        MPI_Comm comm = mesh->comm();
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        auto c = std::ranges::views::iota(0, map->size_local());
        dolfinx::fem::sparsitybuild::cells(sp, std::pair{c, c},
                                           {*dofmap1, *dofmap0});
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

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        MPI_Comm comm = mesh->comm();
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        auto c = std::ranges::views::iota(0, map->size_local());
        dolfinx::fem::sparsitybuild::cells(sp, std::pair{c, c},
                                           {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::interpolation_matrix<T, U>(
            V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES));
        return A;
      },
      nb::rv_policy::take_ownership, nb::arg("V0"), nb::arg("V1"));
}

} // namespace dolfinx_wrappers

#endif
