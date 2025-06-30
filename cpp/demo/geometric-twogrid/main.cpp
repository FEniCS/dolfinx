// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

/// @cond TODO

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>

#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <mpi.h>

#include <basix/finite-element.h>

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/transfer/transfer_matrix.h>

#include "poisson.h"

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

struct PetscEnv
{
  PetscEnv(int argc, char** argv) { PetscInitialize(&argc, &argv, NULL, NULL); }

  ~PetscEnv() { PetscFinalize(); }
};

// recommended to run with
// ./demo_geometric-multigrid -pc_mg_log -all_ksp_monitor -ksp_converged_reason
// -ksp_view
int main(int argc, char** argv)
{
  PetscEnv petscEnv(argc, argv);
  // PetscLogDefaultBegin();

  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::tetrahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);

  auto mesh_coarse = std::make_shared<mesh::Mesh<U>>(dolfinx::mesh::create_box(
      MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, {10, 10, 10},
      mesh::CellType::tetrahedron, part));

  auto V_coarse = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace<U>(mesh_coarse, element, {}));

  // refinement routine requires edges to be initialized
  mesh_coarse->topology()->create_entities(1);
  auto [mesh, parent_cells, parent_facets]
      = dolfinx::refinement::refine(*mesh_coarse, std::nullopt);

  auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
      std::make_shared<mesh::Mesh<U>>(mesh), element, {}));

  auto f_ana = [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
  {
    std::vector<T> f;
    for (std::size_t p = 0; p < x.extent(1); ++p)
    {
      auto x0 = x(0, p);
      auto x1 = x(1, p);
      auto x2 = x(2, p);
      auto pi = std::numbers::pi;
      f.push_back(2 * pi * pi * std::sin(pi * x0) * std::sin(pi * x1)
                  * std::sin(pi * x2));
    }
    return {f, {f.size()}};
  };
  auto f = std::make_shared<fem::Function<T>>(V);
  f->interpolate(f_ana);

  {
    io::VTKFile file(MPI_COMM_WORLD, "f.pvd", "w");
    file.write<T>({*f}, 0.0);
  }

  auto a = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_poisson_a, {V, V}, {}, {}, {}));
  auto a_coarse = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_poisson_a, {V_coarse, V_coarse}, {}, {}, {}));
  auto L = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
  la::petsc::Matrix A(fem::petsc::create_matrix(*a), true);
  la::petsc::Matrix A_coarse(fem::petsc::create_matrix(*a_coarse), true);

  la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                  L->function_spaces()[0]->dofmap()->index_map_bs());

  // TOOD: this somehow breaks?!?
  // V->mesh()->topology_mutable()->create_connectivity(2, 3);
  // mesh::exterior_facet_indices(*V->mesh()->topology())
  auto bc = std::make_shared<const fem::DirichletBC<T>>(
      0.0,
      mesh::locate_entities_boundary(
          *V->mesh(), 0,
          [](auto x) { return std::vector<std::int8_t>(x.extent(1), true); }),
      V);

  // V_coarse->mesh()->topology_mutable()->create_connectivity(2, 3);
  auto bc_coarse = std::make_shared<const fem::DirichletBC<T>>(
      0.0,
      mesh::locate_entities_boundary(
          *V_coarse->mesh(), 0,
          [](auto x) { return std::vector<std::int8_t>(x.extent(1), true); }),
      V_coarse);

  // assemble A
  MatZeroEntries(A.mat());
  fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES), *a,
                       {bc});
  MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                       {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

  // assemble b
  b.set(0.0);
  fem::assemble_vector(b.mutable_array(), *L);
  fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
  b.scatter_rev(std::plus<T>());
  fem::set_bc<T, U>(b.mutable_array(), {bc});

  // assemble A_coarse
  MatZeroEntries(A_coarse.mat());
  fem::assemble_matrix(
      la::petsc::Matrix::set_block_fn(A_coarse.mat(), ADD_VALUES), *a_coarse,
      {bc_coarse});
  MatAssemblyBegin(A_coarse.mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A_coarse.mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A_coarse.mat(), INSERT_VALUES),
                       *V_coarse, {bc_coarse});
  MatAssemblyBegin(A_coarse.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A_coarse.mat(), MAT_FINAL_ASSEMBLY);

  KSP ksp;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, "cg");

  PC pc;
  KSPGetPC(ksp, &pc);
  KSPSetFromOptions(ksp);
  PCSetType(pc, "mg");

  PCMGSetLevels(pc, 2, NULL);
  PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
  PCMGSetCycleType(pc, PC_MG_CYCLE_V);

  // do not rely on coarse grid operators to be generated by
  // restriction/prolongation
  PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE);
  PCMGSetOperators(pc, 0, A_coarse.mat(), A_coarse.mat());

  // PCMGSetNumberSmooth(pc, 1);
  PCSetFromOptions(pc);

  mesh.topology_mutable()->create_connectivity(0, 1);
  mesh.topology_mutable()->create_connectivity(1, 0);
  auto inclusion_map = transfer::inclusion_mapping(*mesh_coarse, mesh);
  la::SparsityPattern sp
      = transfer::create_sparsity_pattern(*V_coarse, *V, inclusion_map);
  la::petsc::Matrix restriction(MPI_COMM_WORLD, sp);
  transfer::assemble_transfer_matrix<double>(
      la::petsc::Matrix::set_block_fn(restriction.mat(), INSERT_VALUES),
      *V_coarse, *V, inclusion_map,
      [](std::int32_t d) -> double { return d == 0 ? 1. : .5; });

  MatAssemblyBegin(restriction.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(restriction.mat(), MAT_FINAL_ASSEMBLY);

  // PETSc figures out to use transpose by dimensions!
  PCMGSetInterpolation(pc, 1, restriction.mat());
  PCMGSetRestriction(pc, 1, restriction.mat());

  // MatView(A.mat(), PETSC_VIEWER_STDOUT_SELF);
  KSPSetOperators(ksp, A.mat(), A.mat());
  KSPSetUp(ksp);

  auto u = std::make_shared<fem::Function<T>>(V);

  la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
  la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);

  KSPSolve(ksp, _b.vec(), _u.vec());
  u->x()->scatter_fwd();

  {
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u}, 0.0);

    io::XDMFFile file_xdmf(mesh.comm(), "u_xdmf.xdmf", "w");
    file_xdmf.write_mesh(mesh);
    file_xdmf.write_function(*u, 0.0);
    file_xdmf.close();

#ifdef HAS_ADIOS2
    io::VTXWriter<U> vtx_writer(mesh.comm(), std::filesystem::path("u_vtx.bp"),
                                {u}, io::VTXMeshPolicy::reuse);
    vtx_writer.write(0);
#endif
  }

  KSPDestroy(&ksp);

  // PetscLogView(PETSC_VIEWER_STDOUT_SELF);
}

/// @endcond