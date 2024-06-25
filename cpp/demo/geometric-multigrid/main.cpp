/// @cond

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mpi.h>
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
#include <sys/types.h>

#include <basix/finite-element.h>

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>

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
int main(int argc, char** argv)
{
  PetscEnv petscEnv(argc, argv);
  // PetscLogDefaultBegin();

  int n_coarse = 32; // also works with 1e6 !
  int n_fine = 2 * n_coarse;

  auto create_FEM_space = [](int n)
  {
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        dolfinx::mesh::create_interval<U>(MPI_COMM_WORLD, n, {0.0, 1.0}, part));
    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::interval, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace<U>(mesh, element, {}));
    return std::make_pair<decltype(mesh), decltype(V)>(std::move(mesh),
                                                       std::move(V));
  };

  auto [mesh, V] = create_FEM_space(n_fine);
  auto [mesh_coarse, V_coarse] = create_FEM_space(n_coarse);

  auto interpolate_f = [](auto V)
  {
    auto f_ana
        = [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    {
      std::vector<T> f;
      for (std::size_t p = 0; p < x.extent(1); ++p)
        f.push_back(-2 * std::numbers::pi * std::numbers::pi);
      return {f, {f.size()}};
    };
    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(f_ana);
    return f;
  };

  const auto f = interpolate_f(V);
  const auto f_coarse = interpolate_f(V_coarse);

  {
    io::VTKFile file(MPI_COMM_WORLD, "f.pvd", "w");
    file.write<T>({*f}, 0.0);
  }

  auto create_variational_problem = [](auto V, auto f)
  {
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_a, {V, V}, {}, {}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
    la::petsc::Matrix A(fem::petsc::create_matrix(*a), true);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    auto&& facets = mesh::locate_entities_boundary(
        *V->mesh(), 0,
        [](auto x) { return std::vector<std::int8_t>(x.extent(1), true); });
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 0, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    return std::make_tuple<decltype(a), decltype(L), decltype(bc), decltype(A),
                           decltype(b)>(
        std::move(a), std::move(L), std::move(bc), std::move(A), std::move(b));
  };

  auto [a, L, bc, A, b] = create_variational_problem(V, f);
  auto [a_coarse, L_coarse, bc_coarse, A_coarse, b_coarse]
      = create_variational_problem(V_coarse, f_coarse);

  {
    // assemble A
    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
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
    fem::set_diagonal<T>(
        la::petsc::Matrix::set_fn(A_coarse.mat(), INSERT_VALUES), *V_coarse,
        {bc_coarse});
    MatAssemblyBegin(A_coarse.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_coarse.mat(), MAT_FINAL_ASSEMBLY);
  }

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

  Mat interpolation;
  {
    auto local_rows = V->dofmap()->index_map->size_local();
    auto local_cols = V_coarse->dofmap()->index_map->size_local();
    auto global_rows = V->dofmap()->index_map->size_global();
    auto global_cols = V_coarse->dofmap()->index_map->size_global();
    MatCreateAIJ(MPI_COMM_WORLD, local_rows, local_cols, global_rows,
                 global_cols, 2, NULL, 1, NULL, &interpolation);

    // this shoud be quite overkill for the parallel case -> localize the value
    // setting and do not repeat!!!
    for (int64_t idx = 0; idx < n_fine + 1; idx++)
    {
      if (idx % 2 == 0)
      {
        MatSetValue(interpolation, idx, PetscInt(idx / 2), 1., INSERT_VALUES);
      }
      else
      {
        MatSetValue(interpolation, idx, floor(idx / 2.0), .5, INSERT_VALUES);
        MatSetValue(interpolation, idx, ceil(idx / 2.0), .5, INSERT_VALUES);
      }
    }
    MatAssemblyBegin(interpolation, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(interpolation, MAT_FINAL_ASSEMBLY);
  }

  // PETSc figures out to use transpose by dimensions!
  PCMGSetInterpolation(pc, 1, interpolation);
  PCMGSetRestriction(pc, 1, interpolation);

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

    io::XDMFFile file_xdmf(mesh->comm(), "u_xdmf.xdmf", "w");
    file_xdmf.write_mesh(*mesh);
    file_xdmf.write_function(*u, 0.0);
    file_xdmf.close();

#ifdef HAS_ADIOS2
    io::VTXWriter<U> vtx_writer(mesh->comm(), std::filesystem::path("u_vtx.bp"),
                                {u}, io::VTXMeshPolicy::reuse);
    vtx_writer.write(0);
#endif
  }

  auto A_mat = A.mat();
  MatDestroy(&A_mat);

  auto A_coarse_mat = A_coarse.mat();
  MatDestroy(&A_coarse_mat);

  MatDestroy(&interpolation);
  KSPDestroy(&ksp);

  // PetscLogView(PETSC_VIEWER_STDOUT_SELF);
}

/// @endcond