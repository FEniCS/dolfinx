#include "hyperelasticity.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/Vector.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

// Next:
//
// .. code-block:: cpp

class HyperElasticProblem
{
public:
  HyperElasticProblem(
      std::shared_ptr<fem::Form<PetscScalar>> L,
      std::shared_ptr<fem::Form<PetscScalar>> J,
      std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>> bcs)
      : _l(L), _j(J), _bcs(bcs),
        _b(L->function_spaces()[0]->dofmap()->index_map,
           L->function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(la::PETScMatrix(fem::create_matrix(*J, "baij"), false))
  {
    auto map = L->function_spaces()[0]->dofmap()->index_map;
    const int bs = L->function_spaces()[0]->dofmap()->index_map_bs();
    std::int32_t size_local = bs * map->size_local();

    std::vector<PetscInt> ghosts(map->ghosts().begin(), map->ghosts().end());
    std::int64_t size_global = bs * map->size_global();
    VecCreateGhostBlockWithArray(map->comm(), bs, size_local, size_global,
                                 ghosts.size(), ghosts.data(),
                                 _b.array().data(), &_b_petsc);
  }

  /// Destructor
  virtual ~HyperElasticProblem()
  {
    if (_b_petsc)
      VecDestroy(&_b_petsc);
  }

  auto form()
  {
    return [](Vec x)
    {
      VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
    };
  }

  /// Compute F at current point x
  auto F()
  {
    return [&](const Vec x, Vec)
    {
      // Assemble b and update ghosts
      xtl::span<PetscScalar> b(_b.mutable_array());
      std::fill(b.begin(), b.end(), 0.0);
      fem::assemble_vector<PetscScalar>(b, *_l);
      VecGhostUpdateBegin(_b_petsc, ADD_VALUES, SCATTER_REVERSE);
      VecGhostUpdateEnd(_b_petsc, ADD_VALUES, SCATTER_REVERSE);

      // Set bcs
      Vec x_local;
      VecGhostGetLocalForm(x, &x_local);
      PetscInt n = 0;
      VecGetSize(x_local, &n);
      const PetscScalar* array = nullptr;
      VecGetArrayRead(x_local, &array);
      fem::set_bc<PetscScalar>(b, _bcs, xtl::span<const PetscScalar>(array, n),
                               -1.0);
      VecRestoreArrayRead(x, &array);
    };
  }

  /// Compute J = F' at current point x
  auto J()
  {
    return [&](const Vec, Mat A)
    {
      MatZeroEntries(A);
      fem::assemble_matrix(la::PETScMatrix::set_block_fn(A, ADD_VALUES), *_j,
                           _bcs);
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
      fem::set_diagonal(la::PETScMatrix::set_fn(A, INSERT_VALUES),
                        *_j->function_spaces()[0], _bcs);
      MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    };
  }

  Vec vector() { return _b_petsc; }

  Mat matrix() { return _matA.mat(); }

private:
  std::shared_ptr<fem::Form<PetscScalar>> _l, _j;
  std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>> _bcs;
  la::Vector<PetscScalar> _b;
  Vec _b_petsc = nullptr;
  la::PETScMatrix _matA;
};

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string thread_name = "RANK " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());

  {
    // Inside the ``main`` function, we begin by defining a tetrahedral mesh
    // of the domain and the function space on this mesh. Here, we choose to
    // create a unit cube mesh with 25 ( = 24 + 1) vertices in one direction
    // and 17 ( = 16 + 1) vertices in the other two directions. With this
    // mesh, we initialize the (finite element) function space defined by the
    // generated code.
    //
    // .. code-block:: cpp

    // Create mesh and define function space
    auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
        mesh::CellType::tetrahedron, mesh::GhostMode::none));

    auto V = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
        functionspace_form_hyperelasticity_F, "u", mesh));

    // Define solution function
    auto u = std::make_shared<fem::Function<PetscScalar>>(V);
    auto a = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_hyperelasticity_J, {V, V},
                                      {{"u", u}}, {}, {}));
    auto L = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_hyperelasticity_F, {V}, {{"u", u}},
                                      {}, {}));

    auto u_rotation = std::make_shared<fem::Function<PetscScalar>>(V);
    u_rotation->interpolate(
        [](auto& x)
        {
          constexpr double scale = 0.005;

          // Center of rotation
          constexpr double x1_c = 0.5;
          constexpr double x2_c = 0.5;

          // Large angle of rotation (60 degrees)
          constexpr double theta = 1.04719755;
          xt::xarray<double> values = xt::zeros_like(x);

          // New coordinates
          auto x1 = xt::row(x, 1);
          auto x2 = xt::row(x, 2);
          xt::row(values, 1) = scale
                               * (x1_c + (x1 - x1_c) * std::cos(theta)
                                  - (x2 - x2_c) * std::sin(theta) - x1);
          xt::row(values, 2) = scale
                               * (x2_c + (x1 - x1_c) * std::sin(theta)
                                  - (x2 - x2_c) * std::cos(theta) - x2);
          return values;
        });

    auto u_clamp = std::make_shared<fem::Function<PetscScalar>>(V);
    u_clamp->interpolate([](auto& x) -> xt::xarray<double>
                         { return xt::zeros_like(x); });

    // Create Dirichlet boundary conditions
    auto u0 = std::make_shared<fem::Function<PetscScalar>>(V);

    const auto bdofs_left
        = fem::locate_dofs_geometrical({*V},
                                       [](auto& x) -> xt::xtensor<bool, 1> {
                                         return xt::isclose(xt::row(x, 0), 0.0);
                                       });
    const auto bdofs_right
        = fem::locate_dofs_geometrical({*V},
                                       [](auto& x) -> xt::xtensor<bool, 1> {
                                         return xt::isclose(xt::row(x, 0), 1.0);
                                       });

    auto bcs
        = std::vector({std::make_shared<const fem::DirichletBC<PetscScalar>>(
                           u_clamp, std::move(bdofs_left)),
                       std::make_shared<const fem::DirichletBC<PetscScalar>>(
                           u_rotation, std::move(bdofs_right))});

    HyperElasticProblem problem(L, a, bcs);
    nls::NewtonSolver newton_solver(MPI_COMM_WORLD);
    newton_solver.setF(problem.F(), problem.vector());
    newton_solver.setJ(problem.J(), problem.matrix());
    newton_solver.set_form(problem.form());
    newton_solver.solve(u->vector());

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({*u}, 0.0);
  }

  common::subsystem::finalize_petsc();

  return 0;
}
