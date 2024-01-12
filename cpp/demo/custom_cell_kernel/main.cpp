// Custom cell kernel (C++)
// .. code-block:: cpp

#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

// .. code-block:: cpp

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    // Would it be possible just to define instead of (T, V, L) (L) alone?
    basix::FiniteElement e = basix::create_element<U>(
        basix::element::family::P,
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, e));

    // Create default domain integral on all local cells
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Define element kernel
    auto mass_cell_kernel = [](double* A, const double*, const double*,
                               const double* coordinate_dofs, const int*,
                               const u_int8_t*) { // Quadrature rules
      static const double weights_48e[3]
          = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};
      // Precomputed values of basis functions and precomputations
      // FE* dimensions: [permutation][entities][points][dofs]
      static const double FE0_C0_Q48e[1][1][3][3]
          = {{{{0.6666666666666667, 0.1666666666666667, 0.1666666666666666},
               {0.1666666666666667, 0.1666666666666666, 0.6666666666666666},
               {0.1666666666666668, 0.6666666666666666, 0.1666666666666666}}}};
      static const double FE1_C0_D10_Q48e[1][1][1][3] = {{{{-1.0, 1.0, 0.0}}}};
      static const double FE1_C1_D01_Q48e[1][1][1][3] = {{{{-1.0, 0.0, 1.0}}}};
      // Quadrature loop independent computations for quadrature rule 48e
      double J_c0 = 0.0;
      double J_c3 = 0.0;
      double J_c1 = 0.0;
      double J_c2 = 0.0;
      for (int ic = 0; ic < 3; ++ic)
      {
        J_c0 += coordinate_dofs[(ic) * 3] * FE1_C0_D10_Q48e[0][0][0][ic];
        J_c3 += coordinate_dofs[(ic) * 3 + 1] * FE1_C1_D01_Q48e[0][0][0][ic];
        J_c1 += coordinate_dofs[(ic) * 3] * FE1_C1_D01_Q48e[0][0][0][ic];
        J_c2 += coordinate_dofs[(ic) * 3 + 1] * FE1_C0_D10_Q48e[0][0][0][ic];
      }
      double sp_48e[4];
      sp_48e[0] = J_c0 * J_c3;
      sp_48e[1] = J_c1 * J_c2;
      sp_48e[2] = sp_48e[0] - sp_48e[1];
      sp_48e[3] = fabs(sp_48e[2]);
      for (int iq = 0; iq < 3; ++iq)
      {
        double fw0 = sp_48e[3] * weights_48e[iq];
        double t0[3];
        for (int i = 0; i < 3; ++i)
        {
          t0[i] = fw0 * FE0_C0_Q48e[0][0][iq][i];
        }
        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            A[3 * (i) + (j)] += FE0_C0_Q48e[0][0][iq][j] * t0[i];
          }
        }
      }
    };
    // More automatic type inference?
    const std::map<
        fem::IntegralType,
        std::vector<std::tuple<
            std::int32_t,
            std::function<void(double*, const double*, const double*,
                               const double*, const int*, const u_int8_t*)>,
            std::vector<std::int32_t>>>>
        integrals
        = {{fem::IntegralType::cell, {{-1, mass_cell_kernel, cells}}}};

    // Define form from integral
    // NOTE: Cannot be done through create_form which is recommended in docs.
    auto a = std::make_shared<fem::Form<T>>(
        fem::Form<T>({V, V}, integrals, {}, {}, false, mesh));

    auto sparsity = la::SparsityPattern(
        MPI_COMM_WORLD, {V->dofmap()->index_map, V->dofmap()->index_map},
        {V->dofmap()->index_map_bs(), V->dofmap()->index_map_bs()});
    fem::sparsitybuild::cells(sparsity, cells, {*V->dofmap(), *V->dofmap()});
    sparsity.finalize();
    auto A = la::MatrixCSR<double>(sparsity);

    auto mat_add_values = A.mat_add_values();
    assemble_matrix(mat_add_values, *a, {});
    A.scatter_rev();
  }

  MPI_Finalize();
  return 0;
}
