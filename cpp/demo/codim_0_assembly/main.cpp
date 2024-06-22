// # Mixed assembly with a function mesh on a subset of cells
//
// This demo illustrates how to:
//
// * Create a submesh of co-dimension 0
// * Assemble a mixed formulation with function spaces defined on the sub mesh
// and parent mesh

#include "mixed_codim0.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto domain = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {1, 4}, mesh::CellType::quadrilateral, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::quadrilateral, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(domain, element, {}));

    // Next we find all cells of the mesh with y<0.5
    const int tdim = domain->topology()->dim();
    auto cells = mesh::locate_entities(
        *domain, tdim,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto y = x(1, p);
            if (std::abs(y) <= 0.5 + eps)
              marker[p] = true;
          }
          return marker;
        });

    // With these cells, we can define a submesh
    auto [submesh, sub_to_parent_cells, v_map, g_map]
        = mesh::create_submesh(*domain, tdim, std::span<const int>(cells));

    // We create the function space used for the trial space
    auto V_sub
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            std::make_shared<mesh::Mesh<U>>(submesh), element, {}));

    // We need to invert the map from submesh cells to parent cells
    std::size_t num_cells_local
        = domain->topology()->index_map(tdim)->size_local()
          + domain->topology()->index_map(tdim)->num_ghosts();
    std::vector<std::int32_t> parent_to_sub(num_cells_local, -1);
    for (std::size_t i = 0; i < sub_to_parent_cells.size(); ++i)
      parent_to_sub[sub_to_parent_cells[i]] = i;

    std::map<std::shared_ptr<const mesh::Mesh<U>>,
             std::span<const std::int32_t>>
        entity_maps
        = {{V_sub->mesh(), std::span<const std::int32_t>(
                               parent_to_sub.data(), parent_to_sub.size())}};
    // Now we need to make a restriction of the cell integrals to those cells
    // that exist in the submesh

    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_map = {};
    subdomain_map[fem::IntegralType::cell].push_back(
        {3, std::span<const std::int32_t>(sub_to_parent_cells)});

    // We can now create the bi-linear form
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_mixed_codim0_a, {V, V_sub}, {}, {},
                            subdomain_map, entity_maps, V->mesh()));

    la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
    sp.finalize();
    la::MatrixCSR<double> A(sp);
    fem::assemble_matrix(A.mat_add_values(), *a, {});
    A.scatter_rev();

    auto a_sub = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_mixed_codim0_a_sub, {V_sub, V_sub}, {}, {}, {}, {}));
    la::SparsityPattern sp_sub = fem::create_sparsity_pattern(*a);
    sp_sub.finalize();

    la::MatrixCSR<double> A_sub(sp_sub);
    fem::assemble_matrix(A_sub.mat_add_values(), *a_sub, {});
    A_sub.scatter_rev();

    std::vector<T> A_flattened = A.to_dense();
    std::stringstream cc;
    cc.precision(3);
    cc << "A:" << std::endl;

    std::size_t num_owned_rows = V->dofmap()->index_map->size_local();
    std::size_t num_sub_cols = V_sub->dofmap()->index_map->size_local()
                               + V_sub->dofmap()->index_map->num_ghosts();
    for (std::size_t i = 0; i < num_owned_rows; i++)
    {
      for (std::size_t j = 0; j < num_sub_cols; j++)
      {
        cc << A_flattened[i * num_sub_cols + j] << " ";
      }
      cc << std::endl;
    }

    std::size_t num_owned_sub_rows = V_sub->dofmap()->index_map->size_local();
    std::vector<T> A_sub_flattened = A_sub.to_dense();
    cc << "A_sub" << std::endl;
    for (std::size_t i = 0; i < num_owned_sub_rows; i++)
    {
      for (std::size_t j = 0; j < num_sub_cols; j++)
      {
        cc << A_sub_flattened[i * num_sub_cols + j] << " ";
      }
      cc << std::endl;
    }
    std::cout << cc.str() << std::endl;
  }

  PetscFinalize();

  return 0;
}
