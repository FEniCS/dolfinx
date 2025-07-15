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
#include <dolfinx/mesh/EntityMap.h>
#include <map>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {1, 4},
        mesh::CellType::quadrilateral,
        mesh::create_cell_partitioner(mesh::GhostMode::shared_facet)));

    basix::FiniteElement element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::quadrilateral, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));

    // Next we find all cells of the mesh with y<0.5
    const int tdim = mesh->topology()->dim();
    std::vector<std::int32_t> marked_cells = mesh::locate_entities(
        *mesh, tdim,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            if (std::abs(x(1, p)) <= 0.5 + eps)
              marker[p] = true;
          }
          return marker;
        });

    // We create a MeshTags object where we mark these cells with 2, and
    // any other cell with 1
    auto cell_map = mesh->topology()->index_map(tdim);
    assert(cell_map);
    std::size_t num_cells_local
        = mesh->topology()->index_map(tdim)->size_local()
          + mesh->topology()->index_map(tdim)->num_ghosts();
    std::vector<std::int32_t> cells(num_cells_local);
    std::iota(cells.begin(), cells.end(), 0);
    std::vector<std::int32_t> values(cells.size(), 1);
    std::ranges::for_each(marked_cells, [&values](auto c) { values[c] = 2; });
    mesh::MeshTags<std::int32_t> cell_marker(mesh->topology(), tdim, cells,
                                             values);

    // We create a submesh consisting of only cells with a given tag by
    // calling `create_submesh`. This function also returns an
    // `EntityMap` object, which relates entities in the submesh to
    // entities in the original mesh. We will need this to assemble our
    // mixed-domain form.
    auto submesh_data = [](auto& mesh, int tdim, auto&& subcells)
    {
      auto [submesh, emap, v_map, g_map]
          = mesh::create_submesh(mesh, tdim, subcells);
      return std::pair(std::make_shared<mesh::Mesh<U>>(std::move(submesh)),
                       std::make_shared<mesh::EntityMap>(std::move(emap)));
    };
    auto [submesh, entity_map] = submesh_data(*mesh, tdim, cell_marker.find(2));

    // We create the function space used for the trial space
    auto W
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            submesh, std::make_shared<fem::FiniteElement<U>>(element)));

    // Next we compute the integration entities on the integration
    // domain `mesh`
    std::vector<std::int32_t> integration_entities
        = fem::compute_integration_domains(
            fem::IntegralType::cell, *mesh->topology(), cell_marker.find(2));
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_data
        = {{fem::IntegralType::cell, {{3, integration_entities}}}};

    // A mixed-domain form involves functions defined over multiple
    // meshes. The mesh passed to `create_form` is called the
    // *integration domain mesh*. To assemble a mixed-domain form, we
    // must supply an `EntityMap` for each additional mesh involved in
    // the form, relating entities in that mesh to the integration
    // domain mesh. In our case, `mesh` is the integration domain mesh,
    // and the only other mesh in our form is `submesh`. Hence, we must
    // provide the entity map object returned when we called
    // `create_submesh`, which relates entities in `submesh` to entities
    // in `mesh`.
    std::vector<std::shared_ptr<const mesh::EntityMap>> entity_maps
        = {entity_map};

    // We can now create the bilinear form
    fem::Form<T> a_mixed
        = fem::create_form<T>(*form_mixed_codim0_a_mixed, {V, W}, {}, {},
                              subdomain_data, entity_maps, V->mesh());

    la::SparsityPattern sp_mixed = fem::create_sparsity_pattern(a_mixed);
    sp_mixed.finalize();
    la::MatrixCSR<PetscScalar> A_mixed(sp_mixed);
    fem::assemble_matrix(A_mixed.mat_add_values(), a_mixed, {});
    A_mixed.scatter_rev();

    fem::Form<T> a
        = fem::create_form<T>(*form_mixed_codim0_a, {W, W}, {}, {}, {}, {});

    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<PetscScalar> A(sp);
    fem::assemble_matrix(A.mat_add_values(), a, {});
    A.scatter_rev();

    std::vector<T> A_mixed_flattened = A_mixed.to_dense();
    std::stringstream cc;
    cc.precision(3);
    cc << "A_mixed:" << std::endl;

    std::size_t num_owned_rows = V->dofmap()->index_map->size_local();
    std::size_t num_sub_cols = W->dofmap()->index_map->size_local()
                               + W->dofmap()->index_map->num_ghosts();
    for (std::size_t i = 0; i < num_owned_rows; ++i)
    {
      for (std::size_t j = 0; j < num_sub_cols; ++j)
        cc << A_mixed_flattened[i * num_sub_cols + j] << " ";
      cc << std::endl;
    }

    std::size_t num_owned_sub_rows = W->dofmap()->index_map->size_local();
    std::vector<T> A_flattened = A.to_dense();
    cc << "A" << std::endl;
    for (std::size_t i = 0; i < num_owned_sub_rows; ++i)
    {
      for (std::size_t j = 0; j < num_sub_cols; ++j)
        cc << A_flattened[i * num_sub_cols + j] << " ";
      cc << std::endl;
    }
    std::cout << cc.str() << std::endl;
  }

  PetscFinalize();

  return 0;
}
