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
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {1, 4}, mesh::CellType::quadrilateral, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::quadrilateral, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element, {}));

    // Next we find all cells of the mesh with y<0.5
    const int tdim = mesh->topology()->dim();
    auto marked_cells = mesh::locate_entities(
        *mesh, tdim,
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
    // We create a MeshTags object where we mark these cells with 2, and any
    // other cell with 1
    auto cell_map = mesh->topology()->index_map(tdim);
    std::size_t num_cells_local
        = mesh->topology()->index_map(tdim)->size_local()
          + mesh->topology()->index_map(tdim)->num_ghosts();
    std::vector<std::int32_t> cells(num_cells_local);
    std::iota(cells.begin(), cells.end(), 0);
    std::vector<std::int32_t> values(cell_map->size_local(), 1);
    std::for_each(marked_cells.begin(), marked_cells.end(),
                  [&values](auto& c) { values[c] = 2; });
    dolfinx::mesh::MeshTags<std::int32_t> cell_marker(mesh->topology(), tdim,
                                                      cells, values);

    std::shared_ptr<mesh::Mesh<U>> submesh;
    std::vector<std::int32_t> submesh_to_mesh;
    {
      auto [_submesh, _submesh_to_mesh, v_map, g_map]
          = mesh::create_submesh(*mesh, tdim, cell_marker.find(2));
      submesh = std::make_shared<mesh::Mesh<U>>(std::move(_submesh));
      submesh_to_mesh = std::move(_submesh_to_mesh);
    }

    // We create the function space used for the trial space
    auto W = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(submesh, element, {}));

    // A mixed-domain form has functions defined over different meshes. The mesh
    // associated with the measure (dx, ds, etc.) is called the integration
    // domain. To assemble mixed-domain forms, maps must be provided taking
    // entities in the integration domain to entities on each mesh in the form.
    // Since one of our forms has a measure defined over `mesh` and involves a
    // function defined over `submesh`, we must provide a map from entities in
    // `mesh` to entities in `submesh`. This is simply the "inverse" of
    // `submesh_to_mesh`.
    std::vector<std::int32_t> mesh_to_submesh(num_cells_local, -1);
    for (std::size_t i = 0; i < submesh_to_mesh.size(); ++i)
      mesh_to_submesh[submesh_to_mesh[i]] = i;

    std::shared_ptr<const mesh::Mesh<U>> const_ptr = submesh;
    std::map<std::shared_ptr<const mesh::Mesh<U>>,
             std::span<const std::int32_t>>
        entity_maps
        = {{const_ptr, std::span<const std::int32_t>(mesh_to_submesh.data(),
                                                     mesh_to_submesh.size())}};

    // Next we compute the integration entities on the integration domain `mesh`
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_map = {};
    auto integration_entities = fem::compute_integration_domains(
        fem::IntegralType::cell, *mesh->topology(), cell_marker.find(2), tdim);

    subdomain_map[fem::IntegralType::cell].push_back(
        {3, std::span<const std::int32_t>(integration_entities.data(),
                                          integration_entities.size())});

    // We can now create the bi-linear form
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_mixed_codim0_a, {V, W}, {}, {}, subdomain_map,
                            entity_maps, V->mesh()));

    la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
    sp.finalize();
    la::MatrixCSR<double> A(sp);
    fem::assemble_matrix(A.mat_add_values(), *a, {});
    A.scatter_rev();
    std::cout << "here3\n" << std::endl;

    auto a_sub = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_mixed_codim0_a_sub, {W, W}, {}, {}, {}, {}));
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
    std::size_t num_sub_cols = W->dofmap()->index_map->size_local()
                               + W->dofmap()->index_map->num_ghosts();
    for (std::size_t i = 0; i < num_owned_rows; i++)
    {
      for (std::size_t j = 0; j < num_sub_cols; j++)
      {
        cc << A_flattened[i * num_sub_cols + j] << " ";
      }
      cc << std::endl;
    }

    std::size_t num_owned_sub_rows = W->dofmap()->index_map->size_local();
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
