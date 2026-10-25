// # Mixed-domain assembly with a submesh of co-dimension 0
//
// This demo illustrates how to:
//
// * Create a submesh of co-dimension 0 from a subset of mesh cells
// * Assemble a form whose trial function is defined on the submesh and
//   whose test function is defined on the parent mesh
//
// ## Problem definition
//
// Let $\Omega = [0, 2] \times [0, 1]$ and let $\Omega_s \subset \Omega$
// be the cells with $y \le 0.5$. With $V$ and $W$ the bilinear (Q1)
// Lagrange spaces on $\Omega$ and on the submesh $\Omega_s$, respectively,
// with basis functions $\phi_i$ and $\psi_j$, we assemble the rectangular
// matrix
//
// $$
//   (A_{\rm mixed})_{ij} = \int_{\Omega_s} \psi_j \phi_i \, {\rm d}x,
// $$
//
// integrating over cells of the parent mesh, and the mass matrix on the
// submesh
//
// $$
//   A_{ij} = \int_{\Omega_s} \psi_j \psi_i \, {\rm d}x.
// $$
//
// Both matrices are printed. A basis function of $V$ associated with a
// vertex in $\Omega_s$ restricts to a basis function of $W$ on
// $\Omega_s$, so the corresponding rows of $A_{\rm mixed}$ and $A$
// coincide, up to the numbering of the degrees-of-freedom.
//
// ## Implementation
//
// Running this demo requires the files:
// {download}`demo_codim_0_assembly/main.cpp`,
// {download}`demo_codim_0_assembly/mixed_codim0.py` and
// {download}`demo_codim_0_assembly/CMakeLists.txt`.
//
// ### UFL form file
//
// The UFL forms are defined in
// {download}`demo_codim_0_assembly/mixed_codim0.py`.
// ````{admonition} UFL form implemented in Python
// :class: dropdown
// ![ufl-code]
// ````
//
// ### C++ program
//
// The submesh is created with {cpp:func}`dolfinx::mesh::create_submesh`,
// which also returns a {cpp:class}`dolfinx::mesh::EntityMap` relating
// submesh cells to parent mesh cells. The form $A_{\rm mixed}$ is
// integrated over the parent mesh (the *integration domain*), and the
// entity map is passed to {cpp:func}`dolfinx::fem::create_form` so that
// the submesh trial function can be evaluated on parent mesh cells.

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
  dolfinx::common::petsc::check(PetscInitialize(&argc, &argv, nullptr, nullptr),
                                "PetscInitialize");

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {1, 4},
        mesh::CellType::quadrilateral, graph::partition_graph,
        mesh::DiagonalType::right, 2, mesh::GhostMode::shared_facet));

    basix::FiniteElement element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::quadrilateral, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      element, mesh->geometry().dim())));

    // Find all cells of the mesh with y <= 0.5
    const int tdim = mesh->topology()->dim();
    std::vector<std::int32_t> marked_cells = mesh::locate_entities(
        *mesh, tdim,
        [](auto x)
        {
          using coord_t = typename decltype(x)::value_type;
          constexpr coord_t eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            if (std::abs(x(1, p)) <= 0.5 + eps)
              marker[p] = true;
          }
          return marker;
        });

    // Create a MeshTags object that marks these cells with 2, and all
    // other cells with 1
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

    // Create a submesh of the cells marked 2. The returned EntityMap
    // relates submesh entities to mesh entities.
    auto submesh_data = [](auto& mesh, int dim, auto&& subcells)
    {
      auto [submesh, emap, v_map, g_map]
          = mesh::create_submesh(mesh, dim, subcells);
      return std::pair(std::make_shared<mesh::Mesh<U>>(std::move(submesh)),
                       std::move(emap));
    };
    auto [submesh, entity_map] = submesh_data(*mesh, tdim, cell_marker.find(2));

    // Create the trial space on the submesh
    auto W
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            submesh, std::make_shared<fem::FiniteElement<U>>(
                         element, submesh->geometry().dim())));

    // Compute the integration entities on the integration domain mesh
    // for the cells marked 2, and associate them with subdomain ID 3,
    // the ID used by dx in the UFL file
    std::vector<std::int32_t> integration_entities
        = fem::compute_integration_domains(
            fem::IntegralType::cell, *mesh->topology(), cell_marker.find(2));
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_data
        = {{fem::IntegralType::cell, {{3, integration_entities}}}};

    // Create the mixed-domain form. The integration domain mesh is
    // passed last, and an EntityMap is supplied for each other mesh in
    // the form (here only submesh).
    fem::Form<T> a_mixed
        = fem::create_form<T>(*form_mixed_codim0_a_mixed, {V, W}, {}, {},
                              subdomain_data, {entity_map}, V->mesh());

    la::SparsityPattern sp_mixed = fem::create_sparsity_pattern(a_mixed);
    sp_mixed.finalize();
    la::MatrixCSR<PetscScalar> A_mixed(sp_mixed);
    fem::assemble_matrix(A_mixed.mat_add_values(), a_mixed, {});
    A_mixed.scatter_rev();

    // Assemble the mass matrix on the submesh
    fem::Form<T> a
        = fem::create_form<T>(*form_mixed_codim0_a, {W, W}, {}, {}, {}, {});

    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<PetscScalar> A(sp);
    fem::assemble_matrix(A.mat_add_values(), a, {});
    A.scatter_rev();

    // Print the owned rows of both matrices
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

  common::petsc::check(PetscFinalize(), "PetscFinalize");
  return 0;
}
