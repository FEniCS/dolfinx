// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdlib>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/RangedIndexSet.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>
#include <map>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
std::map<PetscInt, PetscInt>
get_remote_bcs(const common::IndexMap& map, const common::IndexMap& map_g,
               const std::vector<std::array<PetscInt, 2>>& dofs_local)
{

  std::map<PetscInt, PetscInt> dof_dof_g;

  const std::int32_t bs = map.block_size();
  const std::int32_t size_owned = map.size_local();
  const std::int32_t size_ghost = map.num_ghosts();

  const std::int32_t bs_g = map_g.block_size();
  const std::int32_t size_owned_g = map_g.size_local();
  const std::int32_t size_ghost_g = map_g.num_ghosts();
  const std::array<std::int64_t, 2> range_g = map_g.local_range();
  const std::int64_t offset_g = range_g[0];

  // For each dof local index, store global index in Vg (-1 if no bc)
  std::vector<PetscInt> marker_owned(bs * size_owned, -1);
  std::vector<PetscInt> marker_ghost(bs * size_ghost, -1);
  for (auto& dofs : dofs_local)
  {
    const PetscInt index_block_g = dofs[1] / bs_g;
    const PetscInt pos_g = dofs[1] % bs_g;
    if (dofs[0] < bs * size_owned)
    {
      marker_owned[dofs[0]]
          = bs_g * map_g.local_to_global(index_block_g) + pos_g;
    }
    else
    {
      marker_ghost[dofs[0] - (bs * size_owned)]
          = bs_g * map_g.local_to_global(index_block_g) + pos_g;
    }
  }

  // Build global-to-local map for ghost indices (blocks) in map_g
  std::map<PetscInt, PetscInt> global_to_local_g;
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghosts_g = map_g.ghosts();
  for (Eigen::Index i = 0; i < size_owned_g; ++i)
    global_to_local_g.insert({i + offset_g, i});
  for (Eigen::Index i = 0; i < size_ghost_g; ++i)
    global_to_local_g.insert({ghosts_g[i], i + size_owned_g});

  // For each owned bc index, scatter associated g global index to ghost
  // processes
  std::vector<PetscInt> marker_ghost_rcvd = map.scatter_fwd(marker_owned, bs);
  assert((int)marker_ghost_rcvd.size() == size_ghost * bs);

  // Add to (local index)-(local g index) map
  for (std::size_t i = 0; i < marker_ghost_rcvd.size(); ++i)
  {
    if (marker_ghost_rcvd[i] > -1)
    {
      const PetscInt index_block_g = marker_ghost_rcvd[i] / bs_g;
      const PetscInt pos_g = marker_ghost_rcvd[i] % bs_g;
      const auto it = global_to_local_g.find(index_block_g);
      assert(it != global_to_local_g.end());
      dof_dof_g.insert({bs * size_owned + i, bs_g * it->second + pos_g});
    }
  }

  // Scatter (reverse) data from ghost processes to owner
  std::vector<PetscInt> marker_owner_rcvd(bs * size_owned, -1);
  map.scatter_rev(marker_owner_rcvd, marker_ghost, bs, MPI_MAX);
  assert((int)marker_owner_rcvd.size() == size_owned * bs);
  for (std::size_t i = 0; i < marker_owner_rcvd.size(); ++i)
  {
    if (marker_owner_rcvd[i] >= 0)
    {
      const PetscInt index_global_g = marker_owner_rcvd[i];
      const PetscInt index_block_g = index_global_g / bs_g;
      const PetscInt pos_g = index_global_g % bs_g;
      const auto it = global_to_local_g.find(index_block_g);
      assert(it != global_to_local_g.end());
      dof_dof_g.insert({i, bs_g * it->second + pos_g});
    }
  }

  return dof_dof_g;
}
//-----------------------------------------------------------------------------
// Return list of facet indices that are marked
std::vector<std::int32_t> facets_marked(std::shared_ptr<const mesh::Mesh> mesh,
                                        const mesh::SubDomain& sub_domain,
                                        bool check_midpoint)
{
  // FIXME: This can be made more efficient, we should be able to
  //        extract the facets without first creating a
  //        mesh::MeshFunction on the entire mesh and then extracting
  //        the subset. This is done mainly for convenience (we may
  //        reuse mark() in SubDomain).

  assert(mesh);

  // Create mesh function for sub domain markers on facets and mark
  // all facet as subdomain 1
  const std::size_t dim = mesh->topology().dim();
  mesh->init(dim - 1);
  mesh::MeshFunction<std::size_t> domain(mesh, dim - 1, 1);

  // Mark the sub domain as sub domain 0
  sub_domain.mark(domain, (std::size_t)0, check_midpoint);

  // Build set of boundary facets
  std::vector<std::int32_t> facets;
  for (auto& facet : mesh::MeshRange<mesh::Facet>(*mesh))
  {
    if (domain[facet] == 0)
      facets.push_back(facet.index());
  }

  return facets;
}
//-----------------------------------------------------------------------------
// bool on_facet(
//     const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>
//     coordinates, const mesh::Facet& facet)
// {
//   if (facet.dim() == 1)
//   {
//     // Check if the coordinates are on the same line as the line segment

//     // Create points
//     geometry::Point p(coordinates[0], coordinates[1]);
//     const geometry::Point v0
//         = mesh::Vertex(facet.mesh(), facet.entities(0)[0]).point();
//     const geometry::Point v1
//         = mesh::Vertex(facet.mesh(), facet.entities(0)[1]).point();

//     // Create vectors
//     const geometry::Point v01 = v1 - v0;
//     const geometry::Point vp0 = v0 - p;
//     const geometry::Point vp1 = v1 - p;

//     // Check if the length of the sum of the two line segments vp0 and
//     // vp1 is equal to the total length of the facet
//     if (std::abs(v01.norm() - vp0.norm() - vp1.norm()) < 2.0 * DBL_EPSILON)
//       return true;
//     else
//       return false;
//   }
//   else if (facet.dim() == 2)
//   {
//     // Check if the coordinates are in the same plane as the triangular
//     // facet

//     // Create points
//     const geometry::Point p(coordinates[0], coordinates[1], coordinates[2]);
//     const geometry::Point v0
//         = mesh::Vertex(facet.mesh(), facet.entities(0)[0]).point();
//     const geometry::Point v1
//         = mesh::Vertex(facet.mesh(), facet.entities(0)[1]).point();
//     const geometry::Point v2
//         = mesh::Vertex(facet.mesh(), facet.entities(0)[2]).point();

//     // Create vectors
//     const geometry::Point v01 = v1 - v0;
//     const geometry::Point v02 = v2 - v0;
//     const geometry::Point vp0 = v0 - p;
//     const geometry::Point vp1 = v1 - p;
//     const geometry::Point vp2 = v2 - p;

//     // Check if the sum of the area of the sub triangles is equal to the
//     // total area of the facet
//     if (std::abs(v01.cross(v02).norm() - vp0.cross(vp1).norm()
//                  - vp1.cross(vp2).norm() - vp2.cross(vp0).norm())
//         < 2.0 * DBL_EPSILON)
//     {
//       return true;
//     }
//     else
//       return false;
//   }

//   throw std::runtime_error("Determine if given point is on facet. Not "
//                            "implemented for given facet dimension");

//   return false;
// }
// //-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
                         std::shared_ptr<const function::Function> g,
                         const mesh::SubDomain& sub_domain, Method method,
                         bool check_midpoint)
    : DirichletBC(V, g, facets_marked(V->mesh(), sub_domain, check_midpoint),
                  method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
                         std::shared_ptr<const function::Function> g,
                         const std::vector<std::int32_t>& facet_indices,
                         Method method)
    : _function_space(V), _g(g)
{
  assert(V);
  assert(g);
  assert(g->function_space());
  if (V != g->function_space())
  {
    assert(V->mesh());
    assert(g->function_space()->mesh());
    if (V->mesh() != g->function_space()->mesh())
    {
      throw std::runtime_error("Boundary condition function and constrained "
                               "function do not share mesh.");
    }

    assert(g->function_space()->element());
    if (!V->has_element(*g->function_space()->element()))
    {
      throw std::runtime_error("Boundary condition function and constrained "
                               "function do not have same element.");
    }
  }

  assert(V);
  std::vector<std::array<PetscInt, 2>> dofs_local;
  if (method == Method::topological)
  {
    dofs_local = compute_bc_dofs_topological(*V, g->function_space().get(),
                                             facet_indices);
  }
  else if (method == Method::geometric)
  {
    throw std::runtime_error("BC method not yet supported");
    // dofs_local = compute_bc_dofs_geometric(*V, nullptr, _facets);
  }
  else
    throw std::runtime_error("BC method not yet supported");

  // Remove duplicates
  std::sort(dofs_local.begin(), dofs_local.end());
  dofs_local.erase(std::unique(dofs_local.begin(), dofs_local.end()),
                   dofs_local.end());

  // Get bc dof indices (local) in (V, Vg) spaces on this process that
  // were found by other processes, e.g. a vertex dof on this process that
  // has no connected factes on the boundary.
  const std::map<PetscInt, PetscInt> dofs_remote
      = get_remote_bcs(*V->dofmap()->index_map(),
                       *g->function_space()->dofmap()->index_map(), dofs_local);

  // Add received bc indices to dofs_local
  for (auto& dof_remote : dofs_remote)
  {
    const std::array<PetscInt, 2> ldofs
        = {{dof_remote.first, dof_remote.second}};
    dofs_local.push_back(ldofs);
  }

  // Remove duplicates
  std::sort(dofs_local.begin(), dofs_local.end());
  dofs_local.erase(std::unique(dofs_local.begin(), dofs_local.end()),
                   dofs_local.end());

  _dofs = Eigen::Array<PetscInt, Eigen::Dynamic, 2, Eigen::RowMajor>(
      dofs_local.size(), 2);
  for (std::size_t i = 0; i < dofs_local.size(); ++i)
  {
    _dofs(i, 0) = dofs_local[i][0];
    _dofs(i, 1) = dofs_local[i][1];
  }

  _dof_indices = _dofs.col(0);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::FunctionSpace>
DirichletBC::function_space() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::Function> DirichletBC::value() const
{
  return _g;
}
//-----------------------------------------------------------------------------
const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
DirichletBC::dof_indices() const
{
  return _dof_indices;
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    double scale) const
{
  // FIXME: This one excludes ghosts. Need to straighten out.

  assert(_g);
  la::VecReadWrapper g(_g->vector().vec(), false);
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    if (_dofs(i, 0) < x.rows())
      x[_dofs(i, 0)] = scale * g.x[_dofs(i, 1)];
  }

  // for (auto& dof : _dofs)
  // {
  //   if (dof[0] < x.rows())
  //     x[dof[0]] = scale * g.x[dof[1]];
  // }
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale) const
{
  // FIXME: This one excludes ghosts. Need to straighten out.

  assert(_g);
  assert(x.rows() == x0.rows());
  la::VecReadWrapper g(_g->vector().vec(), false);
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
  {
    if (_dofs(i, 0) < x.rows())
      x[_dofs(i, 0)] = scale * (g.x[_dofs(i, 1)] - x0[_dofs(i, 0)]);
  }
  // for (auto& dof : _dofs)
  // {
  //   if (dof[0] < x.rows())
  //     x[dof[0]] = scale * (g.x[dof[1]] - x0[dof[0]]);
  // }
}
//-----------------------------------------------------------------------------
void DirichletBC::dof_values(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> values) const
{
  assert(_g);
  la::VecReadWrapper g(_g->vector().vec());
  for (Eigen::Index i = 0; i < _dofs.rows(); ++i)
    values[_dofs(i, 0)] = g.x[_dofs(i, 1)];
  // for (auto& dof : _dofs)
  //   values[dof[0]] = g.x[dof[1]];
  g.restore();
}
//-----------------------------------------------------------------------------
void DirichletBC::mark_dofs(std::vector<bool>& markers) const
{
  for (Eigen::Index i = 0; i < _dof_indices.size(); ++i)
  {
    assert(_dof_indices[i] < (PetscInt)markers.size());
    markers[_dof_indices[i]] = true;
  }
}
//-----------------------------------------------------------------------------
std::map<PetscInt, PetscInt>
DirichletBC::shared_bc_to_g(const function::FunctionSpace& V,
                            const function::FunctionSpace& Vg)
{
  // Get mesh
  assert(V.mesh());
  const mesh::Mesh& mesh = *V.mesh();
  const std::size_t tdim = mesh.topology().dim();

  // Get dofmaps
  assert(V.dofmap());
  assert(Vg.dofmap());
  const GenericDofMap& dofmap = *V.dofmap();
  const GenericDofMap& dofmap_g = *Vg.dofmap();

  // Initialise facet-cell connectivity
  mesh.init(tdim - 1);
  mesh.init(tdim, tdim - 1);

  // Allocate space
  const std::size_t num_facet_dofs = dofmap.num_entity_closure_dofs(tdim - 1);

  // Build vector local dofs for each cell facet
  const mesh::CellType& cell_type = mesh.type();
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> facet_dofs;
  for (std::size_t i = 0; i < cell_type.num_entities(tdim - 1); ++i)
    facet_dofs.push_back(dofmap.tabulate_entity_closure_dofs(tdim - 1, i));

  std::vector<std::pair<PetscInt, PetscInt>> dofs;
  for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    assert(facet.num_entities(tdim) > 0);
    if (facet.num_entities(tdim) > 0)
    {
      const std::size_t cell_index = facet.entities(tdim)[0];
      const mesh::Cell cell(mesh, cell_index);

      // Get cell dofmap
      const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
          cell_dofs = dofmap.cell_dofs(cell.index());
      const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
          cell_dofs_g = dofmap_g.cell_dofs(cell.index());

      // Loop over facet dofs
      const size_t facet_local_index = cell.index(facet);
      for (std::size_t i = 0; i < num_facet_dofs; i++)
      {
        const std::size_t index = facet_dofs[facet_local_index][i];
        const PetscInt dof_index = cell_dofs[index];
        const PetscInt dof_index_g = cell_dofs_g[index];
        dofs.push_back({dof_index, dof_index_g});
      }
    }
  }
  return std::map<PetscInt, PetscInt>(dofs.begin(), dofs.end());
}
//-----------------------------------------------------------------------------
std::vector<std::array<PetscInt, 2>> DirichletBC::compute_bc_dofs_topological(
    const function::FunctionSpace& V, const function::FunctionSpace* Vg,
    const std::vector<std::int32_t>& facets)
{
  // Get mesh
  assert(V.mesh());
  const mesh::Mesh& mesh = *V.mesh();
  const std::size_t tdim = mesh.topology().dim();

  // Get dofmap
  assert(V.dofmap());
  const GenericDofMap& dofmap = *V.dofmap();
  const GenericDofMap* dofmap_g = &dofmap;
  if (Vg)
  {
    assert(Vg->dofmap());
    dofmap_g = Vg->dofmap().get();
  }

  // Initialise facet-cell connectivity
  mesh.init(tdim);
  mesh.init(tdim - 1, tdim);

  // Allocate space
  const std::size_t num_facet_dofs = dofmap.num_entity_closure_dofs(tdim - 1);

  // Build vector local dofs for each cell facet
  const mesh::CellType& cell_type = mesh.type();
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> facet_dofs;
  for (std::size_t i = 0; i < cell_type.num_entities(tdim - 1); ++i)
    facet_dofs.push_back(dofmap.tabulate_entity_closure_dofs(tdim - 1, i));

  // Iterate over marked facets
  std::vector<std::array<PetscInt, 2>> bc_dofs;
  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    // Create facet and attached cell
    const mesh::Facet facet(mesh, facets[f]);
    assert(facet.num_entities(tdim) > 0);
    const std::size_t cell_index = facet.entities(tdim)[0];
    const mesh::Cell cell(mesh, cell_index);

    // Get cell dofmap
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dofs
        = dofmap.cell_dofs(cell.index());
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
        cell_dofs_g = dofmap_g->cell_dofs(cell.index());

    // Loop over facet dofs
    const size_t facet_local_index = cell.index(facet);
    for (std::size_t i = 0; i < num_facet_dofs; i++)
    {
      const std::size_t index = facet_dofs[facet_local_index][i];
      const PetscInt dof_index = cell_dofs[index];
      const PetscInt dof_index_g = cell_dofs_g[index];
      bc_dofs.push_back({{dof_index, dof_index_g}});
    }
  }

  return bc_dofs;
}
//-----------------------------------------------------------------------------
// std::set<PetscInt>
// DirichletBC::compute_bc_dofs_geometric(const function::FunctionSpace& V,
//                                        const function::FunctionSpace* Vg,
//                                        const std::vector<std::int32_t>&
//                                        facets)
// {
//   assert(V.element());

//   // Get mesh
//   assert(V.mesh());
//   const mesh::Mesh& mesh = *V.mesh();

//   // Get dofmap
//   assert(V.dofmap());
//   const GenericDofMap& dofmap = *V.dofmap();

//   const GenericDofMap* dofmap_g = &dofmap;
//   if (Vg)
//   {
//     assert(Vg->dofmap());
//     dofmap_g = Vg->dofmap().get();
//   }

//   // Get finite element
//   assert(V.element());
//   const FiniteElement& element = *V.element();

//   // Initialize facets, needed for geometric search
//   // spdlog::info("Computing facets, needed for geometric application of
//   // boundary "
//   //              "conditions.");
//   mesh.init(mesh.topology().dim() - 1);

//   // Speed up the computations by only visiting (most) dofs once
//   common::RangedIndexSet already_visited(
//       dofmap.is_view() ? std::array<std::int64_t, 2>{{0, 0}}
//                        : dofmap.index_map()->local_range(),
//       dofmap.index_map()->block_size());

//   // Topological and geometric dimensions
//   const std::size_t tdim = mesh.topology().dim();
//   const std::size_t gdim = mesh.geometry().dim();

//   // Get dof coordinates on reference element
//   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
//   Eigen::RowMajor>& X
//       = element.dof_reference_coordinates();

//   // Get coordinate mapping
//   if (!mesh.geometry().coord_mapping)
//   {
//     throw std::runtime_error(
//         "CoordinateMapping has not been attached to mesh.");
//   }
//   const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

//   // Create vertex coordinate holder
//   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//       coordinate_dofs;

//   // Coordinates for dofs
//   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x;

//   // Iterate over facets
//   std::vector<PetscInt> bc_dofs;
//   std::vector<PetscInt> bc_dofs_g;
//   for (std::size_t f = 0; f < facets.size(); ++f)
//   {
//     // Create facet and attached cell (get first attached cell)
//     const mesh::Facet facet(mesh, facets[f]);
//     const mesh::Cell cell(mesh, facet.entities(tdim)[0]);

//     // Loop over vertices associated with the facet
//     for (auto& vertex : mesh::EntityRange<mesh::Vertex>(facet))
//     {
//       // Loop the cells associated with the vertex
//       for (auto& c : mesh::EntityRange<mesh::Cell>(vertex))
//       {
//         coordinate_dofs.resize(cell.num_vertices(), gdim);
//         c.get_coordinate_dofs(coordinate_dofs);

//         // Tabulate dof coordinates on physical element
//         cmap.compute_physical_coordinates(x, X, coordinate_dofs);

//         // Get cell dofmap
//         auto cell_dofs = dofmap.cell_dofs(c.index());
//         auto cell_dofs_g = dofmap_g->cell_dofs(c.index());

//         // Loop over all cell dofs
//         for (int i = 0; i < cell_dofs.size(); ++i)
//         {
//           // Check if the dof coordinate is on current facet
//           if (!on_facet(x.row(i), facet))
//             continue;

//           // Skip already checked dofs
//           if (already_visited.in_range(cell_dofs[i])
//               and !already_visited.insert(cell_dofs[i]))
//           {
//             continue;
//           }

//           bc_dofs.push_back(cell_dofs[i]);
//           bc_dofs_g.push_back(cell_dofs_g[i]);
//         }
//       }
//     }
//   }

//   // FIXME: Send to other (neigbouring) processes, maybe just for shared
//   // dofs?

//   return std::set<PetscInt>(bc_dofs.begin(), bc_dofs.end());
// }
//-----------------------------------------------------------------------------
