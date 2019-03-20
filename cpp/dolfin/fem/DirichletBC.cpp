// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "FiniteElement.h"
#include "GenericDofMap.h"
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
#include <spdlog/spdlog.h>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
//-----------------------------------------------------------------------------
template <class T>
std::set<PetscInt> gather_new(MPI_Comm mpi_comm, const GenericDofMap& dofmap,
                              const T& dofs)
{
  std::size_t comm_size = MPI::size(mpi_comm);

  const auto& shared_nodes = dofmap.shared_nodes();
  const int bs = dofmap.block_size();
  assert(dofmap.index_map());

  // Create list of boundary values to send to each processor
  std::vector<std::vector<PetscInt>> proc_map(comm_size);
  for (auto dof : dofs)
  {
    // If the boundary value is attached to a shared dof, add it to the
    // list of boundary values for each of the processors that share it
    const std::div_t div = std::div(dof[0], bs);
    const int component = div.rem;
    const int node_index = div.quot;

    auto shared_node = shared_nodes.find(node_index);
    if (shared_node != shared_nodes.end())
    {
      for (auto proc = shared_node->second.begin();
           proc != shared_node->second.end(); ++proc)
      {
        const std::size_t global_node
            = dofmap.index_map()->local_to_global(node_index);
        proc_map[*proc].push_back(bs * global_node + component);
      }
    }
  }

  // Distribute the lists between neighbours
  std::vector<PetscInt> received_bvc;
  MPI::all_to_all(mpi_comm, proc_map, received_bvc);

  const std::int64_t n0 = dofmap.ownership_range()[0];
  const std::int64_t n1 = dofmap.ownership_range()[1];
  const std::int64_t owned_size = n1 - n0;

  // Add the received boundary values to the local boundary values
  std::set<PetscInt> _vec;
  for (PetscInt index_global : received_bvc)
  {
    // Convert to local (process) dof index
    int local_index = -1;
    if (index_global >= n0 and index_global < n1)
    {
      // Case 0: dof is owned by this process
      local_index = index_global - n0;
    }
    else
    {
      const std::imaxdiv_t div = std::imaxdiv(index_global, bs);
      const std::size_t node = div.quot;
      const int component = div.rem;

      // Get local-to-global for ghost blocks
      const auto& local_to_global = dofmap.index_map()->ghosts();

      // Case 1: dof is not owned by this process
      auto it
          = std::find(local_to_global.data(),
                      local_to_global.data() + local_to_global.size(), node);
      if (it == (local_to_global.data() + local_to_global.size()))
      {
        throw std::runtime_error(
            "Cannot find dof in local_to_global_unowned array");
      }
      else
      {
        std::size_t pos = std::distance(local_to_global.data(), it);
        local_index = owned_size + bs * pos + component;
      }
    }

    assert(local_index >= 0);
    _vec.insert(local_index);
  }

  return _vec;
}
//-----------------------------------------------------------------------------
bool on_facet(
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> coordinates,
    const mesh::Facet& facet)
{
  if (facet.dim() == 1)
  {
    // Check if the coordinates are on the same line as the line segment

    // Create points
    geometry::Point p(coordinates[0], coordinates[1]);
    const geometry::Point v0
        = mesh::Vertex(facet.mesh(), facet.entities(0)[0]).point();
    const geometry::Point v1
        = mesh::Vertex(facet.mesh(), facet.entities(0)[1]).point();

    // Create vectors
    const geometry::Point v01 = v1 - v0;
    const geometry::Point vp0 = v0 - p;
    const geometry::Point vp1 = v1 - p;

    // Check if the length of the sum of the two line segments vp0 and
    // vp1 is equal to the total length of the facet
    if (std::abs(v01.norm() - vp0.norm() - vp1.norm()) < 2.0 * DBL_EPSILON)
      return true;
    else
      return false;
  }
  else if (facet.dim() == 2)
  {
    // Check if the coordinates are in the same plane as the triangular
    // facet

    // Create points
    const geometry::Point p(coordinates[0], coordinates[1], coordinates[2]);
    const geometry::Point v0
        = mesh::Vertex(facet.mesh(), facet.entities(0)[0]).point();
    const geometry::Point v1
        = mesh::Vertex(facet.mesh(), facet.entities(0)[1]).point();
    const geometry::Point v2
        = mesh::Vertex(facet.mesh(), facet.entities(0)[2]).point();

    // Create vectors
    const geometry::Point v01 = v1 - v0;
    const geometry::Point v02 = v2 - v0;
    const geometry::Point vp0 = v0 - p;
    const geometry::Point vp1 = v1 - p;
    const geometry::Point vp2 = v2 - p;

    // Check if the sum of the area of the sub triangles is equal to the
    // total area of the facet
    if (std::abs(v01.cross(v02).norm() - vp0.cross(vp1).norm()
                 - vp1.cross(vp2).norm() - vp2.cross(vp0).norm())
        < 2.0 * DBL_EPSILON)
    {
      return true;
    }
    else
      return false;
  }

  throw std::runtime_error("Determine if given point is on facet. Not "
                           "implemented for given facet dimension");

  return false;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
                         std::shared_ptr<const function::Function> g,
                         const mesh::SubDomain& sub_domain, Method method,
                         bool check_midpoint)
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

  // FIXME: This can be made more efficient, we should be able to
  //        extract the facets without first creating a
  //        mesh::MeshFunction on the entire mesh and then extracting
  //        the subset. This is done mainly for convenience (we may
  //        reuse mark() in SubDomain).

  std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
  assert(mesh);

  // Create mesh function for sub domain markers on facets and mark
  // all facet as subdomain 1
  const std::size_t dim = mesh->topology().dim();
  _function_space->mesh()->init(dim - 1);
  mesh::MeshFunction<std::size_t> domain(mesh, dim - 1, 1);

  // Mark the sub domain as sub domain 0
  sub_domain.mark(domain, (std::size_t)0, check_midpoint);

  // Build set of boundary facets
  std::vector<std::int32_t> _facets;
  for (auto& facet : mesh::MeshRange<mesh::Facet>(*mesh))
  {
    if (domain[facet] == 0)
      _facets.push_back(facet.index());
  }

  std::set<std::array<PetscInt, 2>> dofs_local;
  if (method == Method::topological)
  {
    dofs_local
        = compute_bc_dofs_topological(*V, g->function_space().get(), _facets);
  }
  else if (method == Method::geometric)
  {
    throw std::runtime_error("BC method not yet supported");
    // dofs_local = compute_bc_dofs_geometric(*V, nullptr, _facets);
  }
  else
    throw std::runtime_error("BC method not yet supported");

  const std::set<PetscInt> dofs_remote
      = gather_new(mesh->mpi_comm(), *V->dofmap(), dofs_local);
  // std::set_union(dofs_local.begin(), dofs_local.end(), dofs_remote.begin(),
  //                dofs_remote.end(), std::back_inserter(_dofs));

  const std::map<PetscInt, PetscInt> shared_dofs
      = shared_bc_to_g(*V, *g->function_space());
  for (auto dof_remote : dofs_remote)
  {
    auto it = shared_dofs.find(dof_remote);
    if (it == shared_dofs.end())
      throw std::runtime_error("Oops, can't find dof (A).");

    const std::array<PetscInt, 2> ldofs = {{it->first, it->second}};
    auto it_map = dofs_local.insert(ldofs);
    if (it_map.second)
      std::cout << "Inserted off-process dof (A)" << std::endl;
  }

  _dofs = Eigen::Array<PetscInt, Eigen::Dynamic, 2, Eigen::RowMajor>(
      dofs_local.size(), 2);
  for (auto e = dofs_local.cbegin(); e != dofs_local.cend(); ++e)
  {
    std::size_t pos = std::distance(dofs_local.begin(), e);
    _dofs(pos, 0) = (*e)[0];
    _dofs(pos, 1) = (*e)[1];
  }
  _dof_indices = _dofs.col(0);
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
  std::set<std::array<PetscInt, 2>> dofs_local;
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

  // std::cout << "Local dofs size: " << MPI::rank(MPI_COMM_WORLD) << ", "
  //           << dofs_local.size() << std::endl;

  std::set<PetscInt> dofs_remote
      = gather_new(V->mesh()->mpi_comm(), *V->dofmap(), dofs_local);

  const std::map<PetscInt, PetscInt> shared_dofs
      = shared_bc_to_g(*V, *g->function_space());
  for (auto dof_remote : dofs_remote)
  {
    auto it = shared_dofs.find(dof_remote);
    if (it == shared_dofs.end())
      throw std::runtime_error("Oops, can't find dof (B).");
    const std::array<PetscInt, 2> ldofs = {{it->first, it->second}};
    auto it_map = dofs_local.insert(ldofs);
    if (it_map.second)
      std::cout << "Inserted off-process dof (B)" << std::endl;
  }

  _dofs = Eigen::Array<PetscInt, Eigen::Dynamic, 2, Eigen::RowMajor>(
      dofs_local.size(), 2);
  for (auto e = dofs_local.cbegin(); e != dofs_local.cend(); ++e)
  {
    std::size_t pos = std::distance(dofs_local.begin(), e);
    _dofs(pos, 0) = (*e)[0];
    _dofs(pos, 1) = (*e)[1];
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
std::set<std::array<PetscInt, 2>> DirichletBC::compute_bc_dofs_topological(
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

  return std::set<std::array<PetscInt, 2>>(bc_dofs.begin(), bc_dofs.end());
}
//-----------------------------------------------------------------------------
std::set<PetscInt>
DirichletBC::compute_bc_dofs_geometric(const function::FunctionSpace& V,
                                       const function::FunctionSpace* Vg,
                                       const std::vector<std::int32_t>& facets)
{
  assert(V.element());

  // Get mesh
  assert(V.mesh());
  const mesh::Mesh& mesh = *V.mesh();

  // Get dofmap
  assert(V.dofmap());
  const GenericDofMap& dofmap = *V.dofmap();

  const GenericDofMap* dofmap_g = &dofmap;
  if (Vg)
  {
    assert(Vg->dofmap());
    dofmap_g = Vg->dofmap().get();
  }

  // Get finite element
  assert(V.element());
  const FiniteElement& element = *V.element();

  // Initialize facets, needed for geometric search
  spdlog::info("Computing facets, needed for geometric application of boundary "
               "conditions.");
  mesh.init(mesh.topology().dim() - 1);

  // Speed up the computations by only visiting (most) dofs once
  common::RangedIndexSet already_visited(
      dofmap.is_view() ? std::array<std::int64_t, 2>{{0, 0}}
                       : dofmap.ownership_range());

  // Topological and geometric dimensions
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Get dof coordinates on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = element.dof_reference_coordinates();

  // Get coordinate mapping
  if (!mesh.geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

  // Create vertex coordinate holder
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Coordinates for dofs
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x;

  // Iterate over facets
  std::vector<PetscInt> bc_dofs;
  std::vector<PetscInt> bc_dofs_g;
  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    // Create facet and attached cell (get first attached cell)
    const mesh::Facet facet(mesh, facets[f]);
    const mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Loop over vertices associated with the facet
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(facet))
    {
      // Loop the cells associated with the vertex
      for (auto& c : mesh::EntityRange<mesh::Cell>(vertex))
      {
        coordinate_dofs.resize(cell.num_vertices(), gdim);
        c.get_coordinate_dofs(coordinate_dofs);

        // Tabulate dof coordinates on physical element
        cmap.compute_physical_coordinates(x, X, coordinate_dofs);

        // Get cell dofmap
        auto cell_dofs = dofmap.cell_dofs(c.index());
        auto cell_dofs_g = dofmap_g->cell_dofs(c.index());

        // Loop over all cell dofs
        for (int i = 0; i < cell_dofs.size(); ++i)
        {
          // Check if the dof coordinate is on current facet
          if (!on_facet(x.row(i), facet))
            continue;

          // Skip already checked dofs
          if (already_visited.in_range(cell_dofs[i])
              and !already_visited.insert(cell_dofs[i]))
          {
            continue;
          }

          bc_dofs.push_back(cell_dofs[i]);
          bc_dofs_g.push_back(cell_dofs_g[i]);
        }
      }
    }
  }

  // FIXME: Send to other (neigbouring) processes, maybe just for shared
  // dofs?

  return std::set<PetscInt>(bc_dofs.begin(), bc_dofs.end());
}
//-----------------------------------------------------------------------------
