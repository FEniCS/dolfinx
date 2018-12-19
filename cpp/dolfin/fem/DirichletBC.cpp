// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DirichletBC.h"
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/RangedIndexSet.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
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
    if (std::abs(v01.norm() - vp0.norm() - vp1.norm()) < DOLFIN_EPS)
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
        < DOLFIN_EPS)
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
  if (V == g->function_space())
    std::cout << "Spaces are the same" << std::endl;
  else
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
    std::cout << "Checking remote (A)" << std::endl;
    auto it = shared_dofs.find(dof_remote);
    if (it == shared_dofs.end())
      throw std::runtime_error("Oops, can't find dof (A).");

    auto it_map = dofs_local.insert({it->first, it->second});
    if (it_map.second)
      std::cout << "Inserted off-process dof (A)" << std::endl;
  }

  _dofs = std::vector<std::array<PetscInt, 2>>(dofs_local.begin(),
                                               dofs_local.end());

  _dof_indices = Eigen::Array<PetscInt, Eigen::Dynamic, 1>(_dofs.size());
  std::size_t i = 0;
  for (const auto& dof : _dofs)
    _dof_indices[i++] = dof[0];
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
  if (V == g->function_space())
    std::cout << "Spaces are the same" << std::endl;
  else
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
  // std::cout << "Num facets: " << _facets.size() << "----" << std::endl;
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
  // std::set<std::array<PetscInt, 2>> dofs_remote;
  std::set<PetscInt> dofs_remote
      = gather_new(V->mesh()->mpi_comm(), *V->dofmap(), dofs_local);

  const std::map<PetscInt, PetscInt> shared_dofs
      = shared_bc_to_g(*V, *g->function_space());
  for (auto dof_remote : dofs_remote)
  {
    std::cout << "Checking remote (B)" << std::endl;
    auto it = shared_dofs.find(dof_remote);
    if (it == shared_dofs.end())
      throw std::runtime_error("Oops, can't find dof (B).");
    auto it_map = dofs_local.insert({it->first, it->second});
    if (it_map.second)
      std::cout << "Inserted off-process dof (B)" << std::endl;
  }

  // std::set_union(dofs_local.begin(), dofs_local.end(), dofs_remote.begin(),
  //                dofs_remote.end(), std::back_inserter(_dofs));
  _dofs = std::vector<std::array<PetscInt, 2>>(dofs_local.begin(),
                                               dofs_local.end());
  _dof_indices = Eigen::Array<PetscInt, Eigen::Dynamic, 1>(_dofs.size());
  std::size_t i = 0;
  for (const auto& dof : _dofs)
    _dof_indices[i++] = dof[0];
}
//-----------------------------------------------------------------------------
void DirichletBC::get_boundary_values(Map& boundary_values) const
{
  // Unwrap bc vector
  assert(_g);
  assert(_g->vector());
  assert(_g->vector()->vec());
  const Vec g_vec = _g->vector()->vec();

  // Get local form
  Vec g_local(nullptr);
  VecGhostGetLocalForm(g_vec, &g_local);

  // Get size
  PetscInt g_size = 0;
  VecGetSize(g_local, &g_size);

  // Get array
  PetscScalar const* g_array;
  VecGetArrayRead(g_vec, &g_array);

  const Eigen::Map<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> g(
      g_array, g_size);
  for (auto dof : _dofs)
    boundary_values.insert({dof[0], g[dof[1]]});

  // Restore PETSc array
  VecRestoreArrayRead(g_local, &g_array);
  VecGhostRestoreLocalForm(g_vec, &g_local);
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
const Eigen::Array<PetscInt, Eigen::Dynamic, 1>&
DirichletBC::dof_indices() const
{
  return _dof_indices;
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    double scale) const
{
  // assert(x.rows() == (Eigen::Index)_dofs.size());
  assert(_g);
  assert(_g->vector());
  assert(_g->vector()->vec());

  // Unwrap PETSc bc vector (_g)
  const Vec g_vec = _g->vector()->vec();
  Vec g_local = nullptr;
  VecGhostGetLocalForm(g_vec, &g_local);
  assert(g_local);
  PetscInt g_size = 0;
  VecGetSize(g_local, &g_size);
  PetscScalar const* g_array;
  VecGetArrayRead(g_vec, &g_array);
  const Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> g(
      g_array, g_size);
  // assert(x.rows() == g.rows());

  // FIXME: This one excludes ghosts. Need to straighten out
  for (auto& dof : _dofs)
  {
    if (dof[0] < x.rows())
      x[dof[0]] = g[dof[1]];
  }

  // Restore PETSc array
  VecRestoreArrayRead(g_local, &g_array);
  VecGhostRestoreLocalForm(g_vec, &g_local);
}
//-----------------------------------------------------------------------------
void DirichletBC::set(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale) const
{
  // assert(x.rows() == (Eigen::Index)_dofs.size());
  // assert(x.rows() == x0.rows());

  assert(_g);
  assert(_g->vector());
  assert(_g->vector()->vec());

  // Unwrap PETSc bc vector (_g)
  const Vec g_vec = _g->vector()->vec();
  Vec g_local = nullptr;
  VecGhostGetLocalForm(g_vec, &g_local);
  assert(g_local);
  PetscInt g_size = 0;
  VecGetSize(g_local, &g_size);
  PetscScalar const* g_array;
  VecGetArrayRead(g_vec, &g_array);
  const Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> g(
      g_array, g_size);
  // assert(x.rows() == g.rows());

  // FIXME: This one excludes ghosts. Need to straighten out
  for (auto& dof : _dofs)
  {
    if (dof[0] < x.rows())
      x[dof[0]] = scale * (g[dof[1]] - x0[dof[0]]);
  }

  // Restore PETSc array
  VecRestoreArrayRead(g_local, &g_array);
  VecGhostRestoreLocalForm(g_vec, &g_local);
}
//-----------------------------------------------------------------------------
void DirichletBC::mark_dofs(std::vector<bool>& markers) const
{
  for (Eigen::Index i = 0; i < _dof_indices.size(); ++i)
  {
    assert(_dof_indices[i] < (PetscInt) markers.size());
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
      bc_dofs.push_back({dof_index, dof_index_g});

      // std::cout << "Old adding bc (cell, local, global): " << cell.index()
      //           << ", " << index << ", " << dof_index << std::endl;
    }
  }

  return std::set<std::array<PetscInt, 2>>(bc_dofs.begin(), bc_dofs.end());
}
//-----------------------------------------------------------------------------
// void DirichletBC::compute_bc_geometric(Map& boundary_values,
//                                        LocalData& data) const
// {
//   assert(_function_space);
//   assert(_function_space->element());
//   assert(_g);

//   // Get mesh
//   assert(_function_space->mesh());
//   const mesh::Mesh& mesh = *_function_space->mesh();

//   // Extract the list of facets where the BC *might* be applied
//   // init_facets(mesh.mpi_comm());

//   // Special case
//   if (_facets.empty())
//   {
//     if (MPI::size(mesh.mpi_comm()) == 1)
//       log::warning("Found no facets matching domain for boundary
//       condition.");
//     return;
//   }

//   // Get dofmap
//   assert(_function_space->dofmap());
//   const GenericDofMap& dofmap = *_function_space->dofmap();

//   // Get finite element
//   assert(_function_space->element());
//   const FiniteElement& element = *_function_space->element();

//   // Initialize facets, needed for geometric search
//   log::log(TRACE,
//            "Computing facets, needed for geometric application of boundary "
//            "conditions.");
//   mesh.init(mesh.topology().dim() - 1);

//   // Speed up the computations by only visiting (most) dofs once
//   common::RangedIndexSet already_visited(
//       dofmap.is_view() ? std::array<std::int64_t, 2>{{0, 0}}
//                        : dofmap.ownership_range());

//   // Topological and geometric dimensions
//   const std::size_t tdim = mesh.topology().dim();
//   const std::size_t gdim = mesh.geometry().dim();

//   // Get dof coordinates on reference element
//   const EigenRowArrayXXd& X = element.dof_reference_coordinates();

//   // Get coordinate mapping
//   if (!mesh.geometry().coord_mapping)
//   {
//     throw std::runtime_error(
//         "CoordinateMapping has not been attached to mesh.");
//   }
//   const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

//   // Iterate over facets
//   for (std::size_t f = 0; f < _facets.size(); ++f)
//   {
//     // Create facet
//     const mesh::Facet facet(mesh, _facets[f]);

//     // Create cell (get first attached cell)
//     const mesh::Cell cell(mesh, facet.entities(tdim)[0]);

//     // Get local index of facet with respect to the cell
//     // const std::size_t local_facet = cell.index(facet);

//     // Create vertex coordinate holder
//     EigenRowArrayXXd coordinate_dofs;

//     // Loop the vertices associated with the facet
//     for (auto& vertex : mesh::EntityRange<mesh::Vertex>(facet))
//     {
//       // Loop the cells associated with the vertex
//       for (auto& c : mesh::EntityRange<mesh::Cell>(vertex))
//       {
//         // FIXME: setting the local facet here looks wrong
//         // c.local_facet = local_facet;
//         coordinate_dofs.resize(cell.num_vertices(), gdim);
//         c.get_coordinate_dofs(coordinate_dofs);

//         bool tabulated = false;
//         bool interpolated = false;

//         // Tabulate dofs on cell
//         auto cell_dofs = dofmap.cell_dofs(c.index());

//         // Loop over all dofs on cell
//         for (int i = 0; i < cell_dofs.size(); ++i)
//         {
//           const std::size_t global_dof = cell_dofs[i];

//           // Tabulate coordinates if not already done
//           if (!tabulated)
//           {
//             cmap.compute_physical_coordinates(data.coordinates, X,
//                                               coordinate_dofs);
//             tabulated = true;
//           }

//           // Check if the coordinates are on current facet and thus on
//           // boundary
//           if (!on_facet(data.coordinates.row(i), facet))
//             continue;

//           // Skip already checked dofs
//           if (already_visited.in_range(global_dof)
//               && !already_visited.insert(global_dof))
//           {
//             continue;
//           }

//           // Restrict if not already done
//           if (!interpolated)
//           {
//             _g->restrict(data.w.data(), *_function_space->element(), cell,
//                          coordinate_dofs);
//             interpolated = true;
//           }

//           // Set boundary value
//           const PetscScalar value = data.w[i];
//           boundary_values[global_dof] = value;
//         }
//       }
//     }
//   }
// }
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
  log::log(TRACE,
           "Computing facets, needed for geometric application of boundary "
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
// void DirichletBC::compute_bc_pointwise(Map& boundary_values,
//                                        LocalData& data) const
// {
//   if (!_user_sub_domain)
//     throw std::runtime_error("A SubDomain is required for pointwise
//     search");

//   assert(_g);

//   // Get mesh, dofmap and element
//   assert(_function_space);
//   assert(_function_space->dofmap());
//   assert(_function_space->element());
//   assert(_function_space->mesh());
//   const GenericDofMap& dofmap = *_function_space->dofmap();
//   const FiniteElement& element = *_function_space->element();
//   const mesh::Mesh& mesh = *_function_space->mesh();
//   const std::size_t gdim = mesh.geometry().dim();

//   // Speed up the computations by only visiting (most) dofs once
//   common::RangedIndexSet already_visited(
//       dofmap.is_view() ? std::array<std::int64_t, 2>{{0, 0}}
//                        : dofmap.ownership_range());

//   // Allocate space using cached size
//   if (_num_dofs > 0)
//     boundary_values.reserve(boundary_values.size() + _num_dofs);

//   // Get dof coordinates on reference element
//   const EigenRowArrayXXd& X = element.dof_reference_coordinates();

//   // Get coordinate mapping
//   if (!mesh.geometry().coord_mapping)
//   {
//     throw std::runtime_error(
//         "CoordinateMapping has not been attached to mesh.");
//   }
//   const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

//   // Iterate over cells
//   EigenRowArrayXXd coordinate_dofs;
//   if (MPI::max(mesh.mpi_comm(), _cells_to_localdofs.size()) == 0)
//   {
//     // First time around all cells must be iterated over.  Create map
//     // from cells attached to boundary to local dofs.
//     for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
//     {
//       // Get dof coordinates
//       coordinate_dofs.resize(cell.num_vertices(), gdim);
//       cell.get_coordinate_dofs(coordinate_dofs);

//       // Tabulate coordinates of dofs on cell
//       cmap.compute_physical_coordinates(data.coordinates, X,
//       coordinate_dofs);

//       // Tabulate dofs on cell
//       auto cell_dofs = dofmap.cell_dofs(cell.index());

//       // Interpolate function only once and only on cells where
//       // necessary
//       bool already_interpolated = false;

//       // Loop all dofs on cell
//       std::vector<std::size_t> dofs;
//       for (std::size_t i = 0; i < dofmap.num_element_dofs(cell.index());
//       ++i)
//       {
//         const std::size_t global_dof = cell_dofs[i];

//         // Skip already checked dofs
//         if (already_visited.in_range(global_dof)
//             && !already_visited.insert(global_dof))
//         {
//           continue;
//         }

//         // Check if the coordinates are part of the sub domain (calls
//         // user-defined 'inside' function)
//         if (!_user_sub_domain->inside(data.coordinates.row(i), false)[0])
//           continue;

//         if (!already_interpolated)
//         {
//           already_interpolated = true;

//           // Restrict coefficient to cell
//           _g->restrict(data.w.data(), *_function_space->element(), cell,
//                        coordinate_dofs);

//           // Put cell index in storage for next time function is
//           // called
//           _cells_to_localdofs.insert(std::make_pair(cell.index(), dofs));
//         }

//         // Add local dof to map
//         _cells_to_localdofs[cell.index()].push_back(i);

//         // Set boundary value
//         const PetscScalar value = data.w[i];
//         boundary_values[global_dof] = value;
//       }
//     }
//   }
//   else
//   {
//     // Loop over cells that contain dofs on boundary
//     std::map<std::size_t, std::vector<std::size_t>>::const_iterator it;
//     for (it = _cells_to_localdofs.begin(); it != _cells_to_localdofs.end();
//          ++it)
//     {
//       // Get cell
//       const mesh::Cell cell(mesh, it->first);

//       // Get dof coordinates
//       coordinate_dofs.resize(cell.num_vertices(), gdim);
//       cell.get_coordinate_dofs(coordinate_dofs);

//       // Tabulate coordinates of dofs on cell
//       cmap.compute_physical_coordinates(data.coordinates, X,
//       coordinate_dofs);

//       // Restrict coefficient to cell
//       _g->restrict(data.w.data(), *_function_space->element(), cell,
//                    coordinate_dofs);

//       // Tabulate dofs on cell
//       auto cell_dofs = dofmap.cell_dofs(cell.index());

//       // Loop dofs on boundary of cell
//       for (std::size_t i = 0; i < it->second.size(); ++i)
//       {
//         const std::size_t local_dof = it->second[i];
//         const std::size_t global_dof = cell_dofs[local_dof];

//         // Set boundary value
//         const PetscScalar value = data.w[local_dof];
//         boundary_values[global_dof] = value;
//       }
//     }
//   }

//   // Store num of bc dofs for better performance next time
//   _num_dofs = boundary_values.size();
// }
//-----------------------------------------------------------------------------
