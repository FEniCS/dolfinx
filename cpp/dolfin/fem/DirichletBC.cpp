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

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
                         std::shared_ptr<const function::Function> g,
                         const mesh::SubDomain& sub_domain, Method method,
                         bool check_midpoint)
    : _function_space(V), _g(g), _method(method), _num_dofs(0)
{
  check_data();

  // FIXME: This can be made more efficient, we should be able to
  //        extract the facets without first creating a
  //        mesh::MeshFunction on the entire mesh and then extracting
  //        the subset. This is done mainly for convenience (we may
  //        reuse mark() in SubDomain).

  assert(V);
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
  for (auto& facet : mesh::MeshRange<mesh::Facet>(*mesh))
  {
    if (domain[facet] == 0)
      _facets.push_back(facet.index());
  }
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(
    std::shared_ptr<const function::FunctionSpace> V,
    std::shared_ptr<const function::Function> g,
    std::pair<std::weak_ptr<const mesh::MeshFunction<std::size_t>>, std::size_t>
        sub_domain,
    Method method)
    : _function_space(V), _g(g), _method(method), _num_dofs(0)
{
  check_data();

  // Get mesh
  assert(V);
  assert(V->mesh());
  const mesh::Mesh& mesh = *V->mesh();

  // Initialise facet - cell connectivity
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1, D);

  // Build set of boundary facets
  assert(!sub_domain.first.expired());
  auto domain = sub_domain.first.lock();
  const std::size_t index = sub_domain.second;
  for (auto& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    if ((*domain)[facet] == index)
      _facets.push_back(facet.index());
  }
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
                         std::shared_ptr<const function::Function> g,
                         const std::vector<std::size_t>& facet_indices,
                         Method method)
    : _function_space(V), _g(g), _method(method), _num_dofs(0),
      _facets(facet_indices)
{
  check_data();
}
//-----------------------------------------------------------------------------
void DirichletBC::gather(Map& boundary_values) const
{
  common::Timer timer("DirichletBC gather");

  assert(_function_space->mesh());
  MPI_Comm mpi_comm = _function_space->mesh()->mpi_comm();
  std::size_t comm_size = MPI::size(mpi_comm);

  // Get dofmap
  assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *_function_space->dofmap();
  const auto& shared_nodes = dofmap.shared_nodes();
  const int bs = dofmap.block_size();
  assert(dofmap.index_map());

  // Create list of boundary values to send to each processor
  std::vector<std::vector<std::size_t>> proc_map0(comm_size);
  std::vector<std::vector<PetscScalar>> proc_map1(comm_size);
  for (Map::const_iterator bv = boundary_values.begin();
       bv != boundary_values.end(); ++bv)
  {
    // If the boundary value is attached to a shared dof, add it to the
    // list of boundary values for each of the processors that share it
    const std::div_t div = std::div(bv->first, bs);
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
        proc_map0[*proc].push_back(bs * global_node + component);
        proc_map1[*proc].push_back(bv->second);
      }
    }
  }

  // Distribute the lists between neighbours
  std::vector<std::size_t> received_bvc0;
  std::vector<PetscScalar> received_bvc1;
  MPI::all_to_all(mpi_comm, proc_map0, received_bvc0);
  MPI::all_to_all(mpi_comm, proc_map1, received_bvc1);
  assert(received_bvc0.size() == received_bvc1.size());

  const std::int64_t n0 = dofmap.ownership_range()[0];
  const std::int64_t n1 = dofmap.ownership_range()[1];
  const std::int64_t owned_size = n1 - n0;

  // Reserve space
  const std::size_t num_dofs = boundary_values.size() + received_bvc0.size();
  boundary_values.reserve(num_dofs);

  // Add the received boundary values to the local boundary values
  std::vector<std::pair<std::int64_t, PetscScalar>> _vec(received_bvc0.size());
  for (std::size_t i = 0; i < _vec.size(); ++i)
  {
    // Global dof index
    _vec[i].first = received_bvc0[i];

    // Convert to local (process) dof index
    if (_vec[i].first >= n0 && _vec[i].first < n1)
    {
      // Case 0: dof is owned by this process
      _vec[i].first = received_bvc0[i] - n0;
    }
    else
    {
      const std::imaxdiv_t div = std::imaxdiv(_vec[i].first, bs);
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
        _vec[i].first = owned_size + bs * pos + component;
      }
    }
    _vec[i].second = received_bvc1[i];
  }

  boundary_values.insert(_vec.begin(), _vec.end());
}
//-----------------------------------------------------------------------------
void DirichletBC::get_boundary_values(Map& boundary_values) const
{
  // Create local data
  assert(_function_space);
  LocalData data(*_function_space);

  // Compute dofs and values
  if (_method == Method::topological)
    compute_bc_topological(boundary_values, data);
  else if (_method == Method::geometric)
    compute_bc_geometric(boundary_values, data);
  else if (_method == Method::pointwise)
    compute_bc_pointwise(boundary_values, data);
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& DirichletBC::markers() const { return _facets; }
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
DirichletBC::Method DirichletBC::method() const { return _method; }
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1> DirichletBC::dof_indices() const
{
  // FIXME: Optimise this operation, and consider caching
  Map boundary_values;
  get_boundary_values(boundary_values);

  // FIXME: Eliminate comm
  assert(_function_space->mesh());
  MPI_Comm mpi_comm = _function_space->mesh()->mpi_comm();
  if (MPI::size(mpi_comm) > 1
      and this->method() != DirichletBC::Method::pointwise)
    this->gather(boundary_values);

  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(boundary_values.size());
  std::size_t i = 0;
  for (auto& bc : boundary_values)
    dofs[i++] = bc.first;

  return dofs;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<PetscInt, Eigen::Dynamic, 1>,
          Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
DirichletBC::bcs() const
{
  // FIXME: Optimise this operation, and consider caching
  Map boundary_values;
  get_boundary_values(boundary_values);

  // FIXME: Eliminate comm
  assert(_function_space->mesh());
  MPI_Comm mpi_comm = _function_space->mesh()->mpi_comm();
  if (MPI::size(mpi_comm) > 1
      and this->method() != DirichletBC::Method::pointwise)
    this->gather(boundary_values);

  Eigen::Array<PetscInt, Eigen::Dynamic, 1> indices(boundary_values.size());
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> values(boundary_values.size());
  std::size_t i = 0;
  for (auto& bc : boundary_values)
  {
    indices[i] = bc.first;
    values[i++] = bc.second;
  }

  return std::make_pair(std::move(indices), std::move(values));
}
//-----------------------------------------------------------------------------
void DirichletBC::check_data() const
{
  assert(_g);
  assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

  // Check for common errors, message below might be cryptic
  if (_g->value_rank() == 0 && element.value_rank() == 1)
  {
    throw std::runtime_error("Expecting a vector-valued boundary value but "
                             "given function is scalar");
  }

  if (_g->value_rank() == 1 && element.value_rank() == 0)
  {
    throw std::runtime_error("Expecting a scalar boundary value but given "
                             "function is vector-valued");
  }

  // Check that value shape of boundary value
  if (_g->value_rank() != element.value_rank())
  {
    throw std::runtime_error("Illegal value rank "
                             + std::to_string(_g->value_rank()) + ", expecting "
                             + std::to_string(element.value_rank()));
  }

  for (std::size_t i = 0; i < _g->value_rank(); i++)
  {
    if (_g->value_dimension(i) != element.value_dimension(i))
    {
      throw std::runtime_error(
          "Illegal value dimension " + std::to_string(_g->value_dimension(i))
          + ", expecting " + std::to_string(element.value_dimension(i)));
    }
  }

  // Check user supplied mesh::MeshFunction
  // if (_user_mesh_function)
  // {
  //   // Check that mesh::Meshfunction is initialised
  //   if (!_user_mesh_function->mesh())
  //     throw std::runtime_error("User mesh::MeshFunction is not initialized");

  //   // Check that mesh::Meshfunction is a mesh::FacetFunction
  //   const std::size_t tdim = _user_mesh_function->mesh()->topology().dim();
  //   if (_user_mesh_function->dim() != tdim - 1)
  //   {
  //     throw std::runtime_error("User mesh::MeshFunction is not a facet "
  //                              "mesh::MeshFunction (dimension is wrong)");
  //   }

  //   // Check that mesh::Meshfunction and function::FunctionSpace meshes match
  //   assert(_function_space->mesh());
  //   if (_user_mesh_function->mesh()->id() != _function_space->mesh()->id())
  //   {
  //     throw std::runtime_error("User mesh::MeshFunction and "
  //                              "function::FunctionSpace meshes are
  //                              different");
  //   }
  // }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_topological(Map& boundary_values,
                                         LocalData& data) const
{
  assert(_function_space);
  assert(_g);

  // Get mesh and dofmap
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // Special case
  if (_facets.empty())
  {
    if (MPI::size(mesh.mpi_comm()) == 1)
      log::warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get dofmap
  assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Topological and geometric dimension
  const std::size_t D = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Initialise facet-cell connectivity
  mesh.init(D);
  mesh.init(D - 1, D);

  // Coordinate dofs
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Allocate space
  const std::size_t num_facet_dofs = dofmap.num_entity_closure_dofs(D - 1);
  boundary_values.reserve(boundary_values.size()
                          + _facets.size() * num_facet_dofs);

  // Build local dofs for each facet
  const mesh::CellType& cell_type = mesh.type();
  std::vector<Eigen::Array<int, Eigen::Dynamic, 1>> facet_dofs;
  for (std::size_t i = 0; i < cell_type.num_entities(D - 1); ++i)
    facet_dofs.push_back(dofmap.tabulate_entity_closure_dofs(D - 1, i));

  // Iterate over marked
  assert(_function_space->element());
  for (std::size_t f = 0; f < _facets.size(); ++f)
  {
    // Create facet
    const mesh::Facet facet(mesh, _facets[f]);

    // Get cell to which facet belongs
    assert(facet.num_entities(D) > 0);
    const std::size_t cell_index = facet.entities(D)[0];

    // Create attached cell
    const mesh::Cell cell(mesh, cell_index);

    // Get local index of facet with respect to the cell
    const size_t facet_local_index = cell.index(facet);

    // Get coordinate data and set local facet index
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);
    cell.local_facet = facet_local_index;

    // Restrict coefficient to cell
    _g->restrict(data.w.data(), *_function_space->element(), cell,
                 coordinate_dofs);

    // Tabulate dofs on cell
    auto cell_dofs = dofmap.cell_dofs(cell.index());

    // Pick values for facet
    for (std::size_t i = 0; i < num_facet_dofs; i++)
    {
      const std::size_t local_dof = cell_dofs[facet_dofs[facet_local_index][i]];
      const PetscScalar value = data.w[facet_dofs[facet_local_index][i]];
      boundary_values[local_dof] = value;
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_geometric(Map& boundary_values,
                                       LocalData& data) const
{
  assert(_function_space);
  assert(_function_space->element());
  assert(_g);

  // Get mesh
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // Extract the list of facets where the BC *might* be applied
  // init_facets(mesh.mpi_comm());

  // Special case
  if (_facets.empty())
  {
    if (MPI::size(mesh.mpi_comm()) == 1)
      log::warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get dofmap
  assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Get finite element
  assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

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
  const std::size_t D = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Allocate space using cached size
  if (_num_dofs > 0)
    boundary_values.reserve(boundary_values.size() + _num_dofs);

  // Get dof coordinates on reference element
  const EigenRowArrayXXd& X = element.dof_reference_coordinates();

  // Get coordinate mapping
  if (!mesh.geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

  // Iterate over facets
  for (std::size_t f = 0; f < _facets.size(); ++f)
  {
    // Create facet
    const mesh::Facet facet(mesh, _facets[f]);

    // Create cell (get first attached cell)
    const mesh::Cell cell(mesh, facet.entities(D)[0]);

    // Get local index of facet with respect to the cell
    // const std::size_t local_facet = cell.index(facet);

    // Create vertex coordinate holder
    EigenRowArrayXXd coordinate_dofs;

    // Loop the vertices associated with the facet
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(facet))
    {
      // Loop the cells associated with the vertex
      for (auto& c : mesh::EntityRange<mesh::Cell>(vertex))
      {
        // FIXME: setting the local facet here looks wrong
        // c.local_facet = local_facet;
        coordinate_dofs.resize(cell.num_vertices(), gdim);
        c.get_coordinate_dofs(coordinate_dofs);

        bool tabulated = false;
        bool interpolated = false;

        // Tabulate dofs on cell
        auto cell_dofs = dofmap.cell_dofs(c.index());

        // Loop over all dofs on cell
        for (int i = 0; i < cell_dofs.size(); ++i)
        {
          const std::size_t global_dof = cell_dofs[i];

          // Tabulate coordinates if not already done
          if (!tabulated)
          {
            cmap.compute_physical_coordinates(data.coordinates, X,
                                              coordinate_dofs);
            tabulated = true;
          }

          // Check if the coordinates are on current facet and thus on
          // boundary
          if (!on_facet(data.coordinates.row(i), facet))
            continue;

          // Skip already checked dofs
          if (already_visited.in_range(global_dof)
              && !already_visited.insert(global_dof))
          {
            continue;
          }

          // Restrict if not already done
          if (!interpolated)
          {
            _g->restrict(data.w.data(), *_function_space->element(), cell,
                         coordinate_dofs);
            interpolated = true;
          }

          // Set boundary value
          const PetscScalar value = data.w[i];
          boundary_values[global_dof] = value;
        }
      }
    }
  }

  // Store num of bc dofs for better performance next time
  _num_dofs = boundary_values.size();
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_pointwise(Map& boundary_values,
                                       LocalData& data) const
{
  throw std::runtime_error(
      "DirichletBC::compute_bc_pointwise has not yet been updated.");
  // if (!_user_sub_domain)
  //   throw std::runtime_error("A SubDomain is required for pointwise search");

  // assert(_g);

  // // Get mesh, dofmap and element
  // assert(_function_space);
  // assert(_function_space->dofmap());
  // assert(_function_space->element());
  // assert(_function_space->mesh());
  // const GenericDofMap& dofmap = *_function_space->dofmap();
  // const FiniteElement& element = *_function_space->element();
  // const mesh::Mesh& mesh = *_function_space->mesh();
  // const std::size_t gdim = mesh.geometry().dim();

  // // Speed up the computations by only visiting (most) dofs once
  // common::RangedIndexSet already_visited(
  //     dofmap.is_view() ? std::array<std::int64_t, 2>{{0, 0}}
  //                      : dofmap.ownership_range());

  // // Allocate space using cached size
  // if (_num_dofs > 0)
  //   boundary_values.reserve(boundary_values.size() + _num_dofs);

  // // Get dof coordinates on reference element
  // const EigenRowArrayXXd& X = element.dof_reference_coordinates();

  // // Get coordinate mapping
  // if (!mesh.geometry().coord_mapping)
  // {
  //   throw std::runtime_error(
  //       "CoordinateMapping has not been attached to mesh.");
  // }
  // const CoordinateMapping& cmap = *mesh.geometry().coord_mapping;

  // // Iterate over cells
  // EigenRowArrayXXd coordinate_dofs;
  // if (MPI::max(mesh.mpi_comm(), _cells_to_localdofs.size()) == 0)
  // {
  //   // First time around all cells must be iterated over.  Create map
  //   // from cells attached to boundary to local dofs.
  //   for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  //   {
  //     // Get dof coordinates
  //     coordinate_dofs.resize(cell.num_vertices(), gdim);
  //     cell.get_coordinate_dofs(coordinate_dofs);

  //     // Tabulate coordinates of dofs on cell
  //     cmap.compute_physical_coordinates(data.coordinates, X,
  //     coordinate_dofs);

  //     // Tabulate dofs on cell
  //     auto cell_dofs = dofmap.cell_dofs(cell.index());

  //     // Interpolate function only once and only on cells where
  //     // necessary
  //     bool already_interpolated = false;

  //     // Loop all dofs on cell
  //     std::vector<std::size_t> dofs;
  //     for (std::size_t i = 0; i < dofmap.num_element_dofs(cell.index()); ++i)
  //     {
  //       const std::size_t global_dof = cell_dofs[i];

  //       // Skip already checked dofs
  //       if (already_visited.in_range(global_dof)
  //           && !already_visited.insert(global_dof))
  //       {
  //         continue;
  //       }

  //       // Check if the coordinates are part of the sub domain (calls
  //       // user-defined 'inside' function)
  //       if (!_user_sub_domain->inside(data.coordinates.row(i), false)[0])
  //         continue;

  //       if (!already_interpolated)
  //       {
  //         already_interpolated = true;

  //         // Restrict coefficient to cell
  //         _g->restrict(data.w.data(), *_function_space->element(), cell,
  //                      coordinate_dofs);

  //         // Put cell index in storage for next time function is
  //         // called
  //         _cells_to_localdofs.insert(std::make_pair(cell.index(), dofs));
  //       }

  //       // Add local dof to map
  //       _cells_to_localdofs[cell.index()].push_back(i);

  //       // Set boundary value
  //       const PetscScalar value = data.w[i];
  //       boundary_values[global_dof] = value;
  //     }
  //   }
  // }
  // else
  // {
  //   // Loop over cells that contain dofs on boundary
  //   std::map<std::size_t, std::vector<std::size_t>>::const_iterator it;
  //   for (it = _cells_to_localdofs.begin(); it != _cells_to_localdofs.end();
  //        ++it)
  //   {
  //     // Get cell
  //     const mesh::Cell cell(mesh, it->first);

  //     // Get dof coordinates
  //     coordinate_dofs.resize(cell.num_vertices(), gdim);
  //     cell.get_coordinate_dofs(coordinate_dofs);

  //     // Tabulate coordinates of dofs on cell
  //     cmap.compute_physical_coordinates(data.coordinates, X,
  //     coordinate_dofs);

  //     // Restrict coefficient to cell
  //     _g->restrict(data.w.data(), *_function_space->element(), cell,
  //                  coordinate_dofs);

  //     // Tabulate dofs on cell
  //     auto cell_dofs = dofmap.cell_dofs(cell.index());

  //     // Loop dofs on boundary of cell
  //     for (std::size_t i = 0; i < it->second.size(); ++i)
  //     {
  //       const std::size_t local_dof = it->second[i];
  //       const std::size_t global_dof = cell_dofs[local_dof];

  //       // Set boundary value
  //       const PetscScalar value = data.w[local_dof];
  //       boundary_values[global_dof] = value;
  //     }
  //   }
  // }

  // // Store num of bc dofs for better performance next time
  // _num_dofs = boundary_values.size();
}
//-----------------------------------------------------------------------------
bool DirichletBC::on_facet(
    const Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1>> coordinates,
    const mesh::Facet& facet) const
{
  // Check if the coordinates are on the same line as the line segment
  if (facet.dim() == 1)
  {
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
  // Check if the coordinates are in the same plane as the triangular
  // facet
  else if (facet.dim() == 2)
  {
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

    // Check if the sum of the area of the sub triangles is equal to
    // the total area of the facet
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
DirichletBC::LocalData::LocalData(const function::FunctionSpace& V)
    : w(V.dofmap()->max_element_dofs(), 0.0),
      // FIXME: the below should not be max_element_dofs! It should be fixed.
      coordinates(V.dofmap()->max_element_dofs(), V.mesh()->geometry().dim())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
