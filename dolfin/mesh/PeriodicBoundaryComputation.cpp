// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-01-10
// Last changed:

#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntityIterator.h"
#include "SubDomain.h"
#include "PeriodicBoundaryComputation.h"

using namespace dolfin;

// Comparison operator for hashing coordinates. Note that two
// coordinates are considered equal if equal to within specified
// tolerance.
struct lt_coordinate
{
  lt_coordinate(double tolerance) : TOL(tolerance) {}

  bool operator() (const std::vector<double>& x,
                   const std::vector<double>& y) const
  {
    std::size_t n = std::max(x.size(), y.size());
    for (std::size_t i = 0; i < n; ++i)
    {
      double xx = 0.0;
      double yy = 0.0;
      if (i < x.size())
        xx = x[i];
      if (i < y.size())
        yy = y[i];

      if (xx < (yy - TOL))
        return true;
      else if (xx > (yy + TOL))
        return false;
    }
    return false;
  }

  // Tolerance
  const double TOL;

};

//-----------------------------------------------------------------------------
std::map<unsigned int, std::pair<unsigned int, unsigned int>>
  PeriodicBoundaryComputation::compute_periodic_pairs(const Mesh& mesh,
                                                      const SubDomain& sub_domain,
                                                      const std::size_t dim)
{
  // MPI communication
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get geometric and topological dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  // Arrays used for mapping coordinates
  std::vector<double> x(gdim);
  std::vector<double> y(gdim);

  // Wrap x and y (Array view of x and y)
  Array<double> _x(gdim, x.data());
  Array<double> _y(gdim, y.data());

  std::vector<std::size_t> slave_entities;
  std::vector<std::vector<double>> slave_mapped_coords;

  // Min/max coordinates of facet midpoints [min_x, max_x]. Used to
  // build bounding box of all master entity midpoints on this process
  std::vector<double> x_min_max;

  // Map from master entity midpoint coordinate to local facet index
  std::map<std::vector<double>, unsigned int, lt_coordinate>
    master_coord_to_entity_index((lt_coordinate(sub_domain.map_tolerance)));

  // Initialise facet-cell connectivity
  mesh.init(tdim - 1, tdim);
  mesh.init(dim);

  std::vector<bool> visited(mesh.num_entities(dim), false);
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Consider boundary entities only
    const bool global_exterior_facet = (f->num_global_entities(tdim) == 1);
    if (global_exterior_facet)
    {
      for (MeshEntityIterator e(*f, dim); !e.end(); ++e)
      {
        // Avoid visiting entities more than once
        if (visited[e->index()])
          continue;
        else
          visited[e->index()] = true;

        // Copy entity coordinate
        const Point midpoint = e->midpoint();
        std::copy(midpoint.coordinates(), midpoint.coordinates() + gdim,
                  x.begin());

        // Check if entity lies on a 'master' or 'slave' boundary
        if (sub_domain.inside(_x, true))
        {
          // Build bounding box data for master entity midpoints
          if (x_min_max.empty())
          {
            x_min_max = x;
            x_min_max.insert(x_min_max.end(), x.begin(), x.end());
          }

          for (std::size_t i = 0; i < gdim; ++i)
          {
            x_min_max[i]        = std::min(x_min_max[i], x[i]);
            x_min_max[gdim + i] = std::max(x_min_max[gdim + i], x[i]);
          }

          // Insert (midpoint coordinates, local index) into map
          master_coord_to_entity_index.insert({x, e->index()});
        }
        else
        {
          // Let's check the user is going to map all coordinates
          for (size_t i = 0; i < y.size(); i++)
          {
            y[i] = std::numeric_limits<double>::quiet_NaN();
          }

          // Get mapped midpoint (y) of slave entity
          sub_domain.map(_x, _y);

          // Check for NaNs after the map
          for (size_t i = 0; i < y.size(); i++)
          {
            if (std::isnan(y[i]))
            {
              dolfin_error("PeriodicBoundaryComputation.cpp",
                           "periodic boundary mapping",
                           "Need to set coordinate %d in sub_domain.map", i);
            }
          }

          // Check if entity lies on a 'slave' boundary
          if (sub_domain.inside(_y, true))
          {
            // Store slave local index and midpoint coordinates
            slave_entities.push_back(e->index());
            slave_mapped_coords.push_back(y);
          }
        }
      }
    }
  }

  // Communicate bounding boxes for master entities
  std::vector<std::vector<double>> bounding_boxes;
  MPI::all_gather(mpi_comm, x_min_max, bounding_boxes);

  // Number of MPI processes
  std::size_t num_processes = MPI::size(mpi_comm);

  // Build send buffer of mapped slave midpoint coordinate to
  // processes that may own the master entity
  std::vector<std::vector<double>> slave_mapped_coords_send(num_processes);
  std::vector<std::vector<unsigned int>> sent_slave_indices(num_processes);
  for (std::size_t i = 0; i < slave_entities.size(); ++i)
  {
    for (std::size_t p = 0; p < num_processes; ++p)
    {
      // Slave mapped coordinates from process p
      std::vector<double>& slave_mapped_coords_send_p
        = slave_mapped_coords_send[p];

      // Check if mapped slave falls within master entity bounding box
      // on process p
      if (in_bounding_box(slave_mapped_coords[i], bounding_boxes[p],
                          sub_domain.map_tolerance))
      {
        sent_slave_indices[p].push_back(slave_entities[i]);
        slave_mapped_coords_send_p.insert(slave_mapped_coords_send_p.end(),
                                          slave_mapped_coords[i].begin(),
                                          slave_mapped_coords[i].end());
      }
    }
  }

  // Send slave midpoints to possible owners of corresponding master
  // entity
  std::vector<std::vector<double>> slave_mapped_coords_recv;
  MPI::all_to_all(mpi_comm,  slave_mapped_coords_send,
                  slave_mapped_coords_recv);
  dolfin_assert(slave_mapped_coords_recv.size() == num_processes);

  // Check if this process owns the master facet for a received (mapped)
  // slave
  std::vector<double> coordinates(gdim);
  std::vector<std::vector<unsigned int>> master_local_entity(num_processes);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<double>& slave_mapped_coords_p
      = slave_mapped_coords_recv[p];
    for (std::size_t i = 0; i < slave_mapped_coords_p.size(); i += gdim)
    {
      // Unpack received mapped slave midpoint coordinate
      std::copy(&slave_mapped_coords_p[i],
                &slave_mapped_coords_p[i] + gdim, coordinates.begin());

      // Check is this process has a master entity that is paired with
      // a received slave entity
      std::map<std::vector<double>, unsigned int>::const_iterator
        it = master_coord_to_entity_index.find(coordinates);

      // If this process owns the master, insert master entity index,
      // else insert std::numeric_limits<unsigned int>::max()
      if (it !=  master_coord_to_entity_index.end())
        master_local_entity[p].push_back(it->second);
      else
        master_local_entity[p].push_back(std::numeric_limits<unsigned int>::max());
    }
  }

  // Send local index of master entity back to owner of slave entity
  std::vector<std::vector<unsigned int>> master_entity_local_index_recv;
  MPI::all_to_all(mpi_comm, master_local_entity,
                  master_entity_local_index_recv);

  // Build map from slave facets on this process to master facet (local
  // facet index, process owner)
  std::map<unsigned int, std::pair<unsigned int, unsigned int>>
    slave_to_master_entity;
  std::size_t num_local_slave_entities = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<unsigned int> master_entity_index_p
      = master_entity_local_index_recv[p];
    const std::vector<unsigned int> sent_slaves_p = sent_slave_indices[p];
    dolfin_assert(master_entity_index_p.size() == sent_slaves_p.size());

    for (std::size_t i = 0; i < master_entity_index_p.size(); ++i)
    {
      if (master_entity_index_p[i] < std::numeric_limits<unsigned int>::max())
      {
        ++num_local_slave_entities;
        slave_to_master_entity.insert({sent_slaves_p[i], {p, master_entity_index_p[i]}});
      }
    }
  }

  return slave_to_master_entity;
}
//-----------------------------------------------------------------------------
MeshFunction<std::size_t>
PeriodicBoundaryComputation::masters_slaves(std::shared_ptr<const Mesh> mesh,
                                            const SubDomain& sub_domain,
                                            const std::size_t dim)
{
  dolfin_assert(mesh);

  // Create MeshFunction and initialise to zero
  MeshFunction<std::size_t> mf(*mesh, dim, 0);

  // Compute marker
  const std::map<unsigned int, std::pair<unsigned int, unsigned int>>
    slaves = compute_periodic_pairs(*mesh, sub_domain, dim);

  // Mark master and slaves, and pack off-process masters to send
  std::vector<std::vector<std::size_t>>
    master_dofs_send(MPI::size(mesh->mpi_comm()));
  std::map<unsigned int,
           std::pair<unsigned int, unsigned int>>::const_iterator slave;
  for (slave = slaves.begin(); slave != slaves.end(); ++slave)
  {
    // Set slave
    mf[slave->first] = 2;

    // Pack master entity to send to all sharing processes
    dolfin_assert(slave->second.first < master_dofs_send.size());
    master_dofs_send[slave->second.first].push_back(slave->second.second);
  }

  // Send/receive master entities
  std::vector<std::vector<std::size_t>> master_dofs_recv;
  MPI::all_to_all(mesh->mpi_comm(), master_dofs_send, master_dofs_recv);

  // Build list of sharing processes
  std::unordered_map<unsigned int,
                     std::vector<std::pair<unsigned int, unsigned int>>>
    shared_entities_map
    = DistributedMeshTools::compute_shared_entities(*mesh, dim);
  std::unordered_map<unsigned int,
                     std::vector<std::pair<unsigned int,
                                           unsigned int>>>::const_iterator e;
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
    shared_entities(mesh->num_entities(dim));
  for (e = shared_entities_map.begin(); e != shared_entities_map.end(); ++e)
  {
    dolfin_assert(e->first < shared_entities.size());
    shared_entities[e->first] = e->second;
  }

  // Mark and pack master to send to all sharing processes
  master_dofs_send.clear();
  master_dofs_send.resize(MPI::size(mesh->mpi_comm()));
  for (std::size_t p = 0; p < master_dofs_recv.size(); ++p)
  {
    for (std::size_t i = 0; i < master_dofs_recv[p].size(); ++i)
    {
      // Get local index
      const std::size_t local_index = master_dofs_recv[p][i];

      // Mark locally
      mf[local_index] = 1;

      // Pack to send to sharing processes
      const std::vector<std::pair<unsigned int, unsigned int>> sharing
        = shared_entities[local_index] ;
      for (std::size_t j = 0; j < sharing.size(); ++j)
      {
        dolfin_assert(sharing[j].first < master_dofs_send.size());
        master_dofs_send[sharing[j].first].push_back(sharing[j].second);
      }
    }
  }

  // Send/receive master entities
  MPI::all_to_all(mesh->mpi_comm(), master_dofs_send, master_dofs_recv);

  // Mark master entities in mesh function
  for (std::size_t i = 0; i < master_dofs_recv.size(); ++i)
    for (std::size_t j = 0; j < master_dofs_recv[i].size(); ++j)
      mf[master_dofs_recv[i][j]] = 1;

  return mf;
}
//-----------------------------------------------------------------------------
bool
PeriodicBoundaryComputation::in_bounding_box(const std::vector<double>& point,
                                       const std::vector<double>& bounding_box,
                                       const double tol)
{
  // Return false if bounding box is empty
  if (bounding_box.empty())
    return false;

  const std::size_t gdim = point.size();
  dolfin_assert(bounding_box.size() == 2*gdim);
  for (std::size_t i = 0; i < gdim; ++i)
  {
    if (!(point[i] >= (bounding_box[i] - tol)
          && point[i] <= (bounding_box[gdim + i] + tol)))
    {
      return false;
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
