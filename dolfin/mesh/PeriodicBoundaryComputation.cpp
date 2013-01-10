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
#include <utility>
#include <vector>
#include <boost/unordered_map.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>
#include "PeriodicBoundaryComputation.h"

using namespace dolfin;

// Comparison operator for hashing coordinates. Note that two
// coordinates are considered equal if equal to within round-off.
struct lt_coordinate
{
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

      if (xx < (yy - DOLFIN_EPS))
        return true;
      else if (xx > (yy + DOLFIN_EPS))
        return false;
    }
    return false;
  }
};

//-----------------------------------------------------------------------------
std::map<std::size_t, std::pair<std::size_t, std::size_t> >
   PeriodicBoundaryComputation::compute_periodic_facet_pairs(const Mesh& mesh,
                                                const SubDomain& sub_domain)
{
  // Get topological and geometric dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  // Make sure we have the facet - cell connectivity
  mesh.init(tdim - 1, tdim);

  // Number facets globally

  // Arrays used for mapping coordinates
  std::vector<double> x(gdim);
  std::vector<double> y(gdim);

  // Wrap x and y (Array view of x and y)
  Array<double> _x(gdim, x.data());
  Array<double> _y(gdim, y.data());

  std::vector<std::size_t> slave_facets;
  std::vector<std::vector<double> > slave_mapped_midpoints;

  // Min/max coordinates of facet midpoints [min_x, max_x]. Used to
  // build bounding box of all master facet midpoints on this process
  std::vector<double> x_min_max;

  // Map from master facet midpoint coordinate to local facet index
  std::map<std::vector<double>, std::size_t, lt_coordinate> master_midpoint_to_facet_index;

  // Iterate over facets to find master/slave facets
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (!facet->exterior())
      continue;

    // Get midpoint of facet
    const Point midpoint = facet->midpoint();
    std::copy(midpoint.coordinates(), midpoint.coordinates() + gdim,
              x.begin());

    // Check if facet lies on a 'master' or 'slave' boundary
    if (sub_domain.inside(_x, true))
    {
      // Build bounding box data for master facet midpoints
      if (x_min_max.empty())
      {
        x_min_max = x;
        x_min_max.insert(x_min_max.end(), x.begin(), x.end());
      }
      for (std::size_t i = 0; i < gdim; ++i)
      {
       x_min_max[i]        = std::min(x_min_max[i], x[i]);
       x_min_max[i + gdim] = std::max(x_min_max[i + gdim], x[i]);
      }

      // Insert (midpoint coordinates, local index) into map
      master_midpoint_to_facet_index.insert(std::make_pair(x, facet->index()));
    }
    else
    {
      // Get mapped midpoint (y) of slave facet
      sub_domain.map(_x, _y);

      // Check if facet lies on a 'slave' boundary
      if (sub_domain.inside(_y, true))
      {
        // Store slave local index and midpoint coordinates
        slave_facets.push_back(facet->index());
        slave_mapped_midpoints.push_back(y);
      }

    }
  }

  // Communicate bounding boxes for master facets
  std::vector<std::vector<double> > bounding_boxes;
  MPI::all_gather(x_min_max, bounding_boxes);

  // Number of MPI processes
  std::size_t num_processes = MPI::num_processes();

  // Build send buffer of mapped slave midpoint coordinate to processes
  // that may own the master facet
  std::vector<std::vector<double> > slave_mapped_midpoints_send(num_processes);
  std::vector<std::vector<std::size_t> > sent_slave_indices(num_processes);
  for (std::size_t i = 0; i < slave_facets.size(); ++i)
  {
    //cout << "Going over facet: " << i << ", " << slave_facets.size() << endl;
    for (std::size_t p = 0; p < num_processes; ++p)
    {
      // Slave mapped midpoints from process p
      std::vector<double>& slave_mapped_midpoints_send_p = slave_mapped_midpoints_send[p];

      // Check if mapped slave falls within master facet bounding box
      // on process p
      if (in_bounding_box(slave_mapped_midpoints[i], bounding_boxes[p]))
      {
        sent_slave_indices[p].push_back(slave_facets[i]);
        slave_mapped_midpoints_send_p.insert(slave_mapped_midpoints_send_p.end(),
                                             slave_mapped_midpoints[i].begin(),
                                             slave_mapped_midpoints[i].end());
      }
    }
  }

  // Send slave midpoints to possible owners of correspoding master facet
  std::vector<std::vector<double> > slave_mapped_midpoints_recv;
  MPI::all_to_all(slave_mapped_midpoints_send, slave_mapped_midpoints_recv);
  dolfin_assert(slave_mapped_midpoints_recv.size() == num_processes);

  // Check if this process owns the master facet for a reveived (mapped)
  std::vector<double> midpoint(gdim);
  std::vector<std::vector<std::size_t> > master_local_facet(num_processes);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<double>& slave_mapped_midpoints_p = slave_mapped_midpoints_recv[p];
    for (std::size_t i = 0; i < slave_mapped_midpoints_p.size(); i += gdim)
    {
      // Unpack rceived mapped slave midpoint coordinate
      std::copy(&slave_mapped_midpoints_p[i],
                &slave_mapped_midpoints_p[i] + gdim, midpoint.begin());

      // Check is this process has a master facet that is paired with
      // a received slave facet
      std::map<std::vector<double>, std::size_t, lt_coordinate>::const_iterator
        it = master_midpoint_to_facet_index.find(midpoint);

      // If this process owns the master, insert master facet index,
      // else insert std::numeric_limits<std::size_t>::max()
      if (it !=  master_midpoint_to_facet_index.end())
        master_local_facet[p].push_back(it->second);
      else
        master_local_facet[p].push_back(std::numeric_limits<std::size_t>::max());
    }
  }

  // Send local index of master facet back to owner of slave facet
  std::vector<std::vector<std::size_t> > master_facet_local_index_recv;
  MPI::all_to_all(master_local_facet,  master_facet_local_index_recv);

  // Build map from slave facets on this process to master facet (local
  // facet index, process owner)
  std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facet;
  std::size_t num_local_slave_facets = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<std::size_t> master_facet_index_p = master_facet_local_index_recv[p];
    const std::vector<std::size_t> sent_slaves_p = sent_slave_indices[p];
    dolfin_assert(master_facet_index_p.size() == sent_slaves_p.size());

    for (std::size_t i = 0; i < master_facet_index_p.size(); ++i)
    {
      if (master_facet_index_p[i] < std::numeric_limits<std::size_t>::max())
      {
        ++num_local_slave_facets;
        if (!slave_to_master_facet.insert(std::make_pair(sent_slaves_p[i],
                                     std::make_pair(p, master_facet_index_p[i]))).second)
        {
          dolfin_error("PeriodicDomain.cpp",
                       "build peridic master-slave mapping",
                       "More than one master facts for slave facet");
        }
      }
    }
  }

  // Number global master and slave facets
  const std::size_t num_global_master_facets = MPI::sum(master_midpoint_to_facet_index.size());
  const std::size_t num_global_slave_facets = MPI::sum(num_local_slave_facets);

  // Check that number of global master and slave facets match
  if (num_global_master_facets != num_global_slave_facets)
  {
    dolfin_error("PeriodicDomain.cpp",
                 "global number of slave and master facets",
                 "Number of slave and master facets is not equal");
  }

  return slave_to_master_facet;
}
//-----------------------------------------------------------------------------
bool PeriodicBoundaryComputation::in_bounding_box(const std::vector<double>& point,
                                     const std::vector<double>& bounding_box)
{
  // Return false if bounding box is empty
  if (bounding_box.empty())
    return false;

  const std::size_t gdim = point.size();
  dolfin_assert(bounding_box.size() == 2*gdim);
  for (std::size_t i = 0; i < gdim; ++i)
  {
    if (!(point[i] >= (bounding_box[i] - DOLFIN_EPS)
      && point[i] <= (bounding_box[gdim + i] + DOLFIN_EPS)))
    {
      return false;
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
