// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells 2007
// Modified by Johan Hake 2009
//
// First added:  2007-07-08
// Last changed: 2011-11-14

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
#include "PeriodicDomain.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void PeriodicDomain::compute_periodic_facet_pairs(const Mesh& mesh,
                                                  const SubDomain& sub_domain)
{
  error("PeriodicDomain::compute_periodic_facet_pairs is under development");

  #ifdef HAS_MPI

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
  Array<double> _y(gdim, x.data());

  //typedef std::map<std::vector<double>, dof_pair, lt_coordinate> coordinate_map;

  std::vector<std::size_t> master_facets;
  std::vector<std::size_t> slave_facets;

  // Iterate over facets to find master/slave facets
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (!facet->exterior())
      continue;

    // Get mid-side of vertex
    const Point midpoint = facet->midpoint();
    std::copy(midpoint.coordinates(), midpoint.coordinates() + gdim,
              x.begin());

    // Check if facet lies on a 'master' or 'slave' boundary
    if (sub_domain.inside(_x, true))
      master_facets.push_back(facet->global_index());
    else
    {
      sub_domain.map(_x, _y);
      if (sub_domain.inside(_y, true))
        slave_facets.push_back(facet->global_index());
    }
  }

  // Get number of 'master' facets
  std::size_t num_master_facets = MPI::sum(master_facets.size());

  // Ownership ranges for master facets vertices for all processes
  const std::size_t num_processes = MPI::num_processes();
  std::vector<std::size_t> ownership;
  for (std::size_t p = 0; p < num_processes; ++p)
    ownership.push_back(MPI::local_range(p, num_master_facets, num_processes).second);

  // Pack master facets to send to 'owner'
  std::vector<std::vector<std::size_t> > send_buffer(num_processes);
  std::size_t p = 0;
  for (std::size_t i = 0; i < master_facets.size(); ++i)
  {
    while (master_facets[i] >= ownership[p])
      ++p;

    dolfin_assert(p < send_buffer.size());
    send_buffer[p].push_back(master_facets[i]);
  }

  // Create receive buffer
  std::vector<std::vector<std::size_t> > recv_buffer(num_processes);

  // Communicate data
  MPICommunicator mpi_comm;
  boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
  boost::mpi::all_to_all(comm, send_buffer, recv_buffer);

  // Send slave facet to
  //. . . .

  #endif
}
//-----------------------------------------------------------------------------
