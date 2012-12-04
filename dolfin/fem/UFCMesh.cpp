// Copyright (C) 2012 Anders Logg
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
// First added:  2012-11-05
// Last changed: 2012-11-23

#include <vector>
#include <set>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include "UFCMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFCMesh::UFCMesh(const Mesh& mesh,
                 const MeshFunction<std::size_t>& domain_markers,
                 std::size_t domain) : ufc::mesh()
{
  // Check that we get cell markers, extend later
  if (domain_markers.dim() != mesh.topology().dim())
  {
    dolfin_error("UFCMesh.cpp",
                 "create mapping of degrees of freedom",
                 "Only cell-based restricted function spaces are currently supported. ");
  }

  // FIXME: Not yet working in parallel
  not_working_in_parallel("Restricted function space");

  // Set topological and geometric dimensions
  topological_dimension = mesh.topology().dim();
  geometric_dimension = mesh.geometry().dim();

  // Use sets to count the number of entities of each dimension
  std::vector<std::set<std::size_t> > entity_sets(topological_dimension + 1);

  // Count the number of entities of each dimension
  std::size_t num_cells = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Skip cell if it is not included in restriction
    if (domain_markers[cell->index()] != domain)
      continue;

    // Count cell
    num_cells += 1;

    // Count entities
    for (std::size_t d = 0; d < topological_dimension; d++)
    {
      // Skip if there are not entities of current dimension
      if (mesh.num_entities(d) == 0)
        continue;

      // Count entities
      for (MeshEntityIterator entity(*cell, d); !entity.end(); ++entity)
        entity_sets[d].insert(entity->index());
    }
  }

  // Store number of entities
  num_entities = new unsigned int[topological_dimension + 1];
  for (std::size_t d = 0; d < topological_dimension; d++)
    num_entities[d] = entity_sets[d].size();
  num_entities[topological_dimension] = num_cells;

  // Print some info
  for (std::size_t d = 0; d <= topological_dimension; d++)
  {
    info("Number of entities of dimension %d for restricted mesh: %d (out of %d).",
         d, num_entities[d], mesh.num_entities(d));
  }
}
//-----------------------------------------------------------------------------
