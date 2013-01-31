// Copyright (C) 2010 Anders Logg
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
// First added:  2010-02-08
// Last changed: 2013-01-09

#include "BoundaryMesh.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "FacetCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FacetCell::FacetCell(const BoundaryMesh& mesh, const Cell& facet)
  : Cell(mesh, 0), _facet_index(0)
{
  const std::size_t D = mesh.topology().dim();

  // Get map from facets (boundary cells) to mesh cells
  const MeshFunction<std::size_t>& cell_map = mesh.entity_map(D);

  // Get mesh facet corresponding to boundary cell
  Facet mesh_facet(mesh, cell_map[facet]);

  // Get cell index (pick first, there is only one)
  dolfin_assert(mesh_facet.num_entities(D) == 1);
  _local_index = mesh_facet.entities(D)[0];

  // Get local index of facet
  _facet_index = index(mesh_facet);
}
//-----------------------------------------------------------------------------
FacetCell::~FacetCell()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t FacetCell::facet_index() const
{
  return _facet_index;
}
//-----------------------------------------------------------------------------
