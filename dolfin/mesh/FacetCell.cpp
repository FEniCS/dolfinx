// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-08
// Last changed: 2010-02-08

#include "Facet.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "FacetCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FacetCell::FacetCell(const Mesh& mesh, const Cell& facet)
  : Cell(mesh, 0), _facet_index(0)
{
  // Get mapping from facets (boundary cells) to mesh cells
  MeshFunction<uint>* cell_map = facet.mesh().data().mesh_function("cell map");

  // Check that mapping exists
  if (!cell_map)
    error("Unable to create create cell corresponding to facet, missing cell map.");

  // Get mesh facet corresponding to boundary cell
  Facet mesh_facet(mesh, (*cell_map)[facet]);

  // Get cell index (pick first, there is only one)
  const uint D = mesh.topology().dim();
  assert(mesh_facet.num_entities(D) == 1);
  _index = mesh_facet.entities(D)[0];

  // Get local index of facet
  _facet_index = index(mesh_facet);
}
//-----------------------------------------------------------------------------
FacetCell::~FacetCell()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint FacetCell::facet_index() const
{
  return _facet_index;
}
//-----------------------------------------------------------------------------
