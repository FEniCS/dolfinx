// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Face.h"
#include "Cell.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
// double Face::area() const
// {
//   assert(_mesh);

//   const std::size_t D = _mesh->topology().dim();

//   // If the Face is the same topological dimension as cell
//   if (D == 2)
//   {
//     // Get the cell corresponding to this Face
//     const Cell cell(*_mesh, this->index());

//     // Return the generalized volume (area)
//     return cell.volume();
//   }
//   else
//   {

//     // Initialize needed connectivity
//     _mesh->create_connectivity(2, D);

//     // Get cell to which face belong (first cell when there is more than one)
//     const Cell cell(*_mesh, this->entities(D)[0]);

//     // Get local index of facet with respect to the cell
//     const std::size_t local_facet = cell.index(*this);

//     return cell.facet_area(local_facet);
//   }
// }
// //-----------------------------------------------------------------------------
