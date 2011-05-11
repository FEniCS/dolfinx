// Copyright (C) 2010 Marie E. Rognes
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-10-11
// Last changed: 2011-03-17

#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include "marking.h"

//-----------------------------------------------------------------------------
void dolfin::mark(dolfin::MeshFunction<bool>& markers,
                  const dolfin::Vector& indicators,
                  const std::string strategy,
                  const double fraction)
{
  if (strategy == "dorfler")
    dolfin::dorfler_mark(markers, indicators, fraction);
  else
    dolfin::error("Unknown marking strategy.");

  // Count number of marked cells
  uint num_marked = 0;
  for(uint i=0; i < markers.size(); i++)
  {
    if (markers[i])
      num_marked++;
  }

  // Report the number of marked cells
  log(INFO,
      "Marking %d cells out of %d (%.1f%%) for refinement",
      num_marked, markers.size(), 100.0*num_marked/markers.size());
}
//-----------------------------------------------------------------------------
void dolfin::dorfler_mark(dolfin::MeshFunction<bool>& markers,
                          const dolfin::Vector& indicators,
                          const double fraction)
{
  // Extract mesh
  const dolfin::Mesh& mesh(markers.mesh());

  // Initialize marker mesh function
  markers.set_all(false);

  // Compute sum of error indicators
  const double eta_T_H = indicators.sum();

  // Determine stopping criterion for marking
  const double stop = fraction*eta_T_H;

  // Sort cell indices by indicators
  std::map<double, uint> sorted_cells;
  std::map<double, uint>::reverse_iterator it;
  for (dolfin::uint i = 0; i < mesh.num_cells(); i++)
    sorted_cells[indicators[i]] = i;

  // Mark using Dorfler algorithm
  double eta_A = 0.0;
  for (it = sorted_cells.rbegin(); it != sorted_cells.rend(); it++) {
    if (eta_A > stop)
      return;

    markers[it->second] = true;
    eta_A += it->first;
  }
}
//-----------------------------------------------------------------------------
