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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011
//
// First added:  2010-10-11
// Last changed: 2011-11-12

#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "marking.h"

//-----------------------------------------------------------------------------
void dolfin::mark(dolfin::MeshFunction<bool>& markers,
                  const dolfin::MeshFunction<double>& indicators,
                  const std::string strategy,
                  const double fraction)
{
  if (strategy == "dorfler")
    dolfin::dorfler_mark(markers, indicators, fraction);
  else
  {
    dolfin::dolfin_error("marking.cpp",
                         "set refinement markers",
                         "Unknown marking strategy (\"%s\")",
                         strategy.c_str());
  }

  // Count number of marked cells
  std::size_t num_marked = 0;
  for(std::size_t i = 0; i < markers.size(); i++)
  {
    if (markers[i])
      num_marked++;
  }

  // Report the number of marked cells
  log(PROGRESS,
      "Marking %d cells out of %d (%.1f%%) for refinement",
      num_marked, markers.size(), 100.0*num_marked/markers.size());
}
//-----------------------------------------------------------------------------
void dolfin::dorfler_mark(dolfin::MeshFunction<bool>& markers,
                          const dolfin::MeshFunction<double>& indicators,
                          const double fraction)
{
  // Extract mesh
  const dolfin::Mesh& mesh = *markers.mesh();

  // Initialize marker mesh function
  markers.set_all(false);

  // Sort cell indices by indicators and compute sum of error
  // indicators
  std::map<double, std::size_t> sorted_cells;
  std::map<double, std::size_t>::reverse_iterator it;
  double eta_T_H = 0;
  for (std::size_t i = 0; i < mesh.num_cells(); i++)
  {
    const double value = indicators[i];
    eta_T_H += value;
    sorted_cells[value] = i;
  }

  // Determine stopping criterion for marking
  const double stop = fraction*eta_T_H;

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
