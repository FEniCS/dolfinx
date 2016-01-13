// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Garth N. Wells, 2009
// Modified by Martin S. Alnæs, 2012
//
// First added:  2007-05-14
// Last changed: 2012-08-31
//
// This demo demonstrates how to compute functionals (or forms in
// general) over subsets of the mesh. The two functionals lift and
// drag are computed for the pressure field around a dolphin. Here, we
// use the pressure field obtained from solving the Stokes equations
// (see demo program in the sub directory
// src/demo/pde/stokes/taylor-hood).
//
// The calculation only includes the pressure contribution (not shear
// forces).

#include <dolfin.h>
#include "Functionals.h"

using namespace dolfin;

// Define sub domain for the dolphin
class Fish : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (x[0] > DOLFIN_EPS && x[0] < (1.0 - DOLFIN_EPS) &&
            x[1] > DOLFIN_EPS && x[1] < (1.0 - DOLFIN_EPS) &&
            on_boundary);
  }
};

int main()
{
  // Read mesh from file
  auto mesh = std::make_shared<Mesh>("../dolfin_fine.xml.gz");

  // Read velocity field from file
  auto Vp = std::make_shared<Functionals::CoefficientSpace_p>(mesh);
  Function p(Vp, "../dolfin_fine_pressure.xml.gz");

  // Mark 'fish'
  auto markers = std::make_shared<FacetFunction<std::size_t>>(mesh, 1);
  Fish fish;
  fish.mark(*markers, 1);

  // Functionals for lift and drag
  Functionals::Form_lift L(mesh, p);
  Functionals::Form_drag D(mesh, p);

  // Attach markers to functionals
  L.ds = markers;
  D.ds = markers;

  // Assemble functionals over sub domain
  const double lift = assemble(L);
  const double drag = assemble(D);

  info("Lift: %f", lift);
  info("Drag: %f", drag);
}
