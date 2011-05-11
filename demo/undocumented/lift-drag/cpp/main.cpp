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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009
//
// First added:  2007-05-14
// Last changed: 2009-10-06
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
#include "Lift.h"
#include "Drag.h"
#include "Pressure.h"

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
  Mesh mesh("../mesh.xml.gz");

  // Read velocity field from file
  Pressure::FunctionSpace Vp(mesh);
  Function p(Vp, "../pressure.xml.gz");

  // Functionals for lift and drag
  Lift::Functional L(mesh, p);
  Drag::Functional D(mesh, p);

  // Assemble functionals over sub domain
  Fish fish;
  double lift = assemble(L, fish);
  double drag = assemble(D, fish);

  info("Lift: %f", lift);
  info("Drag: %f", drag);
}
