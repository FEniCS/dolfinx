// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-14
// Last changed: 2007-08-20
//
// This demo demonstrates how to compute functionals (or forms
// in general) over subsets of the mesh. The two functionals
// lift and drag are computed for the pressure field around
// a dolphin. Here, we use the pressure field obtained from
// solving the Stokes equations (see demo program in the
// sub directory src/demo/pde/stokes/taylor-hood).

#include <dolfin.h>
#include "Lift.h"
#include "Drag.h"

using namespace dolfin;

int main()
{
  // Read velocity field from file and get the mesh
  Function p("../pressure.xml.gz");
  Mesh& mesh(p.mesh());

  // Define sub domain for the dolphin
  class Fish : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return (x[0] > DOLFIN_EPS && x[0] < (1.0 - DOLFIN_EPS) && 
              x[1] > DOLFIN_EPS && x[1] < (1.0 - DOLFIN_EPS) &&
              on_boundary);
    }
  };  
  
  // Facet normal
  FacetNormal n(mesh);

  // Functionals for lift and drag
  LiftFunctional L(p, n);
  DragFunctional D(p, n);

  // Assemble functionals over sub domain
  Fish fish;
  
  DofMapSet dof_map_set_lift(L, mesh); 
  real lift = assemble(L, mesh, dof_map_set_lift, fish);

  DofMapSet dof_map_set_drag(D, mesh); 
  real drag = assemble(D, mesh, dof_map_set_lift, fish);

  message("Lift: %f", lift);
  message("Drag: %f", drag);
}
