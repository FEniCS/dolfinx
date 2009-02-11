// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-11
// Last changed: 2009-02-11
//
// This demo program demonstrates how to extract matching sub meshes
// from a common mesh.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Fluid sub domain
  class Fluid : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] < 0.5;
    }
  };

  // Structure sub domain
  class Structure : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] > 0.5;
    }
  };

  // Create mesh
  Rectangle mesh(0.0, 0.0, 3.0, 1.0, 10, 30);
  plot(mesh);
  
  // Define sub domains
  Fluid fluid;
  Structure structure;

  // Extract sub meshes
  SubMesh fluid_mesh(mesh, fluid);
  SubMesh structure_mesh(mesh, structure);
  
  // Plot meshes
  plot(mesh);
  plot(fluid_mesh);
  plot(structure_mesh);
}
