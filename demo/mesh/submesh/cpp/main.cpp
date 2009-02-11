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
  // Structure sub domain
  class Structure : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] > 1.4 && x[0] < 1.6 && x[1] < 0.6;
    }
  };
  
  // Create mesh
  Rectangle mesh(0.0, 0.0, 3.0, 1.0, 60, 20);
  
  // Create sub domain markers and mark everything as 0
  MeshFunction<unsigned int> sub_domains(mesh, mesh.topology().dim());
  sub_domains = 0;
  
  // Mark structure domain as 1
  Structure structure;
  structure.mark(sub_domains, 1);
  
  // Extract sub meshes
  SubMesh fluid_mesh(mesh, sub_domains, 0);
  SubMesh structure_mesh(mesh, sub_domains, 1);
  
  // Plot meshes
  plot(fluid_mesh);
  plot(structure_mesh);
}
