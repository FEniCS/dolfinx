// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-24
// Last changed: 2008-02-13
//
// This demo program demonstrates how to mark sub domains
// of a mesh and store the sub domain markers as a mesh
// function to a DOLFIN XML file.
//
// The sub domain markers produced by this demo program
// are the ones used for the Stokes demo programs.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Sub domain for no-slip (everything except inflow and outflow)
  class Noslip : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

  // Sub domain for inflow (right)
  class Inflow : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return x[0] > 1.0 - DOLFIN_EPS && on_boundary;
    }
  };

  // Sub domain for outflow (left)
  class Outflow : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS && on_boundary;
    }
  };

  // Read mesh
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");

  // Create mesh function over the cell facets
  MeshFunction<unsigned int> sub_domains(mesh, mesh.topology().dim() - 1);

  // Mark all facets as sub domain 3
  sub_domains = 3;

  // Mark no-slip facets as sub domain 0
  Noslip noslip;
  noslip.mark(sub_domains, 0);

  // Mark inflow as sub domain 1
  Inflow inflow;
  inflow.mark(sub_domains, 1);

  // Mark outflow as sub domain 2
  Outflow outflow;
  outflow.mark(sub_domains, 2);

  // Save sub domains to file
  File file("subdomains.xml");
  file << sub_domains;
}
