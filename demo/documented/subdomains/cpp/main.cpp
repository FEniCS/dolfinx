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
// First added:  2007-04-24
// Last changed: 2011-01-25
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
  set_log_level(1);

  // Sub domain for no-slip (everything except inflow and outflow)
  class Noslip : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

  // Sub domain for inflow (right)
  class Inflow : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] > 1.0 - DOLFIN_EPS && on_boundary;
    }
  };

  // Sub domain for outflow (left)
  class Outflow : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS && on_boundary;
    }
  };

  // Read mesh
  auto mesh = std::make_shared<Mesh>("../dolfin_fine.xml.gz");

  // Create mesh functions over the cell facets
  MeshFunction<std::size_t> sub_domains(mesh, mesh->topology().dim() - 1);
  MeshFunction<double> sub_domains_double(mesh, mesh->topology().dim() - 1);
  MeshFunction<bool> sub_domains_bool(mesh, mesh->topology().dim() - 1);

  // Mark all facets as sub domain 3
  sub_domains = 3;

  // Mark no-slip facets as sub domain 0, 0.0
  Noslip noslip;
  noslip.mark(sub_domains, 0);
  noslip.mark(sub_domains_double, 0.0);

  // Mark inflow as sub domain 1, 0.1
  Inflow inflow;
  inflow.mark(sub_domains, 1);
  inflow.mark(sub_domains_double, 0.1);

  // Mark outflow as sub domain 2, 0.2
  Outflow outflow;
  outflow.mark(sub_domains, 2);
  outflow.mark(sub_domains_double, 2);

  // Save sub domains to file
  File file("subdomains.xml");
  file << sub_domains;

  // Save sub domains to file
  // FIXME: Not implemented
  //File file_bool("subdomains_bool.xml");
  //file_bool << sub_domains_bool;

  // Save sub domains to file
  File file_double("subdomains_double.xml");
  file_double << sub_domains_double;
}
