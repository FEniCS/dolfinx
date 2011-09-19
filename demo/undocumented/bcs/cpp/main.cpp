// Copyright (C) 2008 Anders Logg
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
// First added:  2008-05-23
// Last changed: 2011-08-23
//
// This demo illustrates how to set boundary conditions for meshes
// that include boundary indicators. The mesh used in this demo was
// generated with VMTK (http://villacamozzi.marionegri.it/~luca/vmtk/).

#include <dolfin.h>
#include "Poisson.h"
#include <boost/assign/list_of.hpp>
#include <dolfin/mesh/MeshDistributed.h>

using namespace dolfin;

int main()
{
  // Create mesh and finite element
  Mesh mesh("../../../../data/meshes/aneurysm.xml.gz");


  //Poisson::FunctionSpace V(mesh);

  File mf("mf.pvd");
  mf << *mesh.domains().facet_domains(mesh);

  //File mf_cells("mf_cells.pvd");
  //mf_cells << *mesh.domains().cell_domains(mesh);

  //Mesh mesh("mesh.xml");

  /*
  MeshValueCollection<dolfin::uint> & coll
       = mesh.domains().markers(2);
  cout << "Size: " << coll.size() << endl;

  File collection("mesh.xml");
  collection << mesh;
  */

  /*
  MeshValueCollection<dolfin::uint> coll(2);
  File collection("value_collection.xml");
  collection >> coll;

  cout << "Size: " << coll.size() << endl;

  // Build list of cell indices
  const std::map<std::pair<dolfin::uint, dolfin::uint>, dolfin::uint>& _coll = coll.values();

  std::vector<dolfin::uint> cells;
  std::map<std::pair<dolfin::uint, dolfin::uint>, dolfin::uint>::const_iterator it;
  for (it = _coll.begin(); it != _coll.end(); ++it)
    cells.push_back(it->first.first);

  //Poisson::FunctionSpace V(mesh);

  MeshPartitioning::number_entities(mesh, 3);
  const std::map<dolfin::uint, std::set<std::pair<dolfin::uint, dolfin::uint> > >  hosts
        = MeshDistributed::off_process_indices(cells, 3, mesh);

  std::map<dolfin::uint, std::set<std::pair<dolfin::uint, dolfin::uint> > >::const_iterator ent;
  for (ent = hosts.begin(); ent != hosts.end(); ++ent)
  {
    const std::set<std::pair<dolfin::uint, dolfin::uint> > procs = ent->second;
    std::set<std::pair<dolfin::uint, dolfin::uint> >::const_iterator proc;
    cout << "Global dof: " << ent->first;
    for (proc = procs.begin(); proc != procs.end(); ++proc)
      cout << "  Local indec: " << proc->second << endl;
  }
  */

  /*
  // Define variational problem
  Constant f(0.0);
  Poisson::FunctionSpace V(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Define boundary condition values
  Constant u0(0.0);
  Constant u1(1.0);
  Constant u2(2.0);
  Constant u3(3.0);

  // Define boundary conditions
  DirichletBC bc0(V, u0, 0);
  DirichletBC bc1(V, u1, 1);
  DirichletBC bc2(V, u2, 2);
  DirichletBC bc3(V, u3, 3);
  std::vector<const BoundaryCondition*> bcs = boost::assign::list_of(&bc0)(&bc1)(&bc2)(&bc3);

  // Compute solution
  Function u(V);
  solve(a == L, u, bcs);

  // Write solution to file
  File file("u.pvd");
  file << u;

  // Plot solution
  plot(u);
  */
  return 0;
}
