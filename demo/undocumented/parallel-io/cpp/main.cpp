// Copyright (C) 2013 Chris Richardson
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
// First added:  2013-04-26
// Last changed: 2013-04-27

#include<boost/filesystem.hpp>

#include <dolfin.h>

#include "P1.h"

using namespace dolfin;

class LocalExpression : public Expression
{
public:
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0]*x[1];
  }
};

int main()
{
  
  if(MPI::num_processes() == 1)
  {
    std::cout <<  "This demo is intended to be run in parallel (using mpirun)" << std::endl;
    std::cout <<  "However, it will also run on a single process" << std::endl;
  }
  
  // make a Function F=x*y on a square mesh
  UnitSquareMesh mesh(20,20);
  P1::FunctionSpace P1(mesh);
  Function F(P1);
  LocalExpression E;
  F.interpolate(E);
  
  // output in XDMF format for visualisation
  // view it in paraview or visit
  //  File F_file("function.xdmf");
  //  F_file << F;
  
  bool file_exists = boost::filesystem::exists("mesh.xdmf");
  MPI::barrier(); // prevents race condition
  
  // Check for file "mesh.xdmf" in folder, and read in
  if (file_exists)
  {
    Mesh mesh2("mesh.xdmf");
    if (MPI::process_number() == 0)
      std::cout <<  "Read mesh using " <<  MPI::num_processes() <<  " processes." << std::endl;
    
    std::cout << "Mesh has " << mesh2.num_cells() << " cells " 
              << " and " << mesh2.num_vertices() << " vertices locally on process " << MPI::process_number() << std::endl;
  }
  else
  {
    // mesh.xdmf does not exist, so create
    File M_file("mesh.xdmf");
    M_file << mesh;
    
    if (MPI::process_number() == 0)
    {
      std::cout <<  "Wrote mesh to file using " << MPI::num_processes() <<  " processes." << std::endl;
      std::cout << "Try rerunning the demo with a different number of processes." << std::endl;
    }
  }
  
  return 0;
}
