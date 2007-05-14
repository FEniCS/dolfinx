// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-19
// Last changed: 2007-01-17

#include <dolfin.h>

#include "AdvectionOperator_3D_3_FFC.h"

const unsigned int num_repetitions = 10;

using namespace dolfin;

double benchCurrent(BilinearForm& a, Mesh& mesh)
{
  cout << "Timing current assembly..." << endl;

  Matrix A;
  
  dolfin_log(false);
  tic();
  for (unsigned int i = 0; i < num_repetitions; i++)
    FEM::assemble(a, A, mesh);
  double t = toc();
  dolfin_log(true);
  
  //cout << A << endl;
  //A.disp();

  return t / static_cast<double>(num_repetitions);
}

double benchOld(BilinearForm& a, Mesh& mesh)
{
  cout << "Timing old assembly..." << endl;

  Matrix A;
  
  dolfin_log(false);
  tic();
  for (unsigned int i = 0; i < num_repetitions; i++)
    FEM::assembleOld(a, A, mesh);
  double t = toc();
  dolfin_log(true);
  
  //cout << A << endl;
  //A.disp();

  return t / static_cast<double>(num_repetitions);
}

double benchSimple(BilinearForm& a, Mesh& mesh)
{
  cout << "Timing simple assembly..." << endl;

  std::vector<std::map<int, double> > A;
  
  dolfin_log(false);
  tic();
  for (unsigned int i = 0; i < num_repetitions; i++)
    FEM::assembleSimple(a, A, mesh);
  double t = toc();
  dolfin_log(true);
  
  //for (unsigned int i = 0; i < A.size(); i++)
  //{
  //  cout << i << ":";
  //  for (std::map<int, double>::iterator it = A[i].begin(); it != A[i].end(); it++)
  //    cout << " (" << it->first << ", " << it->second << ")";
  //  cout << endl;
  //}

  return t / static_cast<double>(num_repetitions);
}

int main()
{
  Unity w;
  AdvectionOperator_3D_3_FFC::BilinearForm a(w);
  
  UnitCube mesh(8, 8, 8);
  mesh.init();

  real t0 = benchCurrent(a, mesh);
  real t1 = benchOld(a, mesh);
  real t2 = benchSimple(a, mesh);

  message("Current assembly: %.3g", t0);
  message("Old assembly:     %.3g", t1);
  message("Simple assembly:  %.3g", t2);

  return 0;
}
