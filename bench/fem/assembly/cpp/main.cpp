// Copyright (C) 2008 Dag Lindbo, Anders Logg, Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2008-08-07

#include <iostream>
#include <dolfin.h>
#include "forms.h"

using namespace dolfin;

real assemble_form(Form& form, Mesh& mesh)
{
  // Assemble once
  const real t0 = time();
  Matrix A;
  assemble(A, form, mesh);
  return time() - t0;
}

real reassemble_form(Form& form, Mesh& mesh)
{
  // Assemble once
  Matrix A;
  assemble(A, form, mesh);

  // Reassemble
  const real t0 = time();
  assemble(A, form, mesh, false);
  return time() - t0;
}

int main()
{
  dolfin_set("output destination", "silent");

  // Backends
  Array<std::string> backends;
  backends.push_back("uBLAS");
  backends.push_back("PETSc");
  backends.push_back("Epetra");
  backends.push_back("MTL4");
  backends.push_back("STL");

  // Forms
  Array<std::string> forms;
  forms.push_back("Poisson2DP1");
  forms.push_back("Poisson2DP2");
  forms.push_back("Poisson2DP3");
  forms.push_back("THStokes2D");
  forms.push_back("StabStokes2D");
  forms.push_back("Elasticity3D");
  forms.push_back("NSEMomentum3D");

  // Tables for results
  Table t0("Assemble total");
  Table t1("Init dof map");
  Table t2("Build sparsity");
  Table t3("Init tensor");
  Table t4("Delete sparsity");
  Table t5("Assemble cells");
  Table t6("Overhead");
  Table t7("Reassemble total");
  
  // Benchmark assembly
  for (unsigned int i = 0; i < backends.size(); i++)
  {
    dolfin_set("linear algebra backend", backends[i]);
    dolfin_set("timer prefix", backends[i]);
    std::cout << "Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "  Form: " << forms[j] << std::endl;
      t0(backends[i], forms[j]) = bench_form(forms[j], assemble_form);
      t1(backends[i], forms[j]) = timing(backends[i] + t1.title(), true);
      t2(backends[i], forms[j]) = timing(backends[i] + t2.title(), true);
      t3(backends[i], forms[j]) = timing(backends[i] + t3.title(), true);
      t4(backends[i], forms[j]) = timing(backends[i] + t4.title(), true);
      t5(backends[i], forms[j]) = timing(backends[i] + t5.title(), true);
    }
  }

  // Benchmark reassembly
  for (unsigned int i = 0; i < backends.size(); i++)
  {
    dolfin_set("linear algebra backend", backends[i]);
    dolfin_set("timer prefix", backends[i]);
    std::cout << "Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "  Form: " << forms[j] << std::endl;
      t7(backends[i], forms[j]) = bench_form(forms[j], reassemble_form);
    }
  }
  
  // Compute overhead
  t6 = t0 - t1 - t2 - t3 - t4 - t5;

  // Display results
  dolfin_set("output destination", "terminal");
  cout << endl; t0.disp();
  cout << endl; t1.disp();
  cout << endl; t2.disp();
  cout << endl; t3.disp();
  cout << endl; t4.disp();
  cout << endl; t5.disp();
  cout << endl; t6.disp();
  cout << endl; t7.disp();
  
  return 0;
}
