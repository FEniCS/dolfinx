// Copyright (C) 2008 Dag Lindbo, Anders Logg, Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2008-07-22

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
  backends.push_back("Assembly");

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
  Table t1("Reassemble total");
  Table t2("Assemble cells");
  Table t3("Build sparsity");
  Table t4("Init tensor");
  Table t5("Delete sparsity");
  Table t6("Overhead");
  
  // Iterate over backends and forms
  for (unsigned int i = 0; i < backends.size(); i++)
  {
    dolfin_set("linear algebra backend", backends[i]);
    dolfin_set("timer prefix", backends[i]);
    std::cout << "Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "  Form: " << forms[j] << std::endl;

      // Benchmark assembly
      t0(backends[i], forms[j]) = bench_form(forms[j], assemble_form);
      t2(backends[i], forms[j]) = timing(backends[i] + "Assemble over cells", true);
      t3(backends[i], forms[j]) = timing(backends[i] + "Build sparsity pattern", true);
      t4(backends[i], forms[j]) = timing(backends[i] + "Initialize tensor for assembly", true);
      t5(backends[i], forms[j]) = timing(backends[i] + "Delete sparsity pattern", true);

      // Benchmark reassembly
      t1(backends[i], forms[j]) = bench_form(forms[j], reassemble_form);
    }
  }
  
  // Compute overhead
  t6 = t0 - t2 - t3 - t4 - t5;

  // Display results
  dolfin_set("output destination", "terminal");
  cout << endl; t0.disp();
  cout << endl; t1.disp();
  cout << endl; t2.disp();
  cout << endl; t3.disp();
  cout << endl; t4.disp();
  cout << endl; t5.disp();
  cout << endl; t6.disp();
  
  return 0;
}
