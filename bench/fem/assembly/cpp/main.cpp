// Copyright (C) 2008-2009 Dag Lindbo, Anders Logg, Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2009-05-20

#include <string>
#include <vector>
#include <iostream>
#include <dolfin.h>
#include "forms.h"

using namespace dolfin;

double assemble_form(Form& form)
{
  // Assemble once
  const double t0 = time();
  Matrix A;
  assemble(A, form);
  return time() - t0;
}

double reassemble_form(Form& form)
{
  // Assemble once
  Matrix A;
  assemble(A, form);

  // Reassemble
  const double t0 = time();
  assemble(A, form, false);
  return time() - t0;
}

int main()
{
  dolfin_set("output destination", "silent");

  // Backends
  std::vector<std::string> backends;
  backends.push_back("uBLAS");
  backends.push_back("PETSc");
  backends.push_back("Epetra");
  backends.push_back("MTL4");
  backends.push_back("STL");

  // Forms
  std::vector<std::string> forms;
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
    std::cout << "BENCH Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "BENCH  Form: " << forms[j] << std::endl;
      const double tt0 = bench_form(forms[j], assemble_form);
      const double tt1 = timing(backends[i] + t1.title(), true);
      const double tt2 = timing(backends[i] + t2.title(), true);
      const double tt3 = timing(backends[i] + t3.title(), true);
      const double tt4 = timing(backends[i] + t4.title(), true);
      const double tt5 = timing(backends[i] + t5.title(), true);
      t0(backends[i], forms[j]) = tt0;
      t1(backends[i], forms[j]) = tt1;
      t2(backends[i], forms[j]) = tt2;
      t3(backends[i], forms[j]) = tt3;
      t4(backends[i], forms[j]) = tt4;
      t5(backends[i], forms[j]) = tt5;
      t6(backends[i], forms[j]) = tt0 - tt1 - tt2 - tt3 - tt4 - tt5;
    }
  }

  // Benchmark reassembly
  for (unsigned int i = 0; i < backends.size(); i++)
  {
    dolfin_set("linear algebra backend", backends[i]);
    dolfin_set("timer prefix", backends[i]);
    std::cout << "BENCH Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "BENCH  Form: " << forms[j] << std::endl;
      t7(backends[i], forms[j]) = bench_form(forms[j], reassemble_form);
    }
  }

  // Display results
  dolfin_set("output destination", "terminal");
  cout << endl; info(t0);
  cout << endl; info(t1);
  cout << endl; info(t2);
  cout << endl; info(t3);
  cout << endl; info(t4);
  cout << endl; info(t5);
  cout << endl; info(t6);
  cout << endl; info(t7);
  
  return 0;
}
