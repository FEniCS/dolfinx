// Copyright (C) 2008-2010 Dag Lindbo, Anders Logg, Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2010-05-03

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
  logging(false);

  // Backends
  std::vector<std::string> backends;
  backends.push_back("uBLAS");
  backends.push_back("PETSc");
  //backends.push_back("Epetra");
  backends.push_back("MTL4");
  backends.push_back("STL");

  // Forms
  std::vector<std::string> forms;
  forms.push_back("LAP1");
  forms.push_back("LAP2");
  forms.push_back("LAP2");
  forms.push_back("TH");
  forms.push_back("STAB");
  forms.push_back("LE");
  forms.push_back("NSE");

  // Tables for results
  Table t0("Assemble total");
  Table t1("Init dofmap");
  Table t2("Build sparsity");
  Table t3("Init tensor");
  Table t4("Delete sparsity");
  Table t5("Assemble cells");
  Table t6("Overhead");
  Table t7("Reassemble total");

  // Benchmark assembly
  for (unsigned int i = 0; i < backends.size(); i++)
  {
    parameters["linear_algebra_backend"] = backends[i];
    parameters["timer_prefix"] = backends[i];
    std::cout << "Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "  Form: " << forms[j] << std::endl;
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
    parameters["linear_algebra_backend"]= backends[i];
    parameters["timer_prefix"] = backends[i];
    std::cout << "Backend: " << backends[i] << std::endl;
    for (unsigned int j = 0; j < forms.size(); j++)
    {
      std::cout << "  Form: " << forms[j] << std::endl;
      t7(backends[i], forms[j]) = bench_form(forms[j], reassemble_form);
    }
  }

  // Display results
  logging(true);
  std::cout << std::endl; info(t0, true);
  std::cout << std::endl; info(t1, true);
  std::cout << std::endl; info(t2, true);
  std::cout << std::endl; info(t3, true);
  std::cout << std::endl; info(t4, true);
  std::cout << std::endl; info(t5, true);
  std::cout << std::endl; info(t6, true);
  std::cout << std::endl; info(t7, true);

  /*
  // Display LaTeX tables
  const bool print_latex = true;
  if (print_latex)
  {
    std::cout << std::endl << t0.str_latex();
    //std::cout << std::endl << t1.str_latex();
    std::cout << std::endl << t2.str_latex();
    std::cout << std::endl << t3.str_latex();
    std::cout << std::endl << t4.str_latex();
    std::cout << std::endl << t5.str_latex();
    std::cout << std::endl << t6.str_latex();
    std::cout << std::endl << t7.str_latex();
  }
  */

  return 0;
}
