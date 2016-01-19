// Copyright (C) 2008-2010 Dag Lindbo, Anders Logg and Ilmar Wilbers
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
// First added:  2008-07-22
// Last changed: 2011-09-21

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
  Assembler assembler;
  assembler.assemble(A, form);
  return time() - t0;
}

double reassemble_form(Form& form)
{
  // Assemble once
  Matrix A;
  Assembler assembler;
  assembler.assemble(A, form);

  // Reassemble
  const double t0 = time();
  assembler.assemble(A, form);
  return time() - t0;
}

int main(int argc, char* argv[])
{
  info("Assembly for various forms and backends");
  set_log_active(false);

  // FIXME: Why?
  parameters["reorder_dofs_serial"] = false;

  // Forms
  std::vector<std::string> forms;
  forms.push_back("poisson1");
  forms.push_back("poisson2");
  forms.push_back("poisson3");
  forms.push_back("stokes");
  forms.push_back("stabilization");
  forms.push_back("elasticity");
  forms.push_back("navierstokes");

  // Backends
  std::vector<std::string> backends = {"PETSc", "Tpetra", "Eigen"};
  {
    for (auto it = backends.begin(); it != backends.end();)
      if (has_linear_algebra_backend(*it))
        ++it;
      else
        it = backends.erase(it);
  }

  // Override forms and backends with command-line arguments
  if (argc == 3)
  {
    forms.clear();
    forms.push_back(argv[1]);
    backends.clear();
    backends.push_back(argv[2]);
  }
  else if (argc != 1)
  {
    std::cout << "Usage: bench [form] [backend]" << std::endl;
    exit(1);
  }

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
  for (unsigned int i = 0; i < forms.size(); i++)
  {
    std::cout << "Form: " << forms[i] << std::endl;
    for (unsigned int j = 0; j < backends.size(); j++)
    {
      parameters["linear_algebra_backend"] = backends[j];
      parameters["timer_prefix"] = backends[j];
      std::cout << "  Backend: " << backends[j] << std::endl;
      const double tt0 = bench_form(forms[i], assemble_form);
      const auto timing1 = timing(backends[j] + t1.name(), TimingClear::clear);
      const auto timing2 = timing(backends[j] + t2.name(), TimingClear::clear);
      const auto timing3 = timing(backends[j] + t3.name(), TimingClear::clear);
      const auto timing4 = timing(backends[j] + t4.name(), TimingClear::clear);
      const auto timing5 = timing(backends[j] + t5.name(), TimingClear::clear);
      const double tt1
        = std::get<1>(timing1)/static_cast<double>(std::get<0>(timing1));
      const double tt2
        = std::get<1>(timing2)/static_cast<double>(std::get<0>(timing2));
      const double tt3
        = std::get<1>(timing3)/static_cast<double>(std::get<0>(timing3));
      const double tt4
        = std::get<1>(timing4)/static_cast<double>(std::get<0>(timing4));
      const double tt5
        = std::get<1>(timing5)/static_cast<double>(std::get<0>(timing5));
      t0(forms[i], backends[j]) = tt0;
      t1(forms[i], backends[j]) = tt1;
      t2(forms[i], backends[j]) = tt2;
      t3(forms[i], backends[j]) = tt3;
      t4(forms[i], backends[j]) = tt4;
      t5(forms[i], backends[j]) = tt5;
      t6(forms[i], backends[j]) = tt0 - tt1 - tt2 - tt3 - tt4 - tt5;
      std::cout << "  BENCH " << forms[i] << "-" << backends[j] << " " << tt0
                << std::endl;
    }
  }

  // Benchmark reassembly
  if (argc == 1)
  {
    for (unsigned int i = 0; i < forms.size(); i++)
    {
      std::cout << "Form: " << forms[i] << std::endl;
      for (unsigned int j = 0; j < backends.size(); j++)
      {
        parameters["linear_algebra_backend"] = backends[j];
        parameters["timer_prefix"] = backends[j];
        std::cout << "  Backend: " << backends[j] << std::endl;
        t7(forms[i], backends[j]) = bench_form(forms[i], reassemble_form);
      }
    }
  }

  // Display results
  set_log_active(true);
  std::cout << std::endl; info(t0, true);
  std::cout << std::endl; info(t1, true);
  std::cout << std::endl; info(t2, true);
  std::cout << std::endl; info(t3, true);
  std::cout << std::endl; info(t4, true);
  std::cout << std::endl; info(t5, true);
  std::cout << std::endl; info(t6, true);
  if (argc == 1)
    std::cout << std::endl; info(t7, true);

  return 0;
}
