// Copyright (C) 2010 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-11-11
// Last changed: 2010-11-29
//
// If run without command-line arguments, this benchmark iterates from
// zero to MAX_NUM_THREADS. If a command-line argument --num_threads n
// is given, the benchmark is run with the specified number of threads.

#include <cstdlib>

#include <dolfin.h>
#include <dolfin/fem/AssemblerTools.h>
#include "Poisson.h"
#include "NavierStokes.h"

#define MAX_NUM_THREADS 4
#define SIZE 12
#define NUM_REPS 20

using namespace dolfin;

double bench(std::string form, const Mesh& mesh)
{
  dolfin::uint num_threads = parameters["num_threads"];
  info_underline("Benchmarking %s, num_threads = %d", form.c_str(), num_threads);

  // Create form
  boost::shared_ptr<FunctionSpace> V, W0, W1, W2, W3, W4;
  boost::shared_ptr<Form> a;
  boost::shared_ptr<Function> w0, w1, w2, w3, w4;
  if (form == "Poisson")
  {
    V.reset(new Poisson::FunctionSpace(mesh));
    a.reset(new Poisson::BilinearForm(V, V));
  }
  else if (form == "NavierStokes")
  {
    V.reset(new NavierStokes::FunctionSpace(mesh));
    W0.reset(new NavierStokes::Form_0_FunctionSpace_2(mesh));
    W1.reset(new NavierStokes::Form_0_FunctionSpace_3(mesh));
    W2.reset(new NavierStokes::Form_0_FunctionSpace_4(mesh));
    W3.reset(new NavierStokes::Form_0_FunctionSpace_5(mesh));
    W4.reset(new NavierStokes::Form_0_FunctionSpace_6(mesh));
    a.reset(new NavierStokes::BilinearForm(*V, *V));
    w0.reset(new Function(*W0));
    w1.reset(new Function(*W1));
    w2.reset(new Function(*W2));
    w3.reset(new Function(*W3));
    w4.reset(new Function(*W4));
    a->set_coefficient(0, w0);
    a->set_coefficient(1, w1);
    a->set_coefficient(2, w2);
    a->set_coefficient(3, w3);
    a->set_coefficient(4, w4);
  }

  // Create STL matrix
  //STLMatrix A;
  Matrix A;

  // Intialise matrix
  AssemblerTools::init_global_tensor(A, *a, true, false);

  // Assemble
  Timer timer("Total time");
  for (dolfin::uint i = 0; i < NUM_REPS; ++i)
    assemble(A, *a, false);
  const double t = timer.stop();

  // Write summary
  summary(true);

  info("");

  return t;
}

int main(int argc, char* argv[])
{
  // Parse command-line arguments
  parameters.parse(argc, argv);

  //SubSystemsManager::init_petsc();
  //PetscInfoAllow(PETSC_TRUE, PETSC_NULL);
  //PetscOptionsSetValue("-mat_inode_limit", "5");

  // Set backend
  //parameters["linear_algebra_backend"] = "Epetra";

  // Create mesh
  UnitCube old_mesh(SIZE, SIZE, SIZE);
  old_mesh.color("vertex");

  std::vector<unsigned int> coloring_type;
  coloring_type.push_back(3); coloring_type.push_back(0); coloring_type.push_back(3);
  Mesh mesh = old_mesh.renumber_by_color(coloring_type);

  // Test cases
  std::vector<std::string> forms;
  forms.push_back("Poisson");
  forms.push_back("NavierStokes");

  // If parameter num_threads has been set, just run once
  if (parameters["num_threads"].change_count() > 0)
  {
    for (unsigned int i = 0; i < forms.size(); i++)
      bench(forms[i], mesh);
  }

  // Otherwise, iterate from 1 to MAX_NUM_THREADS
  else
  {
    Table timings("Timings");
    Table speedups("Speedups");

    // Iterate over number of threads
    for (int num_threads = 0; num_threads <= MAX_NUM_THREADS; num_threads++)
    {
      // Set the number of threads
      parameters["num_threads"] = num_threads;

      // Iterate over forms
      for (unsigned int i = 0; i < forms.size(); i++)
      {
        // Run test case
        const double t = bench(forms[i], mesh);

        // Store results and scale to get speedups
        std::stringstream s;
        s << num_threads << " threads";
        timings(s.str(), forms[i]) = t;
        speedups(s.str(), forms[i]) = timings.get_value("0 threads", forms[i]) / t;
        if (num_threads == 0)
          speedups(s.str(), "(rel 1 thread " + forms[i] + ")") = "-";
        else
          speedups(s.str(),  "(rel 1 thread " + forms[i] + ")") = timings.get_value("1 threads", forms[i]) / t;
      }
    }

    // Display results
    info("");
    info(timings, true);
    info("");
    info(speedups, true);
  }

  return 0;
}
