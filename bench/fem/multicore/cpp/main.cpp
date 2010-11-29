// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// If run without command-line arguments, this benchmark iterates from
// zero to MAX_NUM_THREADS. If a command-line argument --num_threads n
// is given, the benchmark is run with the specified number of threads.
//
// First added:  2010-11-11
// Last changed: 2010-11-29

#include <cstdlib>

#include <dolfin.h>
#include <dolfin/fem/AssemblerTools.h>
#include "Poisson.h"
#include "NavierStokes.h"

#define MAX_NUM_THREADS 4
#define SIZE 32
#define NUM_REPS 20

using namespace dolfin;

double bench(std::string form, const Mesh& mesh)
{
  dolfin::uint num_threads = parameters["num_threads"];
  info_underline("Benchmarking %s, num_threads = %d", form.c_str(), num_threads);

  // Create form
  FunctionSpace *V(0), *W0(0), *W1(0), *W2(0), *W3(0), *W4(0);
  Form* a(0);
  Function *w0(0), *w1(0), *w2(0), *w3(0), *w4(0);
  if (form == "Poisson")
  {
    V = new Poisson::FunctionSpace(mesh);
    a = new Poisson::BilinearForm(*V, *V);
  }
  else if (form == "NavierStokes")
  {
    V  = new NavierStokes::FunctionSpace(mesh);
    W0 = new NavierStokes::Form_0_FunctionSpace_2(mesh);
    W1 = new NavierStokes::Form_0_FunctionSpace_3(mesh);
    W2 = new NavierStokes::Form_0_FunctionSpace_4(mesh);
    W3 = new NavierStokes::Form_0_FunctionSpace_5(mesh);
    W4 = new NavierStokes::Form_0_FunctionSpace_6(mesh);
    a = new NavierStokes::BilinearForm(*V, *V);
    w0 = new Function(*W0);
    w1 = new Function(*W1);
    w2 = new Function(*W2);
    w3 = new Function(*W3);
    w4 = new Function(*W4);
    a->set_coefficient(0, *w0);
    a->set_coefficient(1, *w1);
    a->set_coefficient(2, *w2);
    a->set_coefficient(3, *w3);
    a->set_coefficient(4, *w4);
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

  // Cleanup
  delete V;
  delete W0;
  delete W1;
  delete W2;
  delete W3;
  delete W4;
  delete w0;
  delete w1;
  delete w2;
  delete w3;
  delete w4;
  delete a;

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
  UnitCube mesh(SIZE, SIZE, SIZE);
  mesh.color("vertex");
  mesh.renumber_by_color();
  //mesh.init(1);

  // Test cases
  std::vector<std::string> forms;
  forms.push_back("Poisson");
  forms.push_back("NavierStokes");

  // If parameter num_threads has been set, just run once
  if (parameters["num_threads"].change_count() > 0)
  {
    for (int i = 0; i < forms.size(); i++)
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
      for (int i = 0; i < forms.size(); i++)
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
