// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// If run without command-line arguments, this benchmark iterates from
// zero to MAX_NUM_THREADS. If a command-line argument --num_threads n
// is given, the benchmark is run with the specified number of threads.
//
// First added:  2010-11-11
// Last changed: 2010-11-11

#include <cstdlib>

#include <dolfin.h>
#include "Poisson.h"
#include "NavierStokes.h"

#define MAX_NUM_THREADS 4
#define SIZE 64

using namespace dolfin;

double bench(std::string form)
{
  dolfin::uint num_threads = parameters["num_threads"];
  info_underline("Benchmarking %s, num_threads = %d", form.c_str(), num_threads);

  // Create mesh
  UnitCube mesh(SIZE, SIZE, SIZE);

  // Create form
  FunctionSpace* V(0);
  Form* a(0);
  Function* w(0);
  if (form == "Poisson")
  {
    V = new Poisson::FunctionSpace(mesh);
    a = new Poisson::BilinearForm(*V, *V);
  }
  else if (form == "NavierStokes")
  {
    V = new NavierStokes::FunctionSpace(mesh);
    a = new NavierStokes::BilinearForm(*V, *V);
    w = new Function(*V);
    a->set_coefficient("w", *w);
  }

  // Create matrix
  STLMatrix A;

  // Assemble
  Timer timer("Total time");
  assemble(A, *a);
  double t = timer.stop();

  // Write summary
  summary(true);

  // Cleanup
  delete V;
  delete a;
  delete w;

  info("");

  return t;
}

int main(int argc, char* argv[])
{
  // Parse command-line arguments
  parameters.parse(argc, argv);

  // Test cases
  std::vector<std::string> forms;
  forms.push_back("Poisson");
  forms.push_back("NavierStokes");

  // If parameter num_threads has been set, just run once
  if (parameters["num_threads"].change_count() > 0)
  {
    for (int i = 0; i < forms.size(); i++)
      bench(forms[i]);
  }

  // Otherwise, iterate from 1 to MAX_NUM_THREADS
  else
  {
    Table timings("Timings");
    Table speedups("Speedups");

    // Iterate over number of threads
    for (int num_threads = 1; num_threads <= MAX_NUM_THREADS; num_threads++)
    {
      // Set the number of threads
      parameters["num_threads"] = num_threads;

      // Iterate over forms
      for (int i = 0; i < forms.size(); i++)
      {
        // Run test case
        double t = bench(forms[i]);

        // Store results and scale to get speedups
        std::stringstream s;
        s << num_threads << " threads";
        timings(s.str(), forms[i]) = t;
        speedups(s.str(), forms[i]) = timings.get_value("1 threads", forms[i]) / t;
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
