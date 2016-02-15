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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-11-11
// Last changed: 2012-11-02
//
// If run without command-line arguments, this benchmark iterates from
// zero to MAX_NUM_THREADS. If a command-line argument --num_threads n
// is given, the benchmark is run with the specified number of threads.

#include <cstdlib>

#include <dolfin.h>
#include <dolfin/fem/AssemblerBase.h>
#include "Poisson.h"
#include "NavierStokes.h"

#define MAX_NUM_THREADS 4
#define SIZE 32
#define NUM_REPS 10

using namespace dolfin;

class PoissonFactory
{
  public:

  static std::shared_ptr<Form> a(std::shared_ptr<const Mesh> mesh)
  {
    // Create function space
    std::shared_ptr<FunctionSpace> _V(new Poisson::FunctionSpace(mesh));
    std::shared_ptr<Form> _a(new Poisson::BilinearForm(_V, _V));;
    return _a;
  }

};

class NavierStokesFactory
{
  public:

  static std::shared_ptr<Form> a(std::shared_ptr<const Mesh> mesh)
  {
    std::shared_ptr<FunctionSpace> _V(new NavierStokes::FunctionSpace(mesh));

    std::shared_ptr<FunctionSpace>
      W0(new NavierStokes::Form_a_FunctionSpace_2(mesh));
    std::shared_ptr<FunctionSpace>
      W1(new NavierStokes::Form_a_FunctionSpace_3(mesh));
    std::shared_ptr<FunctionSpace>
      W2(new NavierStokes::Form_a_FunctionSpace_4(mesh));
    std::shared_ptr<FunctionSpace>
      W3(new NavierStokes::Form_a_FunctionSpace_5(mesh));
    std::shared_ptr<FunctionSpace>
      W4(new NavierStokes::Form_a_FunctionSpace_6(mesh));

    std::shared_ptr<Function> w0(new Function(W0));
    std::shared_ptr<Function> w1(new Function(W1));
    std::shared_ptr<Function> w2(new Function(W2));
    std::shared_ptr<Function> w3(new Function(W3));
    std::shared_ptr<Function> w4(new Function(W4));

    std::shared_ptr<Form> a(new NavierStokes::BilinearForm(_V, _V));

    a->set_coefficient(0, w0);
    a->set_coefficient(1, w1);
    a->set_coefficient(2, w2);
    a->set_coefficient(3, w3);
    a->set_coefficient(4, w4);

    return a;
  }
};

double bench(std::string form, std::shared_ptr<const Form> a)
{
  std::size_t num_threads = parameters["num_threads"];
  info_underline("Benchmarking %s, num_threads = %d", form.c_str(),
                 num_threads);

  // Create matrix
  Matrix A;

  // Assemble once to initialize matrix
  Assembler assembler;
  assemble(A, *a);

  // Run timing
  Timer timer("Total time");
  for (std::size_t i = 0; i < NUM_REPS; ++i)
    assemble(A, *a);
  const double t = timer.stop();

  // Report timings
  list_timings(TimingClear::clear,
               { TimingType::wall, TimingType::user, TimingType::system });

  info("");

  return t;
}

int main(int argc, char* argv[])
{
  info("Runtime of threaded assembly benchmark");

  // Parse command-line arguments
  parameters.parse(argc, argv);

  // Create mesh
  auto old_mesh = std::make_shared<UnitCubeMesh>(SIZE, SIZE, SIZE);
  old_mesh->color("vertex");
  auto mesh = std::make_shared<Mesh>(old_mesh->renumber_by_color());

  // Test cases
  std::vector<std::pair<std::string, std::shared_ptr<const Form> > > forms;
  forms.push_back(std::make_pair("Poisson", PoissonFactory::a(mesh)));
  forms.push_back(std::make_pair("NavierStokes", NavierStokesFactory::a(mesh)));

  // If parameter num_threads has been set, just run once
  if (parameters["num_threads"].change_count() > 0)
  {
    for (std::size_t i = 0; i < forms.size(); i++)
      bench(forms[i].first, forms[i].second);
  }

  // Otherwise, iterate from 1 to MAX_NUM_THREADS
  else
  {
    Table run_timings("Timings");
    Table speedups("Speedups");

    // Iterate over number of threads
    for (int num_threads = 0; num_threads <= MAX_NUM_THREADS; num_threads++)
    {
      // Set the number of threads
      parameters["num_threads"] = num_threads;

      // Iterate over forms
      for (std::size_t i = 0; i < forms.size(); i++)
      {
        // Run test case
        const double t = bench(forms[i].first, forms[i].second);

        // Store results and scale to get speedups
        std::stringstream s;
        s << num_threads << " threads";
        run_timings(s.str(), forms[i].first) = t;
        speedups(s.str(), forms[i].first)
          = run_timings.get_value("0 threads", forms[i].first)/t;
        if (num_threads == 0)
          speedups(s.str(), "(rel 1 thread " + forms[i].first + ")") = "-";
        else
        {
          speedups(s.str(),  "(rel 1 thread " + forms[i].first + ")")
            = run_timings.get_value("1 threads", forms[i].first)/t;
        }
      }
    }

    // Display results
    info("");
    info(run_timings, true);
    info("");
    info(speedups, true);
  }

  return 0;
}
