// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2007-05-24

#include <dolfin.h>

using namespace dolfin;

// Run a few simple tests
void bench(unsigned int n, unsigned int M)
{
  double t = 0.0;
  double MM = static_cast<double>(M);

  UnitCube mesh(n, n, n);
  cout << "n = " << n << ": " << mesh << endl;

  // Create unit cube
  tic();
  for (unsigned int i = 0; i < M; ++i)
    UnitCube mesh(n, n, n);
  t = toc() / MM;
  message("  Create unit cube mesh: %.3e", t);

  // Iterate over entities
  unsigned int sum = 0;
  tic();
  for (unsigned int i = 0; i < M; ++i)
  {
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        sum += v->index();
  }
  t = toc() / MM;
  message("  Iterate over entities: %.3e (sum = %u)", t, sum);
  
  // Uniform refinement
  dolfin_set("output destination", "silent");
  tic();
  mesh.refine();
  t = toc();
  dolfin_set("output destination", "terminal");
  message("  Uniform refinement:    %.3e", t);
}

// Just create a single mesh (useful for memory benchmarking)
void bench(unsigned int n)
{
  UnitCube mesh(n, n, n);
  cout << mesh << endl;
  message("Mesh created, sleeping for 5 seconds...");
  sleep(5);
}

int main(int argc, char** argv)
{
  if ( argc > 1 )
  {
    // Create a single mesh of size n x n x n
    unsigned int n = static_cast<unsigned int>(atoi(argv[1]));
    bench(n);
  }
  else
  for (unsigned int n = 1; n <= 32; n++)
  {
    // Run a series of benchmarks
    bench(n, 100);
  }

  return 0;
}
