// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2009-11-11
//
// This program demonstrates the use of the TimeSeries
// class for storing a series of meshes and vectors.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  error("Time series demo needs to be updated for removal of Mesh::refine.");
  /*
  // Create empty time series
  TimeSeries series("primal");

  // Create a mesh and a vector
  UnitSquare mesh(2, 2);
  Vector x;

  // Add a bunch of meshes and vectors to the series
  double t = 0.0;
  while (t < 1.0)
  {
    // Refine mesh and resize vector
    mesh.refine();
    x.resize(mesh.num_vertices());

    // Append to series
    series.store(mesh, t);
    series.store(x, t);

    t += 0.2;
  }

  // Retrieve mesh and vector at some point in time
  series.retrieve(mesh, 0.3);
  series.retrieve(x, 0.3);

  // Plot mesh
  plot(mesh);

  return 0;
  */
}
