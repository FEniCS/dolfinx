// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-11
// Last changed: 2008-08-11

#ifndef __HARMONIC_SMOOTHING_H
#define __HARMONIC_SMOOTHING_H

namespace dolfin
{

  class Mesh;

  /// This class implements harmonic mesh smoothing. Poisson's equation
  /// is solved with zero right-hand side (Laplace's equation) for each
  /// coordinate direction to compute new coordinates for all vertices,
  /// given new locations for the coordinates of the boundary.

  class HarmonicSmoothing
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, Mesh& new_boundary);

  };

}

#endif
