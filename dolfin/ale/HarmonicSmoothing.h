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

  /// Documentation of class

  class HarmonicSmoothing
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, Mesh& new_boundary);

  };

}

#endif
