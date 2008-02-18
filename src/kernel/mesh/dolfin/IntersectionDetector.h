// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-02-18

#ifndef __INTERSECTION_DETECTOR_H
#define __INTERSECTION_DETECTOR_H

#include <dolfin/constants.h>

// Forward declarations
struct _GNode;
typedef _GNode GNode;

namespace dolfin
{

  class Mesh;
  class BoundaryMesh;
  class Cell;
  class Point;
  template <class T> class Array;

  /// This class provides an interface for computing intersection
  /// (overlap) between a mesh and an individual cell or point

  class IntersectionDetector
  {
  public:

    /// Constructor
    IntersectionDetector(Mesh& mesh);

    /// Destructor
    ~IntersectionDetector();

    // FIXME: Should we use mesh functions instead of Array?

    /// Compute overlap with mesh
    void overlap(Cell& c, Array<uint>& overlap);

    /// Compute overlap with point
    void overlap(Point& p, Array<uint>& overlap);

  private:

    // The mesh
    Mesh& mesh;

    // The tree
    GNode* tree;

  };

}

#endif
