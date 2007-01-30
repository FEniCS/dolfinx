// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

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
    
    IntersectionDetector();

    void init(Mesh& mesh);

    void overlap(Cell& c, Array<uint>& overlap);
    void overlap(Point& p, Array<uint>& overlap);

  private:

    GNode* tree;
    Mesh* mesh;

  };

}

#endif
