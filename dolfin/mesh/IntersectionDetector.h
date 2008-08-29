// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-08-29

#ifndef __INTERSECTION_DETECTOR_H
#define __INTERSECTION_DETECTOR_H

#include <dolfin/common/types.h>

namespace dolfin
{
  class Mesh;
  class Cell;
  class Point;
  class GTSInterface;

  template <class T> class Array;

  class IntersectionDetector
  {
  public:

    /// Create intersection detector for mesh
    IntersectionDetector(Mesh& mesh);

    /// Destructor
    ~IntersectionDetector();

    /// Compute overlap with cell
    void overlap(Cell& c, Array<uint>& overlap);

    /// Compute overlap with point
    void overlap(Point& p, Array<uint>& overlap);

    /// Compute overlap with bounding box defined by two points
    void overlap(Point& p1, Point& p2, Array<uint>& overlap);

    /// Compute overlap with curve defined by points
    void overlap(Array<Point>& points, Array<uint>& overlap);

  private:

    // Interface to GTS
    GTSInterface* gts;

  };

}

#endif
