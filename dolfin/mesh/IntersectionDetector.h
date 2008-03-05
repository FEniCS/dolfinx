// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-02-18

#ifndef __INTERSECTION_DETECTOR_H
#define __INTERSECTION_DETECTOR_H

#include <dolfin/main/constants.h>

#include "GTSInterface.h"

namespace dolfin
{
  class Mesh;
  class Cell;
  class Point;
  //class GTSInterface;
  template <class T> class Array;

  class IntersectionDetector
  {
  public:

    /// Constructor
    IntersectionDetector(Mesh& mesh);

    /// Destructor
    ~IntersectionDetector();

    /// Compute overlap with mesh
    void overlap(Cell& c, Array<uint>& overlap);

    /// Compute overlap with point
    void overlap(Point& p, Array<uint>& overlap);

    /// Compute overlap with bounding box
    void overlap(Point& p1, Point& p2, Array<uint>& overlap);

    // /// Compute overlap with set of points
    //void overlap(Array<Point>& points, Array<uint>& overlap);

    /// Compute overlap with set of line segments
    void curve_overlap(Array<Point>& points, Array<uint>& overlap);

  private:
    IntersectionDetector();
    IntersectionDetector(const IntersectionDetector&);

    GTSInterface gts;
  };
}

#endif
