// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-21
// Last changed: 2008-10-08

#ifndef __INTERSECTION_DETECTOR_H
#define __INTERSECTION_DETECTOR_H

#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
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

    /// Compute cells overlapping point
    void intersection(const Point& p, Array<uint>& cells);

    /// Compute cells overlapping line defined by points
    void intersection(const Point& p1, const Point& p2, Array<uint>& cells);

    /// Compute cells overlapping cell
    void intersection(const Cell& cell, Array<uint>& cells);

    /// Compute overlap with curve defined by points
    void intersection(Array<Point>& points, Array<uint>& intersection);

    /// Compute overlap with mesh
    void intersection(Mesh& mesh, Array<uint>& intersection);
    
  private:

    // Interface to GTS
    GTSInterface* gts;

  };

}

#endif
