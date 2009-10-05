// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2009.
//
// First added:  2006-06-21
// Last changed: 2009-08-27

#ifndef __INTERSECTION_DETECTOR_H
#define __INTERSECTION_DETECTOR_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class Cell;
  class Point;
  class GTSInterface;

  class IntersectionDetector
  {
  public:

    /// Create intersection detector for mesh
    IntersectionDetector(const Mesh& mesh0);

    /// Destructor
    ~IntersectionDetector();

    /// Compute cells overlapping point
    void intersection(const Point& p, std::vector<uint>& cells) const;

    /// Compute cells overlapping line defined by points
    void intersection(const Point& p1, const Point& p2, std::vector<uint>& cells) const;

    /// Compute cells overlapping cell
    void intersection(const Cell& cell, std::vector<uint>& cells) const;

    /// Compute overlap with curve defined by points
    void intersection(std::vector<Point>& points, std::vector<uint>& cells) const;

    /// Compute overlap with mesh
    void intersection(const Mesh& mesh1, std::vector<uint>& cells) const;

    /// Compute overlap with mesh (test version)
    void new_intersection(const Mesh& mesh1, std::vector<uint>& cells) const;


  private:

    /// Compute points for the new intesected mesh
    void compute_polygon(const Mesh& mesh1, const Cell& c0,
                         const std::vector<uint>& intersections) const;

    // Interface to GTS
    GTSInterface* gts;

    // The mesh that we are intersecting
    const Mesh& mesh0;

  };

}

#endif
