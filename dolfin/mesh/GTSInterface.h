// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2009.
//
// First added:  2006-06-21
// Last changed: 2009-01-12

#ifndef __GTS_INTERFACE_H
#define __GTS_INTERFACE_H

#ifdef HAS_GTS

#include <dolfin/common/types.h>
#include <vector>

// Forward declarations
struct  _GtsBBox;
typedef _GtsBBox GtsBBox;
struct  _GNode;
typedef _GNode GNode;

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class Point;
  class Cell;

  class GTSInterface
  {
  public:

    /// Create GTS interface for mesh
    GTSInterface(const Mesh& mesh);

    /// Destructor
    ~GTSInterface();

    /// Compute cells overlapping point
    void intersection(const Point& p, std::vector<uint>& cells);

    /// Compute cells overlapping line defined by points
    void intersection(const Point& p0, const Point& p1, std::vector<uint>& cells);

    /// Compute cells overlapping cell
    void intersection(const Cell& cell, std::vector<uint>& cells);

  private:

    /// Create bounding box for a single point
    GtsBBox* create_box(const Point& p);

    /// Create bounding box for a pair of points
    GtsBBox* create_box(const Point& p0, const Point& p1);

    /// Create bounding box for cell
    GtsBBox* create_box(const Cell& cell);

    // Build tree (hierarchy) of bounding boxes
    void buildCellTree();

    // The mesh
    const Mesh& mesh;

    // GTS tree
    GNode* tree;

  };

}

#endif

#endif
