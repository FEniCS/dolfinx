// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#ifndef __GTS_INTERFACE_H
#define __GTS_INTERFACE_H

#ifdef HAS_PETSC

#include <dolfin/common/types.h>

// Forward declarations
struct  _GtsBBox;
typedef _GtsBBox GtsBBox;
struct  _GNode;
typedef _GNode GNode;

namespace dolfin
{

  class Mesh;
  class Point;
  class Cell;
  template <class T> class Array;

  class GTSInterface
  {
  public:

    /// Create GTS interface for mesh
    GTSInterface(Mesh& m);

    /// Destructor
    ~GTSInterface();

    /// Compute cells overlapping c
    void overlap(Cell& c, Array<uint>& cells);

    /// Compute cells overlapping p
    void overlap(Point& p, Array<uint>& cells);

    /// Compute cells overlapping bounding box constructed from p1 and p2
    void overlap(Point& p1, Point& p2, Array<uint>& cells);

  private:

    /// Construct bounding box of cell
    GtsBBox* bboxCell(Cell&);

    /// Construct bounding box of a single point
    GtsBBox* bboxPoint(const Point&);

    /// Construct bounding box of a pair of points
    GtsBBox* bboxPoint(const Point&, const Point&);

    /// Construct hierarchical space partition tree of mesh
    void buildCellTree();

    // The mesh
    Mesh& mesh;

    // GTS tree
    GNode* tree;

  };

}

#endif

#endif
