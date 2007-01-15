// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#ifndef __GTS_INTERFACE_H
#define __GTS_INTERFACE_H

#include <dolfin/constants.h>

// Forward declarations
struct _GtsBBox;
typedef _GtsBBox GtsBBox;
struct _GNode;
typedef _GNode GNode;

namespace dolfin
{

  class Cell;
  class Mesh;
  template <class T> class Array;

  /// This class provides a set of functions to interface the DOLFIN
  /// mesh with the GTS mesh library

  class GTSInterface
  {
  public:
    
    /// Test
    static void test();

    /// Construct bounding box of cell
    static GtsBBox* bboxCell(Cell& c);

    /// Construct hierarchical space partition tree of mesh
    static GNode* buildCellTree(Mesh& mesh);

    /// Compute cells overlapping c
    static void overlap(Cell& c, GNode* tree, Mesh& mesh, Array<uint>& cells);
  };

}

#endif
