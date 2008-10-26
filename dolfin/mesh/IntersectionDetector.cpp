// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson, 2006.
// Modified by Ola Skavhaug, 2006.
// Modified by Dag Lindbo, 2008.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-21
// Last changed: 2008-10-08

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "Mesh.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"
#include "GTSInterface.h"
#include "IntersectionDetector.h"

using namespace dolfin;

#ifdef HAS_GTS

//-----------------------------------------------------------------------------
IntersectionDetector::IntersectionDetector(Mesh& mesh) : gts(0)
{
  gts = new GTSInterface(mesh);
}
//-----------------------------------------------------------------------------
IntersectionDetector::~IntersectionDetector()
{
  delete gts;
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(p, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p0, const Point& p1, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(p0, p1, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Cell& c, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(c, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Array<Point>& points, Array<uint>& cells) 
{
  // Intersect each segment with mesh
  Array<uint> cc;
  for (uint i = 1; i < points.size(); i++)
    gts->intersection(points[i - 1], points[i], cc);

  // Remove repeated cells
  std::sort(cc.begin(), cc.end());
  Array<unsigned int>::iterator it;
  it = std::unique(cc.begin(), cc.end());
  cc.resize(it - cc.begin());  
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Mesh& mesh, Array<uint>& cells)
{
  // Intersect each cell with mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    intersection(*cell, cells);
  
  // Remove repeated cells
  std::sort(cells.begin(), cells.end());
  Array<unsigned int>::iterator it;
  it = std::unique(cells.begin(), cells.end());
  cells.resize(it - cells.begin());  
}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
IntersectionDetector::IntersectionDetector(Mesh& mesh)
{
  error("DOLFIN has been compiled without GTS, intersection detection not available.");
}
//-----------------------------------------------------------------------------
IntersectionDetector::~IntersectionDetector() {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p0, const Point& p1, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Cell& c, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Array<Point>& points, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Mesh& mesh, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------

#endif
