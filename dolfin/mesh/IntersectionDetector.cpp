// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
// Modified by Dag Lindbo 2008.
//
// First added:  2006-06-21
// Last changed: 2008-08-29

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
void IntersectionDetector::overlap(const Point& p, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->overlap(p, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(const Point& p0, const Point& p1, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->overlap(p0, p1, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Cell& c, Array<uint>& cells)
{
  dolfin_assert(gts);
  gts->overlap(c, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Array<Point>& points, Array<uint>& cells) 
{
  // Intersect each segment with mesh
  Array<uint> cc;
  for (uint i = 1; i < points.size(); i++)
    gts->overlap(points[i - 1], points[i], cc);

  // Sort cells
  std::sort(cc.begin(), cc.end());

  // Remove repeated cells
  cells.clear();
  cells.push_back(cc[0]);
  uint k = cc[0];
  for (uint i = 1; i < cc.size(); i++)
  {
    if (cc[i] > k)
    {
      cells.push_back(cc[i]);
      k = cc[i];
    }
  }
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
void IntersectionDetector::overlap(const Point& p, Array<uint>& overlap) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(const Point& p0, Point& p1, Array<uint>& overlap) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Cell& c, Array<uint>& overlap) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Array<Point>& points, Array<uint>& overlap) {}
//-----------------------------------------------------------------------------

#endif
