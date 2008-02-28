// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
//
// First added:  2006-06-21
// Last changed: 2008-02-18

#include <dolfin/IntersectionDetector.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Mesh.h>
#include <dolfin/Facet.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>

#include <dolfin/GTSInterface.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
IntersectionDetector::IntersectionDetector(Mesh& mesh) : gts(mesh) {}
//-----------------------------------------------------------------------------
IntersectionDetector::~IntersectionDetector() {}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Cell& c, Array<uint>& cells)
{
  cells.clear();
  gts.overlap(c, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Point& p, Array<uint>& cells)
{
  cells.clear();
  gts.overlap(p, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Point& p1, Point& p2, Array<uint>& cells)
{
  cells.clear();
  gts.overlap(p1, p2, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Array<Point>& points, Array<uint>& cells)
{
  cells.clear();
  for(int i=0; i<points.size(); i++)
    gts.overlap(points[i],cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::curve_overlap(Array<Point>& points, 
					 Array<uint>& cells) 
{
  // Intersect each segment with mesh
  Array<uint> cc;
  for(int i=0; i<points.size()-1; i++)
    gts.overlap(points[i],points[i+1],cc);

  // sort cells
  std::sort(cc.begin(),cc.end());

  // remove repeated cells
  cells.clear();
  uint k = 0;
  cells.push_back(cc[k]);
  for(int i=0; i<cc.size(); i++){
    if(cc[i] > k){
      cells.push_back(cc[i]);
      k = cc[i];
    }
  }
}
