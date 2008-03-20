// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
// Modified by Dag Lindbo 2008.
//
// First added:  2006-06-21
// Last changed: 2006-12-01

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "Mesh.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"

#include "GTSInterface.h"

#ifdef HAS_GTS
#include <gts.h>
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
GTSInterface::GTSInterface(Mesh& m) : mesh(m), tree(0) 
{
  if (m.geometry().dim() > 3)
    error("Sorry, GTS interface not implemented for meshes of dimension %d.", m.geometry().dim());

  buildCellTree();
}
//-----------------------------------------------------------------------------
GTSInterface::~GTSInterface() 
{
#ifdef HAS_GTS
  gts_bb_tree_destroy(tree, 1);
#endif
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::bboxCell(Cell& c)
{
#ifdef HAS_GTS
  GtsBBox* bbox;
  Point p;

  VertexIterator v(c);
  p = v->point();

  bbox = gts_bbox_new(gts_bbox_class(), (void *)c.index(),
		      p.x(), p.y(), p.z(),
		      p.x(), p.y(), p.z());
  
  for(++v; !v.end(); ++v)
    {
      p = v->point();
      if (p.x() > bbox->x2) bbox->x2 = p.x();
      if (p.x() < bbox->x1) bbox->x1 = p.x();
      if (p.y() > bbox->y2) bbox->y2 = p.y();
      if (p.y() < bbox->y1) bbox->y1 = p.y();
      if (p.z() > bbox->z2) bbox->z2 = p.z();
      if (p.z() < bbox->z1) bbox->z1 = p.z();
    }
  return bbox;

#else
  error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::bboxPoint(const Point& p)
{
#ifdef HAS_GTS

  GtsBBox* bbox;

  bbox = gts_bbox_new(gts_bbox_class(), (void *)0,
		      p.x(), p.y(), p.z(),
		      p.x(), p.y(), p.z());
  
  return bbox;

#else
  error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::bboxPoint(const Point& p1, const Point& p2)
{
#ifdef HAS_GTS

  GtsBBox* bbox;

  real x1, x2; 
  real y1, y2;
  real z1, z2;

  if(p1.x()<p2.x()){
    x1 = p1.x();
    x2 = p2.x(); }
  else {
    x1 = p2.x();
    x2 = p1.x(); }
  if(p1.y()<p2.y()){
    y1 = p1.y();
    y2 = p2.y(); }
  else {
    y1 = p2.y();
    y2 = p1.y(); }
  if(p1.z()<p2.z()){
    z1 = p1.z();
    z2 = p2.z(); }
  else {
    z1 = p2.z();
    z2 = p1.z(); }

  bbox = gts_bbox_new(gts_bbox_class(), (void *)0,
		      x1, y1, z1,
		      x2, y2, z2);
  
  return bbox;

#else
  error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::buildCellTree()
{
#ifdef HAS_GTS

  if(tree)
    warning("tree already initialized");

  GSList* bboxes = NULL;
 
  for(CellIterator ci(mesh); !ci.end(); ++ci)
    {
      Cell& c = *ci;
      bboxes = g_slist_prepend(bboxes, bboxCell(c));
    }
  
  tree = gts_bb_tree_new(bboxes);
  g_slist_free(bboxes);

#else
  error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::overlap(Cell& c, Array<uint>& cells)
{
#ifdef HAS_GTS
  GtsBBox* bbprobe;
  GtsBBox* bb;
  GSList* overlaps = 0, *overlaps_base;
  uint boundedcell;

  CellType& type = mesh.type();

  bbprobe = bboxCell(c);

  overlaps = gts_bb_tree_overlap(tree, bbprobe);
  overlaps_base = overlaps;

  while(overlaps)
    {
      bb = (GtsBBox *)overlaps->data;
      boundedcell = (uint)(long)bb->bounded;

      Cell close(mesh, boundedcell);

      if(type.intersects(c, close))
	{
	  cells.push_back(boundedcell);
	}
      overlaps = overlaps->next;
    }
  
  g_slist_free(overlaps_base);
  gts_object_destroy(GTS_OBJECT(bbprobe));

#else
  error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::overlap(Point& p, Array<uint>& cells)
{
#ifdef HAS_GTS
  GtsBBox* bbprobe;
  GtsBBox* bb;
  GSList* overlaps = 0, *overlaps_base;
  uint boundedcell;

  CellType& type = mesh.type();

  bbprobe = bboxPoint(p);

  overlaps = gts_bb_tree_overlap(tree, bbprobe);
  overlaps_base = overlaps;

  while(overlaps)
    {
      bb = (GtsBBox *)overlaps->data;
      boundedcell = (uint)(long)bb->bounded;

      Cell close(mesh, boundedcell);

      if(type.intersects(close, p))
	cells.push_back(boundedcell);

      overlaps = overlaps->next;
    }

  g_slist_free(overlaps_base);
  gts_object_destroy(GTS_OBJECT(bbprobe));

#else
  error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::overlap(Point& p1, Point& p2, Array<uint>& cells)
{
#ifdef HAS_GTS
  GtsBBox* bbprobe;
  GtsBBox* bb;
  GSList* overlaps = 0,*overlaps_base;
  uint boundedcell;

  CellType& type = mesh.type();

  bbprobe = bboxPoint(p1,p2);

  overlaps = gts_bb_tree_overlap(tree, bbprobe);
  overlaps_base = overlaps;

  while(overlaps)
    {
      bb = (GtsBBox *)overlaps->data;
      boundedcell = (uint)(long)bb->bounded;

      Cell close(mesh, boundedcell);

      if( type.intersects(close, p1, p2) )
	cells.push_back(boundedcell);

      overlaps = overlaps->next;
    }
  g_slist_free(overlaps_base);
  gts_object_destroy(GTS_OBJECT(bbprobe));

#else
  error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
