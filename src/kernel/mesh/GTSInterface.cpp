// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
//
// First added:  2006-06-21
// Last changed: 2006-12-01

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Mesh.h>
#include <dolfin/Facet.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/GTSInterface.h>

using namespace dolfin;

#ifdef HAVE_GTS_H

#include <gts.h>

#endif
//-----------------------------------------------------------------------------
void GTSInterface::test()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::bboxCell(Cell& c)
{
#ifdef HAVE_GTS_H
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
  dolfin_error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::bboxPoint(Point& p)
{
#ifdef HAVE_GTS_H

  GtsBBox* bbox;

  bbox = gts_bbox_new(gts_bbox_class(), (void *)0,
		      p.x(), p.y(), p.z(),
		      p.x(), p.y(), p.z());
  
  return bbox;

#else
  dolfin_error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
GNode* GTSInterface::buildCellTree(Mesh& mesh)
{
#ifdef HAVE_GTS_H
  GNode* tree;
  GSList* bboxes = 0;
  //GtsBBox* bbox;

  for(CellIterator ci(mesh); !ci.end(); ++ci)
  {
    Cell& c = *ci;

    bboxes = g_slist_prepend(bboxes, bboxCell(c));
  }

  tree = gts_bb_tree_new(bboxes);
  g_slist_free(bboxes);

  return tree;

#else
  dolfin_error("missing GTS");
  return 0;
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::overlap(Cell& c, GNode* tree, Mesh& mesh, 
			   Array<uint>& cells)
{
#ifdef HAVE_GTS_H
  GtsBBox* bbprobe;
  GtsBBox* bb;
  GSList* overlaps = 0;
  uint boundedcell;

  CellType& type = mesh.type();

  bbprobe = bboxCell(c);

  overlaps = gts_bb_tree_overlap(tree, bbprobe);

  cells.clear();
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

#else
  dolfin_error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
void GTSInterface::overlap(Point& p, GNode* tree, Mesh& mesh, 
			   Array<uint>& cells)
{
#ifdef HAVE_GTS_H
  GtsBBox* bbprobe;
  GtsBBox* bb;
  GSList* overlaps = 0;
  uint boundedcell;

  CellType& type = mesh.type();

  bbprobe = bboxPoint(p);

  overlaps = gts_bb_tree_overlap(tree, bbprobe);

  cells.clear();
  while(overlaps)
  {
    bb = (GtsBBox *)overlaps->data;
    boundedcell = (uint)(long)bb->bounded;

    Cell close(mesh, boundedcell);

    cout << "test:" << endl;
    cout << close << endl;
    cout << p << endl;

    if(type.intersects(close, p))
    {
      cout << "intersects" << endl;
      cells.push_back(boundedcell);
    }
    else
    {
      cout << "doesn't intersect" << endl;
    }
    overlaps = overlaps->next;
  }

#else
  dolfin_error("missing GTS");
#endif
}
//-----------------------------------------------------------------------------
