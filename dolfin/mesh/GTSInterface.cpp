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

#ifdef HAS_GTS

#include <gts.h>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "Mesh.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"
#include "GTSInterface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GTSInterface::GTSInterface(Mesh& mesh) : mesh(mesh), tree(0)
{
  if (mesh.geometry().dim() > 3)
    error("Sorry, GTS interface not implemented for meshes of dimension %d.",
          mesh.geometry().dim());

  // Build tree (hierarchy) of bounding boxes
  buildCellTree();
}
//-----------------------------------------------------------------------------
GTSInterface::~GTSInterface() 
{
  // Delete tree, including leaves (the bounding boxes)
  gts_bb_tree_destroy(tree, true);
}
//-----------------------------------------------------------------------------
void GTSInterface::intersection(const Point& p, Array<uint>& cells)
{
  // Create probe for point
  GtsBBox* probe = createBox(p);

  // Compute overlap with probe
  GSList* overlaps = gts_bb_tree_overlap(tree, probe);
  
  // Iterate over overlap
  while (overlaps)
  {
    // Extract cell index for bounding box
    const GtsBBox* box = (GtsBBox *) overlaps->data;
    const uint cell_index = (uint)(long) box->bounded;
    
    // Check for intersection with cell
    Cell c(mesh, cell_index);
    if (c.intersects(p))
      cells.push_back(cell_index);
    
    // Go to next bounding box
    overlaps = overlaps->next;
  }
  
  // Delete overlap and probe
  g_slist_free(overlaps);
  gts_object_destroy(GTS_OBJECT(probe));
}
//-----------------------------------------------------------------------------
void GTSInterface::intersection(const Point& p0, const Point& p1, Array<uint>& cells)
{
  // Create probe for line
  GtsBBox* probe = createBox(p0, p1);

  // Compute overlap with probe
  GSList* overlaps = gts_bb_tree_overlap(tree, probe);

  // Iterate over overlap
  while (overlaps)
  {
    // Extract cell index for bounding box
    const GtsBBox* box = (GtsBBox *) overlaps->data;
    const uint cell_index = (uint)(long) box->bounded;

    // Check for intersection with cell
    Cell c(mesh, cell_index);
    if (c.intersects(p0, p1))
      cells.push_back(cell_index);

    // Go to next bounding box
    overlaps = overlaps->next;
  }

  // Delete overlap and probe
  g_slist_free(overlaps);
  gts_object_destroy(GTS_OBJECT(probe));
}
//-----------------------------------------------------------------------------
void GTSInterface::intersection(Cell& cell, Array<uint>& cells)
{
  // Create probe for cell
  GtsBBox* probe = createBox(cell);

  // Compute overlap with probe
  GSList* overlaps = gts_bb_tree_overlap(tree, probe);

  // Iterate over overlap
  while (overlaps)
  {
    // Extract cell index for bounding box
    const GtsBBox* box = (GtsBBox *) overlaps->data;
    const uint cell_index = (uint)(long) box->bounded;

    // Check for intersection with cell
    Cell c(mesh, cell_index);
    if (c.intersects(cell))
      cells.push_back(cell_index);
    
    // Go to next bounding box
    overlaps = overlaps->next;
  }
  
  // Delete overlap and probe
  g_slist_free(overlaps);
  gts_object_destroy(GTS_OBJECT(probe));
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::createBox(const Point& p)
{
  // Create bounding box
  GtsBBox* box = gts_bbox_new(gts_bbox_class(), 0,
                               p.x(), p.y(), p.z(),
                               p.x(), p.y(), p.z());
  
  return box;
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::createBox(const Point& p0, const Point& p1)
{
  // Compute coordinates for bounding box
  const double x0 = std::min(p0.x(), p1.x());
  const double y0 = std::min(p0.y(), p1.y());
  const double z0 = std::min(p0.z(), p1.z());
  const double x1 = std::max(p0.x(), p1.x());
  const double y1 = std::max(p0.y(), p1.y());
  const double z1 = std::max(p0.z(), p1.z());
  
  // Create bounding box
  GtsBBox* box = gts_bbox_new(gts_bbox_class(), 0,
                              x0, y0, z0, x1, y1, z1);
  
  return box;
}
//-----------------------------------------------------------------------------
GtsBBox* GTSInterface::createBox(Cell& cell)
{
  // Pick first vertex
  VertexIterator v(cell);
  Point p = v->point();

  // Compute coordinates for bounding box
  double x0 = p.x();
  double y0 = p.y(); 
  double z0 = p.z();
  double x1 = x0;
  double y1 = y0; 
  double z1 = z0;
  for (++v; !v.end(); ++v)
  {
    p = v->point();
    x0 = std::min(x0, p.x());
    x1 = std::max(x1, p.x());
    y0 = std::min(y0, p.y());
    y1 = std::max(y1, p.y());
    z0 = std::min(z0, p.z());
    z1 = std::max(z1, p.z());
  }

  // Create bounding box
  GtsBBox* box = gts_bbox_new(gts_bbox_class(), (void *) cell.index(),
                               x0, y0, z0, x1, y1, z1);
  
  return box;
}
//-----------------------------------------------------------------------------
void GTSInterface::buildCellTree()
{
  dolfin_assert(tree == 0);

  // Build list of bounding boxes for cells
  GSList* bboxes = 0; 
  for (CellIterator c(mesh); !c.end(); ++c)
    bboxes = g_slist_prepend(bboxes, createBox(*c));
  
  // Build tree (hierarchy) of bounding boxes
  tree = gts_bb_tree_new(bboxes);

  // Delete list of bounding boxes
  g_slist_free(bboxes);
}
//-----------------------------------------------------------------------------

#endif
