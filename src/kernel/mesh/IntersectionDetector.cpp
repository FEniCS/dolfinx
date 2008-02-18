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
IntersectionDetector::IntersectionDetector(Mesh& mesh) : mesh(mesh), tree(0)
{
  // Build tree
  tree = GTSInterface::buildCellTree(mesh);
}
//-----------------------------------------------------------------------------
IntersectionDetector::~IntersectionDetector()
{
  // FIXME: Should delete tree here but need to include GNode properly.
  // FIXME: (warning: invalid use of incomplete type 'struct _GNode')

  //if (tree)
  //  delete (GNode*) tree;
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Cell& c, Array<uint>& cells)
{
  GTSInterface::overlap(c, tree, mesh, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Point& p, Array<uint>& cells)
{
  GTSInterface::overlap(p, tree, mesh, cells);
}
//-----------------------------------------------------------------------------
