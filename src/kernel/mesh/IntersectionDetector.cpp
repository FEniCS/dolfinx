// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
//
// First added:  2006-06-21
// Last changed: 2006-12-01

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
IntersectionDetector::IntersectionDetector()
{
}
//-----------------------------------------------------------------------------
void IntersectionDetector::init(Mesh& mesh)
{
  this->mesh = &mesh;
  tree = GTSInterface::buildCellTree(mesh);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Cell& c, Array<uint>& cells)
{
  GTSInterface::overlap(c, tree, *mesh, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::overlap(Point& p, Array<uint>& cells)
{
  GTSInterface::overlap(p, tree, *mesh, cells);
}
//-----------------------------------------------------------------------------
