// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <math.h>
#include <strings.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/File.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridInit.h>
#include <dolfin/GridRefinement.h>
#include <dolfin/Grid.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Grid::Grid()
{
  gd = new GridData(this);
  bd = new BoundaryData(this);
  rd = new GridRefinementData(this);
  _parent = 0;

  rename("grid", "no description");
  clear();
}
//-----------------------------------------------------------------------------
Grid::Grid(const char* filename)
{
  gd = new GridData(this);
  bd = new BoundaryData(this);
  rd = new GridRefinementData(this);
  _parent = 0;

  rename("grid", "no description");
  clear();

  // Read grid from file
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Grid::Grid(const Grid& grid)
{
  gd = new GridData(this);
  bd = new BoundaryData(this);
  rd = new GridRefinementData(this);
  _parent = 0;

  rename("grid", "no description");
  clear();

  // Specify nodes
  for (NodeIterator n(grid); !n.end(); ++n)
    createNode(n->coord());
  
  // Specify cells
  for (CellIterator c(grid); !c.end(); ++c)
    switch (c->type()) {
    case Cell::triangle:
      createCell(c->node(0), c->node(1), c->node(2));
      break;
    case Cell::tetrahedron:
      createCell(c->node(0), c->node(1), c->node(2), c->node(3));
      break;
    default:
      dolfin_error("Unknown cell type.");
    }

  // Compute connectivity
  init();

  // Copy cell markers
  
}
//-----------------------------------------------------------------------------
Grid::~Grid()
{
  clear();

  if ( gd )
    delete gd;
  gd = 0;

  if ( bd )
    delete bd;
  bd = 0;

  if ( rd )
    delete rd;
  rd = 0;
}
//-----------------------------------------------------------------------------
void Grid::clear()
{
  gd->clear();
  bd->clear();
  rd->clear();

  _type = triangles;
  _child = 0;

  // Assume that we need to delete the parent which is created by refine().
  if ( _parent )
    delete _parent;
  _parent = 0;
}
//-----------------------------------------------------------------------------
int Grid::noNodes() const
{
  return gd->noNodes();
}
//-----------------------------------------------------------------------------
int Grid::noCells() const
{
  return gd->noCells();
}
//-----------------------------------------------------------------------------
int Grid::noEdges() const
{
  return gd->noEdges();
}
//-----------------------------------------------------------------------------
int Grid::noFaces() const
{
  return gd->noFaces();
}
//-----------------------------------------------------------------------------
Grid::Type Grid::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
void Grid::mark(Cell* cell)
{
  rd->mark(cell);
}
//-----------------------------------------------------------------------------
void Grid::refine()
{
  // Check that this is the finest grid
  if ( _child )
    dolfin_error("Only the finest grid in a grid hierarchy can be refined.");

  // Create grid hierarchy
  GridHierarchy grids(*this);

  // Refine grid hierarchy
  GridRefinement::refine(grids);

  // Swap data structures with the new finest grid. This is necessary since
  // refine() should replace the current grid with the finest grid. At the
  // same time, we store the data structures of the current grid in the
  // newly created finest grid, which becomes the next finest grid:
  //
  // Before refinement:  g0 <-> g1 <-> g2 <-> ... <-> *this(gd)
  // After refinement:   g0 <-> g1 <-> g2 <-> ... <-> *this(gd) <-> new(ngd)
  // After swap:         g0 <-> g1 <-> g2 <-> ... <-> new(gd)   <-> *this(ngd)

  // Get pointer to new grid
  Grid* new_grid = &(grids.fine());

  // Swap data
  swap(*new_grid);

  


  // Set parent and child
  if ( new_grid->_parent )
    new_grid->_parent->_child = new_grid;
  this->_parent = new_grid;
  new_grid->_child = this;



  // Compute connectivity
  init();
}
//-----------------------------------------------------------------------------
Grid& Grid::parent()
{
  if ( _parent )
    return *_parent;

  dolfin_warning("Grid has now parent.");
  return *this;
}
//-----------------------------------------------------------------------------
Grid& Grid::child()
{
  if ( _child )
    return *_child;

  dolfin_warning("Grid has now child.");
  return *this;
}
//-----------------------------------------------------------------------------
void Grid::show()
{
  cout << "---------------------------------------";
  cout << "----------------------------------------" << endl;

  cout << "Grid with " << noNodes() << " nodes and " 
       << noCells() << " cells:" << endl;
  cout << endl;

  for (NodeIterator n(this); !n.end(); ++n)
    cout << "  " << *n << endl;

  cout << endl;
  
  for (CellIterator c(this); !c.end(); ++c)
    cout << "  " << *c << endl;
  
  cout << endl;
  
  cout << "---------------------------------------";
  cout << "----------------------------------------" << endl;
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(Point p)
{
  return gd->createNode(p);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(real x, real y, real z)
{
  return gd->createNode(x, y, z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int n0, int n1, int n2)
{
  // Warning: grid type will be type of last added cell
  _type = triangles;
  
  return gd->createCell(n0, n1, n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int n0, int n1, int n2, int n3)
{
  // Warning: grid type will be type of last added cell
  _type = tetrahedrons;
  
  return gd->createCell(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Node* n0, Node* n1, Node* n2)
{
  // Warning: grid type will be type of last added cell
  _type = triangles;
  
  return gd->createCell(n0, n1, n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Node* n0, Node* n1, Node* n2, Node* n3)
{
  // Warning: grid type will be type of last added cell
  _type = tetrahedrons;
  
  return gd->createCell(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(int n0, int n1)
{
  return gd->createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(Node* n0, Node* n1)
{
  return gd->createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Face* Grid::createFace(int e0, int e1, int e2)
{
  return gd->createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Face* Grid::createFace(Edge* e0, Edge* e1, Edge* e2)
{
  return gd->createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Node* Grid::getNode(int id)
{
  return gd->getNode(id);
}
//-----------------------------------------------------------------------------
Cell* Grid::getCell(int id)
{
  return gd->getCell(id);
}
//-----------------------------------------------------------------------------
Edge* Grid::getEdge(int id)
{
  return gd->getEdge(id);
}
//-----------------------------------------------------------------------------
Face* Grid::getFace(int id)
{
  return gd->getFace(id);
}
//-----------------------------------------------------------------------------
bool Grid::hasEdge(Node* n0, Node* n1) const
{
  return gd->hasEdge(n0, n1);
}
//-----------------------------------------------------------------------------
void Grid::init()
{
  GridInit::init(*this);
}
//-----------------------------------------------------------------------------
void Grid::swap(Grid& grid)
{
  GridData*           tmp_gd     = this->gd;
  BoundaryData*       tmp_bd     = this->bd;
  GridRefinementData* tmp_rd     = this->rd;
  Grid*               tmp_parent = this->_parent;
  Grid*               tmp_child  = this->_child;
  Type                tmp_type   = this->_type;

  this->gd      = grid.gd;
  this->bd      = grid.bd;
  this->rd      = grid.rd;
  this->_parent = grid._parent;
  this->_child  = grid._child;
  this->_type   = grid._type;

  grid.gd      = tmp_gd;
  grid.bd      = tmp_bd;
  grid.rd      = tmp_rd;
  grid._parent = tmp_parent;
  grid._child  = tmp_child;
  grid._type   = tmp_type;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Grid& grid)
{
  stream << "[ Grid with " << grid.noNodes() << " nodes, "
	 << grid.noCells() << " cells ";

  switch ( grid.type() ) {
  case Grid::triangles:
    stream << "(triangles)";
    break;
  case Grid::tetrahedrons:
    stream << "(tetrahedrons)";
    break;
  default:
    stream << "(unknown type)";
  }

  stream << ", and " << grid.noEdges() << " edges ]";

  return stream;
}
//-----------------------------------------------------------------------------
