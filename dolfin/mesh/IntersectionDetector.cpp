// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson, 2006.
// Modified by Ola Skavhaug, 2006.
// Modified by Dag Lindbo, 2008.
// Modified by Kristoffer Selim, 2009.
//
// First added:  2006-06-21
// Last changed: 2009-01-14

#include <algorithm>
#include <map>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "Mesh.h"
#include "Edge.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"
#include "GTSInterface.h"
#include "IntersectionDetector.h"

using namespace dolfin;

#ifdef HAS_GTS

//-----------------------------------------------------------------------------
IntersectionDetector::IntersectionDetector(const Mesh& mesh0) : gts(0), mesh0(mesh0)
{
  gts = new GTSInterface(mesh0);
}
//-----------------------------------------------------------------------------
IntersectionDetector::~IntersectionDetector()
{
  delete gts;
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p, std::vector<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(p, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Point& p0, const Point& p1, std::vector<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(p0, p1, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Cell& c, std::vector<uint>& cells)
{
  dolfin_assert(gts);
  gts->intersection(c, cells);
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Array<Point>& points, std::vector<uint>& cells) 
{
  // Intersect each segment with mesh
  std::vector<uint> cc;
  for (uint i = 1; i < points.size(); i++)
    gts->intersection(points[i - 1], points[i], cc);

  // Remove repeated cells
  std::sort(cc.begin(), cc.end());
  Array<unsigned int>::iterator it;
  it = std::unique(cc.begin(), cc.end());
  cc.resize(it - cc.begin());  
}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Mesh& mesh1, std::vector<uint>& cells)
{
  // For testing
  //new_intersection(mesh1, cells);
  //return;

  // Intersect each cell with mesh
  for (CellIterator cell(mesh1); !cell.end(); ++cell)
    intersection(*cell, cells);
  
  // Remove repeated cells
  std::sort(cells.begin(), cells.end());
  Array<unsigned int>::iterator it;
  it = std::unique(cells.begin(), cells.end());
  cells.resize(it - cells.begin());
}
//-----------------------------------------------------------------------------
void IntersectionDetector::new_intersection(const Mesh& mesh1,
                                            std::vector<uint>& cells)
{
  // Note that two different meshes are used in this function:
  //
  // 1. mesh0 (the mesh for which the detector was created)
  //
  // This mesh is the mesh that we are intersecting
  //
  // 2. mesh1 (input mesh)
  //
  // This mesh is the mesh that we are intersecting with. Typically,
  // this input mesh will be the boundary of some other mesh, see 
  // /demo/mesh/intersection/python/demo.py . 

  // Intersect each cell with mesh
  for (CellIterator cell(mesh1); !cell.end(); ++cell)
    intersection(*cell, cells);
  
  // Remove repeated cells
  std::sort(cells.begin(), cells.end());
  std::vector<unsigned int>::iterator it;
  it = std::unique(cells.begin(), cells.end());
  cells.resize(it - cells.begin());
  
  // Map from cell numbers in mesh0 to intersecting cells in mesh1
  std::map<uint, std::vector<uint> > cell_intersections;
  typedef std::map<uint, std::vector<uint> >::iterator map_iterator;

  // Compute which cells in mesh1 intersect each cell in mesh0
  for (CellIterator c1(mesh1); !c1.end(); ++c1)
  {
    cout << endl;
    cout << "Computing intersection with facet in Omega1: " << *c1 << endl;

    // Compute which cells are intersected by c1
    std::vector<uint> intersected_cells;
    intersection(*c1, intersected_cells);

    cout << "Found " << intersected_cells.size() << " cells in Omega0" << endl;

    // Iterate over intersected cells and add intersection
    for (uint i = 0; i < intersected_cells.size(); i++)
    {
      // Get the cell index
      const uint c0 = intersected_cells[i];

      // Check if cell has been intersected before
      map_iterator it = cell_intersections.find(c0);
      if (it == cell_intersections.end())
      {
        // Add c1 to intersections for c0
        std::vector<uint> intersections;
        intersections.push_back(c1->index());
        cell_intersections[c0] = intersections;
      }
      else
      {
        // Add c1 to intersections for c0
        it->second.push_back(c1->index());
      }
    }
  }

  // Print intersections for all intersected cells
  for (map_iterator it = cell_intersections.begin(); it != cell_intersections.end(); ++it)
  {
    const Cell c0(mesh0, it->first);
    compute_polygon(mesh1, c0, it->second);
  }
  
}
//-----------------------------------------------------------------------------
void IntersectionDetector::compute_polygon(const Mesh& mesh1,
                                           const Cell& c0,
                                           const std::vector<uint>& intersections) const
{
  dolfin_assert(intersections.size() > 0);

  // Can only handle triangles and edges
  if (c0.mesh().topology().dim() != 2)
    error("Sorry, can only handle intersection with edges.");
  if (mesh1.topology().dim() != 1)
    error("Sorry, can only handle intersection of triangles."); 

  // Some debugging information
  cout << "Cell no: " << c0.index() << endl;
  cout << "is intersected with: " << endl;
  
  for (uint i = 0; i < intersections.size(); i++)
  {
    Edge edge(mesh1, intersections[i]);
    cout << "Edge: " << edge << endl; 
    for (VertexIterator v(edge); !v.end(); ++v)
      cout << "  x = " << v->point() << endl;

    cout << "wich has the following neigbours: " << endl;
    for (EdgeIterator e(edge); !e.end(); ++e)
      cout << "  " << *e << endl;
  }

  // Prepare list of polygon points
  std::vector<std::vector<double> > points;
  
  // Prepare sets of edge indices
  std::set<uint> intersecting_edges;
  std::set<uint> visited_edges;
  typedef std::set<uint>::iterator set_iterator;
  
  // Add all intersecting edges to set
  for (uint i = 0; i < intersections.size(); i++)
    intersecting_edges.insert(intersections[i]);
  
   // Find starting edge with at most one intersecting neighbouring edge
  uint current_index = 0;
  bool found_edge = false;

  for (uint i = 0; i < intersections.size(); i++)
  {
    current_index = intersections[0];    
    Edge current_edge(mesh1, current_index);
    
    // Count number of intersecting neighbouring edges
    uint num_neighbours = 0;
    for (EdgeIterator e(current_edge); !e.end(); ++e)
    {
      if (intersecting_edges.count(e->index()) > 0)
        num_neighbours += 1;
    }
    
    // Want to find edge with at most one intersecting neighbouring edge
    if (num_neighbours < 2)
    {
      found_edge = true;
      break;
    }
  }
  
  if (!found_edge)
    error("Unable to find first edge!");
  
  cout <<"*************************"<< endl;
  cout << "Start to walk along edge" << endl;
  cout <<"*************************"<< endl;

  while (true)
  {
    // Create current edge
    Edge current_edge(mesh1, current_index);
    visited_edges.insert(current_index);
    cout << "Current edge: " << current_edge << endl;
    cout << "with corresponding vertices " << endl;
    
    // Get vertex coordinates
    std::vector<std::vector<double> > x;
    std::vector<double> xx(2);  
    std::pair<double, double> test;

    for (VertexIterator v(current_edge); !v.end(); ++v)

    {
      std::vector<double> xx(2);
      xx[0] = v->x()[0];
      xx[1] = v->x()[1]; 
      
      cout << "xx[0] = "<< xx[0] << ",  " << "xx[1] =  " << xx[1] << endl;
      x.push_back(xx);      
    }
    
    points.push_back(x[0]);
    points.push_back(x[1]);       
        
    // Find next neighbour
    uint next_index = current_index;
    for (EdgeIterator e(current_edge); !e.end(); ++e)
    {
      if (intersecting_edges.count(e->index()) > 0 &&
          visited_edges.count(e->index()) == 0)
      {
        next_index = e->index();
        break;
      }
    }

    // Check if not found
    if (next_index == current_index)
    {
      cout <<"*******************************"<< endl;
      cout << "No more edges, walk ends here" << endl;
      cout <<"*******************************"<< endl;
      break;
    }
    current_index = next_index;  
  }

  cout << "Polygon points " << endl;
  for (uint i = 0; i < points.size(); i++)
  {
    cout << "i = " << i << ": x = (" << points[i][0] << ", " << points[i][1] << ")" << endl;
  }      
}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
IntersectionDetector::IntersectionDetector(const Mesh& mesh)
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
void IntersectionDetector::intersection(const Cell& c, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(Array<Point>& points, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------
void IntersectionDetector::intersection(const Mesh& mesh, Array<uint>& intersection) {}
//-----------------------------------------------------------------------------

#endif
