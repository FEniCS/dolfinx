// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
// Modified by Garth N. Wells 2006.
//
// First added:  2006-06-05
// Last changed: 2006-12-06

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/Tetrahedron.h>

using namespace dolfin;

// Alignment convention for edges according to the DOLFIN manual (first vertex)
static int edge_alignment[6] = {1, 2, 0, 0, 1, 2};

// Alignment convention for faces according to the DOLFIN manual
static int face_alignment_00[4] = {5, 3, 2, 0}; // Edge 0 for alignment 0
static int face_alignment_01[4] = {0, 1, 3, 1}; // Edge 1 for alignment 0 (otherwise 1)
static int face_alignment_10[4] = {0, 1, 3, 1}; // Edge 0 for alignment 2
static int face_alignment_11[4] = {4, 5, 4, 2}; // Edge 1 for alignment 2 (otherwise 3)
static int face_alignment_20[4] = {4, 5, 4, 2}; // Edge 0 for alignment 4
static int face_alignment_21[4] = {5, 3, 2, 0}; // Edge 1 for alignment 4 (otherwise 5)

//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::dim() const
{
  return 3;
}
//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 4; // vertices
  case 1:
    return 6; // edges
  case 2:
    return 4; // faces
  case 3:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // faces
  case 3:
    return 4; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::alignment(const Cell& cell, uint dim, uint e) const
{
  // Compute alignment according the convention in the DOLFIN manual
  if ( dim == 1 )
  {
    // Compute alignment of given edge by checking first vertex
    const uint* edge_vertices = cell.mesh().topology()(dim, 0)(cell.entities(dim)[e]);
    const uint* cell_vertices = cell.entities(0);
    return ( edge_vertices[0] == cell_vertices[edge_alignment[e]] ? 0 : 1 );
  }
  else if ( dim == 2 )
  {
    // Compute alignment of given face by checking the first two edges
    const uint* face_edges = cell.mesh().topology()(dim, 1)(cell.entities(dim)[e]);
    const uint* cell_edges = cell.entities(1);
    if ( face_edges[0] == cell_edges[face_alignment_00[e]] )
      return ( face_edges[1] == cell_edges[face_alignment_01[e]] ? 0 : 1 );
    else if ( face_edges[0] == cell_edges[face_alignment_10[e]] )
      return ( face_edges[1] == cell_edges[face_alignment_11[e]] ? 2 : 3 );
    else if ( face_edges[0] == cell_edges[face_alignment_20[e]] )
      return ( face_edges[1] == cell_edges[face_alignment_21[e]] ? 4 : 5 );
    dolfin_error("Unable to compute alignment of tetrahedron.");
  }
  else
    dolfin_error("Unable to compute alignment for entity of dimension %d for tetrehdron.");
  
  return 0;
}
//-----------------------------------------------------------------------------
void Tetrahedron::createEntities(uint** e, uint dim, const uint v[]) const
{
  // We only need to know how to create edges and faces
  switch ( dim )
  {
  case 1:
    // Create the six edges
    e[0][0] = v[1]; e[0][1] = v[2];
    e[1][0] = v[2]; e[1][1] = v[0];
    e[2][0] = v[0]; e[2][1] = v[1];
    e[3][0] = v[0]; e[3][1] = v[3];
    e[4][0] = v[1]; e[4][1] = v[3];
    e[5][0] = v[2]; e[5][1] = v[3];
    break;
  case 2:
    // Create the four faces
    e[0][0] = v[1]; e[0][1] = v[3]; e[0][2] = v[2];
    e[1][0] = v[2]; e[1][1] = v[3]; e[1][2] = v[0];
    e[2][0] = v[3]; e[2][1] = v[1]; e[2][2] = v[0];
    e[3][0] = v[0]; e[3][1] = v[1]; e[3][2] = v[2];
    break;
  default:
    dolfin_error1("Don't know how to create entities of topological dimension %d.", dim);
  }
}
//-----------------------------------------------------------------------------
void Tetrahedron::refineCell(Cell& cell, MeshEditor& editor,
			     uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the ten new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint v3 = v[3];
  const uint e0 = offset + e[0];
  const uint e1 = offset + e[1];
  const uint e2 = offset + e[2];
  const uint e3 = offset + e[3];
  const uint e4 = offset + e[4];
  const uint e5 = offset + e[5];

  // Regular refinement: 8 new cells
  editor.addCell(current_cell++, v0, e1, e3, e2);
  editor.addCell(current_cell++, v1, e2, e4, e0);
  editor.addCell(current_cell++, v2, e0, e5, e1);
  editor.addCell(current_cell++, v3, e5, e4, e3);
  editor.addCell(current_cell++, e0, e4, e5, e1);
  editor.addCell(current_cell++, e1, e5, e3, e4);
  editor.addCell(current_cell++, e2, e3, e4, e1);
  editor.addCell(current_cell++, e0, e1, e2, e4);
}
//-----------------------------------------------------------------------------
void Tetrahedron::refineCellIrregular(Cell& cell, MeshEditor& editor, 
				      uint& current_cell, uint refinement_rule, 
				      uint* marked_edges) const
{
  dolfin_warning("Not implemented yet.");

  /*
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the ten new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint v3 = v[3];
  const uint e0 = offset + e[0];
  const uint e1 = offset + e[1];
  const uint e2 = offset + e[2];
  const uint e3 = offset + e[3];
  const uint e4 = offset + e[4];
  const uint e5 = offset + e[5];

  // Refine according to refinement rule
  // The rules are numbered according to the paper: 
  // J. Bey, "Tetrahedral Grid Refinement", 1995.   
  switch ( refinement_rule )
  {
  case 1:
    // Rule 1: 4 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    editor.addCell(current_cell++, v3, e5, e4, e3);
    break;
  case 2:
    // Rule 2: 2 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    break;
  case 3:
    // Rule 3: 3 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    break;
  case 4:
    // Rule 4: 4 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    editor.addCell(current_cell++, v3, e5, e4, e3);
    break;
  default:
    dolfin_error1("Illegal rule for irregular refinement of tetrahedron.");
  }
  */
}
//-----------------------------------------------------------------------------
real Tetrahedron::volume(const MeshEntity& tetrahedron) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if ( geometry.dim() != 3 )
    dolfin_error("Only know how to compute the volume of a tetrahedron when embedded in R^3.");

  // Get the coordinates of the four vertices
  const uint* vertices = tetrahedron.entities(0);
  const real* x0 = geometry.x(vertices[0]);
  const real* x1 = geometry.x(vertices[1]);
  const real* x2 = geometry.x(vertices[2]);
  const real* x3 = geometry.x(vertices[3]);

  // Formula for volume from http://mathworld.wolfram.com
  real v = ( x0[0] * ( x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2] - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2] ) -
             x1[0] * ( x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2] - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2] ) +
             x2[0] * ( x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2] - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2] ) -
             x3[0] * ( x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2] - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2] ) );
  
  return std::abs(v) / 6.0;
}
//-----------------------------------------------------------------------------
real Tetrahedron::diameter(const MeshEntity& tetrahedron) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if ( geometry.dim() != 3 )
    dolfin_error("Only know how to compute the diameter of a tetrahedron when embedded in R^3.");
  
  // Get the coordinates of the four vertices
  const uint* vertices = tetrahedron.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);
  Point p2 = geometry.point(vertices[2]);
  Point p3 = geometry.point(vertices[3]);

  // Compute side lengths
  real a  = p1.distance(p2);
  real b  = p0.distance(p2);
  real c  = p0.distance(p1);
  real aa = p0.distance(p3);
  real bb = p1.distance(p3);
  real cc = p2.distance(p3);
                                
  // Compute "area" of triangle with strange side lengths
  real la   = a*aa;
  real lb   = b*bb;
  real lc   = c*cc;
  real s    = 0.5*(la+lb+lc);
  real area = sqrt(s*(s-la)*(s-lb)*(s-lc));
                                
  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  return area / ( 3.0*volume(tetrahedron) );
}
//-----------------------------------------------------------------------------
real Tetrahedron::normal(const Cell& cell, uint facet, uint i) const
{
  dolfin_error("Not implemented. Please fix this Kristian. ;-)");
  return 0.0;
}
//-----------------------------------------------------------------------------
std::string Tetrahedron::description() const
{
  std::string s = "tetrahedron (simplex of topological dimension 3)";
  return s;
}
//-----------------------------------------------------------------------------
