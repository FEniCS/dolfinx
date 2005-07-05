// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Par Ingelstrom 2004.
//
// First added:  2003
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Node.h>
#include <dolfin/TetMeshRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRule(Cell& cell, int no_marked_edges)
{
  dolfin_assert(cell.type() == Cell::tetrahedron);

  // Choose refinement rule
  
  if ( checkRuleRegular(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular1(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular2(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular3(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular4(cell, no_marked_edges) )
    return true;

  // We didn't find a matching rule for refinement
  return false;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refine(Cell& cell, Mesh& mesh)
{
  // Refine cell according to marker
  switch ( cell.marker() ) {
  case Cell::marked_for_no_ref:
    refineNoRefine(cell, mesh);
    break;
  case Cell::marked_for_reg_ref:
    refineRegular(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_1:
    refineIrregular1(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_2:
    refineIrregular2(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_3:
    refineIrregular3(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_4:
    refineIrregular4(cell, mesh);
    break;
  default:
    // We should not reach this case, cell cannot be
    // marked_for_coarsening or marked_according_to_ref
    dolfin_error("Inconsistent cell markers.");
  }
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRuleRegular(Cell& cell, int no_marked_edges)
{
  // Check if cell should be regularly refined.
  // A cell is refined regularly if all edges are marked.

  if ( no_marked_edges != 6 )
    return false;

  cell.marker() = Cell::marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 1

  if ( no_marked_edges != 3 )
    return false;

  if ( !markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_1;
  return true;
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 2

  if ( no_marked_edges != 1 )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_2;
  return true;
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRuleIrregular3(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 3

  if ( no_marked_edges != 2 )
    return false;

  if ( !markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_3;
  return true;
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::checkRuleIrregular4(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 4

  if ( no_marked_edges != 3 )
    return false;

  // Note that this has already been checked by checkRule3(), but this
  // way the algorithm is a little cleaner.
  if ( markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_4;
  return true;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineNoRefine(Cell& cell, Mesh& mesh)
{
  // Don't refine the tetrahedron and create a copy in the new mesh.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_no_ref);
  
  // Create new nodes with the same coordinates as existing nodes
  Node& n0 = createNode(cell.node(0), mesh, cell);
  Node& n1 = createNode(cell.node(1), mesh, cell);
  Node& n2 = createNode(cell.node(2), mesh, cell);
  Node& n3 = createNode(cell.node(3), mesh, cell);

  // Create a new cell
  cell.initChildren(1);
  createCell(n0, n1, n2, n3, mesh, cell);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::unref;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineRegular(Cell& cell, Mesh& mesh)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_reg_ref);
  
  // Create new nodes with the same coordinates as existing nodes
  Node& n0 = createNode(cell.node(0), mesh, cell);
  Node& n1 = createNode(cell.node(1), mesh, cell);
  Node& n2 = createNode(cell.node(2), mesh, cell);
  Node& n3 = createNode(cell.node(3), mesh, cell);

  // Create new nodes with the new coordinates 
  Node& n01 = createNode(cell.node(0).midpoint(cell.node(1)), mesh, cell);
  Node& n02 = createNode(cell.node(0).midpoint(cell.node(2)), mesh, cell);
  Node& n03 = createNode(cell.node(0).midpoint(cell.node(3)), mesh, cell);
  Node& n12 = createNode(cell.node(1).midpoint(cell.node(2)), mesh, cell);
  Node& n13 = createNode(cell.node(1).midpoint(cell.node(3)), mesh, cell);
  Node& n23 = createNode(cell.node(2).midpoint(cell.node(3)), mesh, cell);

  // Create new cells 
  cell.initChildren(8);
  createCell(n0,  n01, n02, n03, mesh, cell);
  createCell(n01, n1,  n12, n13, mesh, cell);
  createCell(n02, n12, n2,  n23, mesh, cell);
  createCell(n03, n13, n23, n3,  mesh, cell);
  createCell(n01, n02, n03, n13, mesh, cell);
  createCell(n01, n02, n12, n13, mesh, cell);
  createCell(n02, n03, n13, n23, mesh, cell);
  createCell(n02, n12, n13, n23, mesh, cell);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_reg;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular1(Cell& cell, Mesh& mesh)
{
  // Three edges are marked on the same face. Insert three new nodes
  // at the midpoints on the marked edges, connect the new nodes to
  // each other, as well as to the node that is not on the marked
  // face. This gives 4 new tetrahedra.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_1);

  // Sort nodes by the number of marked edges
  PArray<Node*> nodes;
  sortNodes(cell,nodes);
  
  // Create new nodes with the same coordinates as the old nodes
  Node& n0 = createNode(*nodes(0), mesh, cell);
  Node& n1 = createNode(*nodes(1), mesh, cell);
  Node& n2 = createNode(*nodes(2), mesh, cell);
  Node& nn = createNode(*nodes(3), mesh, cell); // Not marked
       
  // Find edges
  Edge* e01 = cell.findEdge(*nodes(0), *nodes(1));
  Edge* e02 = cell.findEdge(*nodes(0), *nodes(2));
  Edge* e12 = cell.findEdge(*nodes(1), *nodes(2));
  dolfin_assert(e01);
  dolfin_assert(e02);
  dolfin_assert(e12);

  // Create new nodes on the edges of the marked face
  Node& n01 = createNode(e01->midpoint(), mesh, cell);
  Node& n02 = createNode(e02->midpoint(), mesh, cell);
  Node& n12 = createNode(e12->midpoint(), mesh, cell);
  
  // Create new cells 
  cell.initChildren(4);
  createCell(nn, n01, n02, n12, mesh, cell);
  createCell(nn, n01, n02, n0,  mesh, cell);
  createCell(nn, n01, n12, n1,  mesh, cell);
  createCell(nn, n02, n12, n2,  mesh, cell);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular2(Cell& cell, Mesh& mesh)
{
  // One edge is marked. Insert one new node at the midpoint of the
  // marked edge, then connect this new node to the two nodes not on
  // the marked edge. This gives 2 new tetrahedra.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_2);

  // Sort nodes by the number of marked edges
  PArray<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node& n0  = createNode(*nodes(0), mesh, cell);
  Node& n1  = createNode(*nodes(1), mesh, cell);
  Node& nn0 = createNode(*nodes(2), mesh, cell); // Not marked
  Node& nn1 = createNode(*nodes(3), mesh, cell); // Not marked

  // Find the marked edge
  Edge* e = cell.findEdge(*nodes(0), *nodes(1));
  dolfin_assert(e);

  // Create new node on marked edge 
  Node& ne = createNode(e->midpoint(), mesh, cell);
  
  // Create new cells
  cell.initChildren(2);
  createCell(ne, nn0, nn1, n0, mesh, cell);
  createCell(ne, nn0, nn1, n1, mesh, cell);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular3(Cell& cell, Mesh& mesh)
{
  // Two edges are marked, both on the same face. There are two
  // possibilities, and the chosen alternative must match the
  // corresponding face of the neighbor tetrahedron. If this neighbor 
  // is marked for regular refinement, so is this tetrahedron. 
  //   
  // We insert two new nodes at the midpoints of the marked edges. 
  // Three new edges are created by connecting the two new nodes to 
  // each other and to the node opposite the face of the two marked 
  // edges. Finally, an edge is created by either
  // 
  //   (1) connecting new node 1 with the endnode of marked edge 2,
  //       that is not common with marked edge 1, or
  //
  //   (2) connecting new node 2 with the endnode of marked edge 1, 
  //       that is not common with marked edge 2.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_3);

  // Sort nodes by the number of marked edges
  PArray<Node*> nodes;
  sortNodes(cell, nodes);

  // Find edges
  Edge* e0 = cell.findEdge(*nodes(0), *nodes(1));
  Edge* e1 = cell.findEdge(*nodes(0), *nodes(2));
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Find common face
  Face* f = cell.findFace(*e0, *e1);
  dolfin_assert(f);

  // Find neighbor with common face
  Cell* neighbor = findNeighbor(cell, *f);
  dolfin_assert(neighbor);

  if ( neighbor->marker() == Cell::marked_for_reg_ref ) {
    // If neighbor is marked for regular refinement so is cell
    refineIrregular31(cell,mesh);
  }
  else if ( neighbor->marker() == Cell::marked_for_irr_ref_3) {
    // If neighbor is marked refinement by rule 3, 
    // just chose an orientation, and it will be up to 
    // the neighbor to make sure the common face match
    refineIrregular32(cell,mesh,nodes);
  }
  else if ( neighbor->marker() == Cell::marked_according_to_ref ) {
    // If neighbor has been refined irregular according to 
    // refinement rule 3, make sure the common face matches
    refineIrregular33(cell, mesh, nodes, *neighbor);
  }
  else {
    // This case shouldn't happen
    dolfin_error("Unable to handle refinement rule of neighbor.");
  }
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular4(Cell& cell, Mesh& mesh)
{
  // Two edges are marked, opposite to each other. We insert two new
  // nodes at the midpoints of the marked edges, insert a new edge
  // between the two nodes, and insert four new edges by connecting
  // the new nodes to the endpoints of the opposite edges.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_4);

  // Find the two marked edges
  PArray<Edge*> marked_edges(2);
  marked_edges = 0;
  int cnt = 0;
  for (EdgeIterator e(cell); !e.end(); ++e)
    if (e->marked()) marked_edges(cnt++) = e;

  // Create new nodes with the same coordinates as the old nodes
  Node& n00 = createNode(marked_edges(0)->node(0), mesh, cell);
  Node& n01 = createNode(marked_edges(0)->node(1), mesh, cell);
  Node& n10 = createNode(marked_edges(1)->node(0), mesh, cell);
  Node& n11 = createNode(marked_edges(1)->node(1), mesh, cell);

  // Create new node on marked edge 
  Node& n_e0 = createNode(marked_edges(0)->midpoint(), mesh, cell);
  Node& n_e1 = createNode(marked_edges(1)->midpoint(), mesh, cell);

  // Create new cells 
  cell.initChildren(4);
  createCell(n_e0, n_e1, n00, n10, mesh, cell);
  createCell(n_e0, n_e1, n00, n11, mesh, cell);
  createCell(n_e0, n_e1, n01, n10, mesh, cell);
  createCell(n_e0, n_e1, n01, n11, mesh, cell);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular31(Cell& cell, Mesh& mesh)
{
  // If neighbor with common 2-marked-edges-face is marked for regular
  // refinement do the same for cell
  cell.marker() = Cell::marked_for_reg_ref;
  refineRegular(cell,mesh);
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular32(Cell& cell, Mesh& mesh, 
					  PArray<Node*>& sorted_nodes)
{
  // If neighbor is marked refinement by rule 3, 
  // just chose an orientation, and it will be up to 
  // the neighbor to make sure the common face match

  // Create new nodes with the same coordinates as the old nodes
  Node& n_dm  = createNode(*sorted_nodes(0), mesh, cell);
  Node& n_m0  = createNode(*sorted_nodes(1), mesh, cell);
  Node& n_m1  = createNode(*sorted_nodes(2), mesh, cell);
  Node& n_nm  = createNode(*sorted_nodes(3), mesh, cell);

  // Create new node on marked edge 
  Node& n_e0 = createNode(sorted_nodes(0)->midpoint(*sorted_nodes(1)), mesh, cell);
  Node& n_e1 = createNode(sorted_nodes(0)->midpoint(*sorted_nodes(2)), mesh, cell);

  // Create new cells 
  cell.initChildren(3);
  createCell(n_dm, n_e0, n_e1, n_nm, mesh, cell);
  createCell(n_m0, n_m1, n_e1, n_nm, mesh, cell);
  createCell(n_e0, n_e1, n_m0, n_nm, mesh, cell);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TetMeshRefinement::refineIrregular33(Cell& cell, Mesh& mesh, 
					  PArray<Node*>& sorted_nodes,
					  Cell& face_neighbor)
{

  // Create new nodes with the same coordinates as the old nodes
  Node& n_dm  = createNode(*sorted_nodes(0), mesh, cell);
  Node& n_m0  = createNode(*sorted_nodes(1), mesh, cell);
  Node& n_m1  = createNode(*sorted_nodes(2), mesh, cell);
  Node& n_nm  = createNode(*sorted_nodes(3), mesh, cell);

  // Create new node on marked edge 
  Node& n_e0 = createNode(sorted_nodes(0)->midpoint(*sorted_nodes(1)), mesh, cell);
  Node& n_e1 = createNode(sorted_nodes(0)->midpoint(*sorted_nodes(2)), mesh, cell);

  // Create new cells
  cell.initChildren(3);
  createCell(n_dm, n_e0, n_e1, n_nm, mesh, cell);
  
  // If neighbor has been refined irregular according to 
  // refinement rule 3, make sure the common face matches
  Cell* c;
  for (int i = 0; i < face_neighbor.noChildren(); i++) {
    c = face_neighbor.child(i);
    if ( !(c->haveNode(n_dm)) ){
      if ( c->haveNode(n_e0) && c->haveNode(n_e1) ){
	if ( c->haveNode(n_m0) ){
	  createCell(n_e0, n_e1, n_m0, n_nm, mesh, cell);
	  createCell(n_m0, n_m1, n_e1, n_nm, mesh, cell);
	  break;
	}
	else{
	  createCell(n_e0, n_e1, n_m1, n_nm, mesh, cell);
	  createCell(n_m0, n_m1, n_e0, n_nm, mesh, cell);
	  break;
	}		
      }
    }
  }
    
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
bool TetMeshRefinement::markedEdgesOnSameFace(Cell& cell)
{
  // Check if the marked edges of cell are on the same face: 
  //
  //   0 marked edge  -> false 
  //   1 marked edge  -> true 
  //   2 marked edges -> true if edges have any common nodes
  //   3 marked edges -> true if there is a face with the marked edges 
  //   4 marked edges -> false 
  //   5 marked edges -> false 
  //   6 marked edges -> false 

  // Count the number of marked edges
  int cnt = 0; 
  for (EdgeIterator e(cell); !e.end(); ++e)
    if (e->marked()) cnt++;

  // Case 0, 1, 4, 5, 6
  dolfin_assert(cnt >= 0 && cnt <= 6);
  if (cnt == 0) return false;
  if (cnt == 1) return true;
  if (cnt > 3)  return false;
  
  // Create a list of the marked edges
  PArray<Edge*> marked_edges(cnt);
  marked_edges = 0;
  cnt = 0; 
  for (EdgeIterator e(cell); !e.end(); ++e)
    if (e->marked()) marked_edges(cnt++) = e;

  // Check that number of marked edges are consistent  
  dolfin_assert(cnt == 2 || cnt == 3);

  // Case 2
  if (cnt == 2){
    if (marked_edges(0)->contains(marked_edges(1)->node(0)) || 
	marked_edges(0)->contains(marked_edges(1)->node(1)))
      return true;
    return false;
  }
  
  // Case 3
  if (cnt == 3){
    for (FaceIterator f(cell); !f.end(); ++f){
      if (f->equals(*marked_edges(0), *marked_edges(1), *marked_edges(2)))
	return true;
    }
    return false;
  }
  
  // We shouldn't reach this case
  dolfin_error("Inconsistent edge markers.");
  return false;
}
//-----------------------------------------------------------------------------
Cell* TetMeshRefinement::findNeighbor(Cell& cell, Face& face)
{
  // Find a cell neighbor sharing a common face

  for (CellIterator c(cell); !c.end(); ++c) {

    // Don't check the cell itself
    if ( c->id() == cell.id() )
      continue;
    for (FaceIterator f(*c); !f.end(); ++f) {
      if ( f->id() == face.id() ) {
	return c;
      }
    }
  }

  // If no neighbor is found, return the cell itself
  return &cell;
}
//-----------------------------------------------------------------------------
Cell& TetMeshRefinement::createCell(Node& n0, Node& n1, Node& n2, Node& n3,
				    Mesh& mesh, Cell& cell)
{
  Cell& c = mesh.createCell(n0, n1, n2, n3);
  c.setParent(cell);
  cell.addChild(c);
  
  return c;
}
//-----------------------------------------------------------------------------
Cell& TetMeshRefinement::createChildCopy(Cell& cell, Mesh& mesh)
{
  return createCell(*cell.node(0).child(), *cell.node(1).child(),
		    *cell.node(2).child(), *cell.node(3).child(), mesh, cell);
}
//-----------------------------------------------------------------------------
