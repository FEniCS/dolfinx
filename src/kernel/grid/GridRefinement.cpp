// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridIterator.h>
#include <dolfin/GridRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void GridRefinement::refine(GridHierarchy& grids)
{
  // Write a message
  dolfin_start("Refining grid:");
  cout << grids.fine().rd->noMarkedCells()
       << " cells marked for refinement." << endl;
  
  // Init marks for the finest grid (others already exist)
  initMarks(grids.fine());

  // Refine grid hierarchy
  globalRefinement(grids);

  // Write a message
  cout << "Grid hierarchy consists of " << grids.size() << " grids." << endl;

  dolfin_end();
}
//-----------------------------------------------------------------------------
void GridRefinement::createFineGrid(GridHierarchy& grids)
{
  // Create the new grid
  //Grid* grid = new Grid(grids.fine());

  // Add the grid to the grid hierarchy
  //grids.add(*grid);
}
//-----------------------------------------------------------------------------
void GridRefinement::globalRefinement(GridHierarchy& grids)
{
  // The global grid refinement algorithm working on the whole grid hierarchy.
  // This is algorithm GlobalRefinement() in Beys paper.

  // Phase I: Visit all grids top-down
  for (GridIterator grid(grids,last); !grid.end(); --grid) {
    evaluateMarks(*grid);
    closeGrid(*grid);
  }
  
  // Phase II: Vist all grids bottom-up
  for (GridIterator grid(grids); !grid.end(); ++grid) {
    if (grid.index() > 0)
      closeGrid(*grid);
    unrefineGrid(*grid);
    refineGrid(*grid);
  }

  // FIXME: k = k + 1 ?

}
//-----------------------------------------------------------------------------
void GridRefinement::initMarks(Grid& grid)
{
  // Make sure that all cells have markers
  for (CellIterator c(grid); !c.end(); ++c)
    c->initMarker();

  // Make sure that all edges have markers
  for (EdgeIterator e(grid); !e.end(); ++e)
    e->initMarker();

  // Set markers for cells and edges
  for (List<Cell*>::Iterator c(grid.rd->marked_cells); !c.end(); ++c) {

    // Mark cell for regular refinement
    (*c)->marker() = marked_for_reg_ref;

    // Mark edges of the cell
    for (EdgeIterator e(**c); !e.end(); ++e)
      e->mark(**c);

  }
}
//-----------------------------------------------------------------------------
void GridRefinement::evaluateMarks(Grid& grid)
{
  // Evaluate and adjust marks for a grid.
  // This is algorithm EvaluateMarks() in Beys paper.

  for (CellIterator c(grid); !c.end(); ++c) {

    if ( c->status() == ref_reg && childrenMarkedForCoarsening(*c) )
      c->marker() = marked_for_no_ref;
    
    if ( c->status() == ref_irr ) {
      
      if ( edgeOfChildMarkedForRefinement(*c) )
	c->marker() = marked_for_reg_ref;
      else
	c->marker() = marked_for_no_ref;

    }
   
  }
}
//-----------------------------------------------------------------------------
void GridRefinement::closeGrid(Grid& grid)
{
  // Perform the green closer on a grid.
  // This is algorithm CloseGrid() in Bey's paper.

  // Create a list of all elements that need to be closed
  List<Cell*> cells;
  for (CellIterator c(grid); !c.end(); ++c)
    if ( edgeMarkedByOther(*c) )
      cells.add(c);

  // Repeat until the list of elements is empty
  while ( !cells.empty() ) {

    // Get first cell and remove it from the list
    Cell* cell = cells.pop();

    // Close cell
    closeCell(*cell);

  }
  
}
//-----------------------------------------------------------------------------
void GridRefinement::refineGrid(Grid& grid)
{
  // Refine a grid according to marks.
  // This is algorithm RefineGrid() in Bey's paper.

  // Change markers from marked_for_coarsening to marked_for_no_ref
  for (CellIterator c(grid); c.end(); ++c)
    if ( c->marker() == marked_for_coarsening )
      c->marker() = marked_for_no_ref;
  
  // Refine other cells
  for (CellIterator c(grid); !c.end(); ++c) {
    
    // Skip cells which are marked_according_to_ref
    if ( c->marker() == marked_according_to_ref )
      continue;

    // Refine cell according to refinement
    switch ( c->marker() ) {
    case marked_for_reg_ref:
      regularRefinement(*c, grid);
      break;
    case marked_for_irr_ref_1:
      irregularRefinementRule1(*c, grid);
      break;
    case marked_for_irr_ref_2:
      irregularRefinementRule2(*c, grid);
      break;
    case marked_for_irr_ref_3:
      irregularRefinementRule3(*c, grid);
      break;
    case marked_for_irr_ref_4:
      irregularRefinementRule4(*c, grid);
      break;
    case marked_for_no_ref:
      // Do nothing
      break;
    default:
      // We should not rearch this case, cannot be
      // marked_for_coarsening or marked_according_to_ref
      dolfin_error("Inconsistent cell markers.");
    }

  }

}
//-----------------------------------------------------------------------------
void GridRefinement::unrefineGrid(Grid& grid)
{
  // Unrefine a grid according to marks.
  // This is algorithm UnrefineGrid() in Bey's paper.

  
}
//-----------------------------------------------------------------------------
void GridRefinement::closeCell(Cell& cell)
{
  // Close a cell, either by regular or irregular refinement.
  // This is algorithm CloseElement() in Bey's paper.


  
}
//-----------------------------------------------------------------------------
void GridRefinement::regularRefinement(Cell& cell, Grid& grid)
{
  // Regular refinement:
  //
  //     Triangles:    1 -> 4 
  //     Tetrahedrons: 1 -> 8 

  switch (cell.type()) {
  case Cell::triangle: 
    regularRefinementTri(cell,grid);
    break;
  case Cell::tetrahedron: 
    regularRefinementTet(cell,grid);
    break;
  default: 
    dolfin_error("Unknown cell type, unable to refine cell.");
  }

  //  parent->setStatus(Cell::REFINED_REGULAR);
  //  if (parent->marker() == Cell::MARKED_FOR_REGULAR_REFINEMENT) 
  //    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}
//-----------------------------------------------------------------------------
void GridRefinement::regularRefinementTri(Cell& cell, Grid& grid)
{
  // Refine 1 triangle into 4 new ones, introducing new nodes 
  // at the midpoints of the edges. 

  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node *n0 = grid.createNode(cell.coord(0));
  Node *n1 = grid.createNode(cell.coord(1));
  Node *n2 = grid.createNode(cell.coord(2));

  // Update parent-child info 
  n0->setParent(cell.node(0));
  n1->setParent(cell.node(1));
  n2->setParent(cell.node(2));

  cell.node(0)->setParent(n0);
  cell.node(1)->setParent(n1);
  cell.node(2)->setParent(n2);

  // Create new nodes with the new coordinates 
  Node *n01 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(1)->coord()));
  Node *n02 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(2)->coord()));
  Node *n12 = grid.createNode(cell.node(1)->coord().midpoint(cell.node(2)->coord()));

  // Create new cells 
  Cell *t1 = grid.createCell(n0, n01,n02);
  Cell *t2 = grid.createCell(n01,n1, n12);
  Cell *t3 = grid.createCell(n02,n12,n2 );
  Cell *t4 = grid.createCell(n01,n12,n02);

  // Update parent-child info 
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);

  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
}
//-----------------------------------------------------------------------------
void GridRefinement::regularRefinementTet(Cell& cell, Grid& grid)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges.
  
  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node *n0 = grid.createNode(cell.coord(0));
  Node *n1 = grid.createNode(cell.coord(1));
  Node *n2 = grid.createNode(cell.coord(2));
  Node *n3 = grid.createNode(cell.coord(3));

  // Update parent-child info 
  n0->setParent(cell.node(0));
  n1->setParent(cell.node(1));
  n2->setParent(cell.node(2));
  n3->setParent(cell.node(3));

  // Create new nodes with the new coordinates 
  Node *n01 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(1)->coord()));
  Node *n02 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(2)->coord()));
  Node *n03 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(3)->coord()));
  Node *n12 = grid.createNode(cell.node(1)->coord().midpoint(cell.node(2)->coord()));
  Node *n13 = grid.createNode(cell.node(1)->coord().midpoint(cell.node(3)->coord()));
  Node *n23 = grid.createNode(cell.node(2)->coord().midpoint(cell.node(3)->coord()));

  // Create new cells 
  Cell *t1 = grid.createCell(n0, n01,n02,n03);
  Cell *t2 = grid.createCell(n01,n1, n12,n13);
  Cell *t3 = grid.createCell(n02,n12,n2, n23);
  Cell *t4 = grid.createCell(n03,n13,n23,n3 );
  Cell *t5 = grid.createCell(n01,n02,n03,n13);
  Cell *t6 = grid.createCell(n01,n02,n12,n13);
  Cell *t7 = grid.createCell(n02,n03,n13,n23);
  Cell *t8 = grid.createCell(n02,n12,n13,n23);

  // Update parent-child info 
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);
  t5->setParent(&cell);
  t6->setParent(&cell);
  t7->setParent(&cell);
  t8->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
  cell.addChild(t5);
  cell.addChild(t6);
  cell.addChild(t7);
  cell.addChild(t8);
}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tri(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tet(Cell& cell, Grid& grid)
{
  // 3 edges are marked on the same face: 
  // insert 3 new nodes at the midpoints on the marked edges, connect the 
  // new nodes to each other, as well as to the node that is not on the 
  // marked face. This gives 4 new tetrahedrons. 

  // Check if the cell is marked correctly 
  dolfin_assert(cell.status() == ref_irr_by_1);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_m1 = grid.createNode(cell.markedNode(0).coord());
  Node *n_m2 = grid.createNode(cell.markedNode(1).coord());
  Node *n_m3 = grid.createNode(cell.markedNode(2).coord());
  Node *n_nm = grid.createNode(cell.nonMarkedNode(0).coord());
         
  // Update parent-child info
  n_m1->setParent(cell.markedNode(0));
  n_m2->setParent(cell.markedNode(1));
  n_m3->setParent(cell.markedNode(2));
  n_nm->setParent(cell.nonMarkedNode(0));

  // Create new nodes on the edges of the marked face
  Node *n_e12 = grid.createNode(cell.findEdge(n_m1,n_m2).midpoint());
  Node *n_e13 = grid.createNode(cell.findEdge(n_m1,n_m3).midpoint());
  Node *n_e23 = grid.createNode(cell.findEdge(n_m2,n_m3).midpoint());
  
  // Create new cells 
  Cell *t1 = grid.createCell(n_nm,n_e12,n_e13,n_e23);
  Cell *t2 = grid.createCell(n_nm,n_e12,n_e13,n_m1);
  Cell *t3 = grid.createCell(n_nm,n_e12,n_e23,n_m2);
  Cell *t4 = grid.createCell(n_nm,n_e13,n_e23,n_m3);
  
  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tri(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tet(Cell& cell, Grid& grid)
{
  // 1 edge is marked:
  // Insert 1 new node at the midpoint of the marked edge, then connect 
  // this new node to the 2 nodes not on the marked edge. This gives 2 new 
  // tetrahedrons. 
  
  // Check if the cell is marked correctly 
  dolfin_assert(cell.status() == ref_irr_by_2);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_m1  = grid.createNode(cell.markedNode(0).coord());
  Node *n_m2  = grid.createNode(cell.markedNode(1).coord());
  Node *n_nm1 = grid.createNode(cell.nonMarkedNode(0).coord());
  Node *n_nm2 = grid.createNode(cell.nonMarkedNode(1).coord());

  // Update parent-child info
  n_m1->setParent(cell.markedNode(0));
  n_m2->setParent(cell.markedNode(1));
  n_nm1->setParent(cell.nonMarkedNode(0));
  n_nm2->setParent(cell.nonMarkedNode(2));

  // Create new node on marked edge 
  Node *n_e = grid.createNode(cell.findEdge(n_m1,n_m2).midpoint());
  
  // Create new cells 
  Cell *t1 = grid.createCell(n_e,n_nm1,n_nm2,n_m1);
  Cell *t2 = grid.createCell(n_e,n_nm1,n_nm2,n_m2);
  
  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tri(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tet(Cell& cell, Grid& grid)
{
  // 2 edges are marked, on the same face 
  // (here there are 2 possibilities, and the chosen 
  // alternative must match the corresponding face 
  // of the neighbor tetrahedron): 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 3 new edges by connecting the two new nodes to each 
  // other and the node opposite the face of the 2 marked edges, 
  // and insert 1 new edge by 
  // alt.1: connecting new node 1 with the endnode of marked edge 2, 
  // that is not common with marked edge 1, or 
  // alt.2: connecting new node 2 with the endnode of marked edge 1, 
  // that is not common with marked edge 2.

  // Check if the cell is marked correctly 
  dolfin_assert(cell.marker.status()==CellMarker::ref_irr_by_3);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_nm = grid.createNode(cell.nonMarkedNode(0).coord());
  Node *n_dm = grid.createNode(cell.markedNode(0).coord()); 
  // (assuming that the node that is marked by two edges has the lowest index) 
  Node *n_m1 = grid.createNode(cell.markedNode(1).coord());
  Node *n_m2 = grid.createNode(cell.markedNode(2).coord());

  // Update parent-child info
  n_nm->setParent(cell.nonMarkedNode(0).coord());
  n_dm->setParent(cell.markedNode(0).coord()); 
  n_m1->setParent(cell.markedNode(1).coord()); 
  n_m2->setParent(cell.markedNode(2).coord()); 

  // Create new node on marked edge 
  Node *n_e1 = grid.createNode(cell.markedEdge(0).midpoint());
  Node *n_e2 = grid.createNode(cell.markedEdge(1).midpoint());

  // Create new cells 
  Cell *t1;
  Cell *t2;
  Cell *t3;

  // Check neighbor face to marked face to refine correctly 

  //    t1 = grid.createCell(n_e1,n_e2,n_e11,n_e21);
  //    t2 = grid.createCell(n_e1,n_e2,n_e11,n_e22);
  //    t3 = grid.createCell(n_e1,n_e2,n_e12,n_e21);

  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tri(Cell& c, Grid& grid)
{

}
//-----------------------------------------------------------------------------
void GridRefinement::irregularRefinementRule1Tet(Cell& cell, Grid& grid)
{
  // 2 edges are marked, opposite to each other: 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 4 new edges by connecting the new nodes to the 
  // endpoints of the opposite edges of the respectively new nodes. 

  // Check if the cell is marked correctly 
  dolfin_assert(cell.marker.status()==CellMarker::ref_irr_by_4);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_e11 = grid.createNode(cell.markedEdge(0).node(0).coord());
  Node *n_e12 = grid.createNode(cell.markedEdge(0).node(1).coord());
  Node *n_e21 = grid.createNode(cell.markedEdge(1).node(0).coord());
  Node *n_e22 = grid.createNode(cell.markedEdge(1).node(1).coord());

  // Update parent-child info
  n_e11->setParent(cell.markedEdge(0).node(0).coord());
  n_e12->setParent(cell.markedEdge(0).node(1).coord());
  n_e21->setParent(cell.markedEdge(1).node(0).coord());
  n_e22->setParent(cell.markedEdge(1).node(1).coord());

  // Create new node on marked edge 
  Node *n_e1 = grid.createNode(cell.markedEdge(0).midpoint());
  Node *n_e2 = grid.createNode(cell.markedEdge(1).midpoint());

  // Create new cells 
  Cell *t1 = grid.createCell(n_e1,n_e2,n_e11,n_e21);
  Cell *t2 = grid.createCell(n_e1,n_e2,n_e11,n_e22);
  Cell *t3 = grid.createCell(n_e1,n_e2,n_e12,n_e21);
  Cell *t4 = grid.createCell(n_e1,n_e2,n_e12,n_e22);

  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
}
//-----------------------------------------------------------------------------
bool GridRefinement::childrenMarkedForCoarsening(Cell& cell)
{
  for (int i = 0; i < cell.noChildren(); i++)
    if ( cell.child(i)->marker() != marked_for_coarsening )
      return false;
    
  return true;
}
//-----------------------------------------------------------------------------
bool GridRefinement::edgeOfChildMarkedForRefinement(Cell& cell)
{
  for (int i = 0; i < cell.noChildren(); i++)
    for (EdgeIterator e(*cell.child(i)); !e.end(); ++e)
      if ( e->marked() )
	return true;

  return false;
}
//-----------------------------------------------------------------------------
bool GridRefinement::edgeMarkedByOther(Cell& cell)
{
  for (EdgeIterator e(cell); !e.end(); ++e)
    if ( e->marked() )
      return true;

  return false;
}





/*

void GridRefinement::unrefineGrid(int grid_level)
{
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c){
    if (c->level() == grid_level) cells.add(c);
  }
      
  List<Cell *> ccells;
  for (CellIterator c(grid); !c.end(); ++c){
    if (c->level() == grid_level+1) ccells.add(c);
  }
      
  for (List<Cell *>::Iterator c(ccells); !c.end(); ++c){
    (*c.pointer())->setMarkedForReUse(false);
    for (int i=0;i<(*c.pointer())->noNodes();i++) (*c.pointer())->node(i)->setMarkedForReUse(false);
    for (int i=0;i<(*c.pointer())->noEdges();i++) (*c.pointer())->edge(i)->setMarkedForReUse(false);
  }

  for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
    if ((*c.pointer())->marker() == Cell::MARKED_ACCORDING_TO_REFINEMENT){
      for (int i=0;i<(*c.pointer())->noChildren();i++){
	(*c.pointer())->child(i)->setMarkedForReUse(true);
      }
    }
  }

  for (List<Cell *>::Iterator c(ccells); !c.end(); ++c){
    //if (!(*c.pointer())->markedForReUse()) grid.removeCell((*c.pointer()));
  }
  
  List<Node *> cnodes;
  for (NodeIterator n(grid); !n.end(); ++n){
    if (n->level() == grid_level+1) cnodes.add(n);
  }
  for (List<Node*>::Iterator n(cnodes); !n.end(); ++n){
    //if (!(*n.pointer())->markedForReUse()) grid.removeNode((*n.pointer()));
  }

  List<Edge *> cedges;
  for (EdgeIterator e(grid); !e.end(); ++e){
    if (e->level() == grid_level+1) cedges.add(e);
  }
  for (List<Edge*>::Iterator e(cedges); !e.end(); ++e){
    //if (!(*e.pointer())->markedForReUse()) grid.removeEdge((*e.pointer()));
  }

}

List<Cell *> GridRefinement::closeCell(Cell *parent)
{
  List<Cell *> new_ref_cells;

  int non_marked_edge;
  int non_marked_edges[3];

  int cnt;
  switch(parent->noMarkedEdges()){ 
  case 6: 
    parent->mark(Cell::MARKED_FOR_REGULAR_REFINEMENT);
    break;
  case 5: 
    parent->mark(Cell::MARKED_FOR_REGULAR_REFINEMENT);
    for (int i=0;i<parent->noEdges();i++){    
      if (!parent->edge(i)->marked()){
	non_marked_edge = i;
	break;
      }
    }

    for (int i=0;i<parent->noNodes();i++){    
      for (int j=0;j<parent->node(i)->noCellNeighbors();j++){    
	for (int k=0;k<parent->node(i)->cell(j)->noEdges();k++){    
	  if (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edge)->id()) 
	    new_ref_cells.add(parent->node(i)->cell(j));
	}
      }
    }

    break;
  case 4: 
    parent->mark(Cell::MARKED_FOR_REGULAR_REFINEMENT);
    cnt = 0;
    for (int i=0;i<parent->noEdges();i++){    
      if (!parent->edge(i)->marked()){
	non_marked_edges[cnt++] = i;
      }
    }

    for (int i=0;i<parent->noNodes();i++){    
      for (int j=0;j<parent->node(i)->noCellNeighbors();j++){    
	for (int k=0;k<parent->node(i)->cell(j)->noEdges();k++){    
	  if ( (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edges[0])->id()) ||  
	       (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edges[1])->id()) )
	    new_ref_cells.add(parent->node(i)->cell(j));
	}
      }
    }

    break;
  case 3: 
    if (parent->markedEdgesOnSameFace()){
      parent->mark(Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_1);
    } else{
      parent->mark(Cell::MARKED_FOR_REGULAR_REFINEMENT);
      cnt = 0;
      for (int i=0;i<parent->noEdges();i++){    
	if (!parent->edge(i)->marked()){
	  non_marked_edges[cnt++] = i;
	}
      }
      
      for (int i=0;i<parent->noNodes();i++){    
	for (int j=0;j<parent->node(i)->noCellNeighbors();j++){    
	  for (int k=0;k<parent->node(i)->cell(j)->noEdges();k++){    
	    if ( (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edges[0])->id()) ||  
		 (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edges[1])->id()) ||  
		 (parent->node(i)->cell(j)->edge(k)->id() == parent->edge(non_marked_edges[2])->id()) )
	      new_ref_cells.add(parent->node(i)->cell(j));
	  }
	}
      }
      
    }
    break;
  case 2: 
    if (parent->markedEdgesOnSameFace()) parent->mark(Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_3);
    else parent->mark(Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_4);
    break;
  case 1: 
    parent->mark(Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_4); 
    break;
  default:
    dolfin_error("wrong number of marked edges");
  }

  return new_ref_cells;
}
*/



/*
void GridRefinement::localIrregularRefinement(Cell *parent)
{
  switch(parent->noMarkedEdges()){ 
  case 6: 
    regularRefinementTetrahedron(parent);
    break;
  case 5: 
    regularRefinementTetrahedron(parent);
    break;
  case 4: 
    regularRefinementTetrahedron(parent);
    break;
  case 3: 
    if (parent->markedEdgesOnSameFace()) irregTetRefByRule1(parent);
    else regularRefinementTetrahedron(parent);    
    break;
  case 2: 
    if (parent->markedEdgesOnSameFace()) irregTetRefByRule3(parent);
    else irregTetRefByRule4(parent);  
    break;
  case 1: 
    irregTetRefByRule2(parent);  
    break;
  default:
    dolfin_error("wrong number of marked edges");
  }
}


void GridRefinement::irregTetRefByRule1(Cell* parent)
{
  // 3 edges are marked on the same face: 
  // insert 3 new nodes at the midpoints on the marked edges, connect the 
  // new nodes to each other, as well as to the node that is not on the 
  // marked face. This gives 4 new tetrahedrons. 

  if (parent->noMarkedEdges() != 3) dolfin_error("wrong size of refinement edges");
  if (!parent->markedEdgesOnSameFace()) dolfin_error("marked edges not on the same face");
  
  //cout << "parent = " << parent->id() << endl;
  
  int marked_nodes[3];
  int marked_edges[3];
  marked_nodes[0] = marked_nodes[1] = marked_nodes[2] = -1;
  marked_edges[0] = marked_edges[1] = marked_edges[2] = -1;
  int cnt_1 = 0;
  int cnt_2 = 0;

  // parent->markEdge(0);
  // parent->markEdge(1);
  // parent->markEdge(3);

  bool taken;
  for (int i=0;i<parent->noEdges();i++){
    if (parent->edge(i)->marked()){
      marked_edges[cnt_1++] = i;
      for (int j=0;j<parent->noNodes();j++){
	if ( parent->edge(i)->node(0)->id() == parent->node(j)->id() ){
	  taken = false;
	  for (int k=0;k<3;k++){
	    //	    cout << "check 0: marked_nodes[k] = " << marked_nodes[k] << ", j = " << j << endl;
	    if ( marked_nodes[k] == j ) taken = true;
	  }
	  if (!taken) marked_nodes[cnt_2++] = j; 	
	}
	if ( parent->edge(i)->node(1)->id() == parent->node(j)->id() ){
	  taken = false;
	  for (int k=0;k<3;k++){
	    if ( marked_nodes[k] == j ) taken = true;
	  }
	  if (!taken) marked_nodes[cnt_2++] = j; 	
	}
      }
    }
  }

  //  cout << "what nodes = " << marked_nodes[0] << ", " << marked_nodes[1] << ", " << marked_nodes[2] << endl;
  //  cout << "1. cnt_1 = " << cnt_1 << ", cnt_2 = " << cnt_2 << endl;

  int face_node;
  for (int i=0;i<4;i++){
    taken = false;
    for (int j=0;j<3;j++){
      if (marked_nodes[j] == i) taken = true;
    }
    if (!taken){
      face_node = i;
      break;
    } 
  }
  
  // cout << "1. marked edges = " << marked_edges[0] << endl;
  // cout << "2. marked edges = " << marked_edges[1] << endl;
  // cout << "3. marked edges = " << marked_edges[2] << endl;
  // 
  // cout << "1. marked edges = " << parent->edge(marked_edges[0])->midpoint() << endl;
  // cout << "2. marked edges = " << parent->edge(marked_edges[1])->midpoint() << endl;
  // cout << "3. marked edges = " << parent->edge(marked_edges[2])->midpoint() << endl;
  //  cout << "1. level = " << parent->level()+1 << endl;

  Node *nf = grid.createNode(parent->level()+1,parent->node(face_node)->coord());
  Node *n0 = grid.createNode(parent->level()+1,parent->node(marked_nodes[0])->coord());
  Node *n1 = grid.createNode(parent->level()+1,parent->node(marked_nodes[1])->coord());
  Node *n2 = grid.createNode(parent->level()+1,parent->node(marked_nodes[2])->coord());

  parent->node(face_node)->addChild(nf);
  parent->node(marked_nodes[0])->addChild(n0);
  parent->node(marked_nodes[1])->addChild(n1);
  parent->node(marked_nodes[2])->addChild(n2);

  ShortList<Node*> edge_nodes(3);
  edge_nodes(0) = grid.createNode(parent->level()+1,parent->edge(marked_edges[0])->midpoint());
  edge_nodes(1) = grid.createNode(parent->level()+1,parent->edge(marked_edges[1])->midpoint());
  edge_nodes(2) = grid.createNode(parent->level()+1,parent->edge(marked_edges[2])->midpoint());

  ShortList<Cell*> new_cell(4);
  for (int i=0;i<3;i++){
    for (int j=0;j<3;j++){
      if ( (parent->node(marked_nodes[i])->id() != parent->edge(marked_edges[j])->node(0)->id()) &&
	   (parent->node(marked_nodes[i])->id() != parent->edge(marked_edges[j])->node(1)->id()) ){
	if (j == 0){
	  new_cell(i) = grid.createCell(parent->level()+1,n0,edge_nodes(1),edge_nodes(2),nf);
	}
	if (j == 1){
	  new_cell(i) = grid.createCell(parent->level()+1,n1,edge_nodes(0),edge_nodes(2),nf);
	}
	if (j == 2){
	  new_cell(i) = grid.createCell(parent->level()+1,n2,edge_nodes(0),edge_nodes(1),nf);
	}
      }
    }
  }

  new_cell(3) = grid.createCell(parent->level()+1,edge_nodes(0),edge_nodes(1),edge_nodes(2),nf);
  
  parent->addChild(new_cell(0));
  parent->addChild(new_cell(1));
  parent->addChild(new_cell(2));
  parent->addChild(new_cell(3));

  if (_create_edges){
    grid.createEdges(new_cell(0));
    grid.createEdges(new_cell(1));
    grid.createEdges(new_cell(2));
    grid.createEdges(new_cell(3));
  }

  parent->setStatus(Cell::REFINED_IRREGULAR_BY_1);
  if (parent->marker() == Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_1) 
    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}


void GridRefinement::irregTetRefByRule2(Cell* parent)
{
  // 1 edge is marked:
  // Insert 1 new node at the midpoint of the marked edge, then connect 
  // this new node to the 2 nodes not on the marked edge. This gives 2 new 
  // tetrahedrons. 
  
  cout << "parent = " << parent->id() << endl;

  //  parent->markEdge(2);

  if (parent->noMarkedEdges() != 1) dolfin_error("wrong size of refinement edges");

  Node *nnew;
  Node *ne0;
  Node *ne1;
  ShortList<Node*> nold(2);
  Cell* cnew1;
  Cell* cnew2;
  int cnt = 0;
  for (int i=0;i<parent->noEdges();i++){
    if (parent->edge(i)->marked()){
      nnew = grid.createNode(parent->level()+1,parent->edge(i)->midpoint());
      ne0  = grid.createNode(parent->level()+1,parent->edge(i)->node(0)->coord());
      ne1  = grid.createNode(parent->level()+1,parent->edge(i)->node(1)->coord());
      parent->edge(i)->node(0)->addChild(ne0);
      parent->edge(i)->node(1)->addChild(ne1);
      for (int j=0;j<parent->noNodes();j++){
	if ( (parent->edge(i)->node(0)->id() != j) && (parent->edge(i)->node(1)->id() != j) ){
	  nold(cnt) = grid.createNode(parent->level()+1,parent->node(j)->coord());
	  parent->node(j)->addChild(nold(cnt));
	  cnt++;
	}
      }
      cnew1 = grid.createCell(parent->level()+1,nnew,ne0,nold(0),nold(1));
      cnew2 = grid.createCell(parent->level()+1,nnew,ne1,nold(0),nold(1));
      break;
    }
  }

  parent->addChild(cnew1);
  parent->addChild(cnew2);

  if (_create_edges){
    grid.createEdges(cnew1);
    grid.createEdges(cnew2);
  }

  parent->setStatus(Cell::REFINED_IRREGULAR_BY_2);
  if (parent->marker() == Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_2) 
    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}

void GridRefinement::irregTetRefByRule3(Cell* parent)
{
  // 2 edges are marked, on the same face 
  // (here there are 2 possibilities, and the chosen 
  // alternative must match the corresponding face 
  // of the neighbor tetrahedron): 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 3 new edges by connecting the two new nodes to each 
  // other and the node opposite the face of the 2 marked edges, 
  // and insert 1 new edge by 
  // alt.1: connecting new node 1 with the endnode of marked edge 2, 
  // that is not common with marked edge 1, or 
  // alt.2: connecting new node 2 with the endnode of marked edge 1, 
  // that is not common with marked edge 2.

  cout << "parent = " << parent->id() << endl;

  // parent->markEdge(0);  
  // parent->markEdge(4);

  if (parent->noMarkedEdges() != 2) dolfin_error("wrong size of refinement edges");
  if (!parent->markedEdgesOnSameFace()) dolfin_error("marked edges not on the same face");

  if (parent->refinedByFaceRule()){
    parent->refineByFaceRule(false);
    return;
  }

  int cnt = 0;
  int marked_edge[2];
  for (int i=0;i<parent->noEdges();i++){
    if (parent->edge(i)->marked()){
      marked_edge[cnt++] = i;
    }
  }

  int face_node;
  int enoded;
  int enode1;
  int enode2;
  int cnt1,cnt2;
  for (int i=0;i<4;i++){
    cnt1 = cnt2 = 0;
    for (int j=0;j<2;j++){
      if (parent->edge(marked_edge[0])->node(j)->id() == parent->node(i)->id()) cnt1++;
      if (parent->edge(marked_edge[1])->node(j)->id() == parent->node(i)->id()) cnt2++;
    }
    cout << "cnt1 = " << cnt1 << ", cnt2 = " << cnt2 << endl;
    if ( (cnt1 == 0) && (cnt2 == 0) ) face_node = i;
    else if ( (cnt1 == 1) && (cnt2 == 1) ) enoded = i;	 
    else if ( (cnt1 == 1) && (cnt2 == 0) ) enode1 = i;	 
    else if ( (cnt1 == 0) && (cnt2 == 1) ) enode2 = i;	 
    else dolfin_error("impossible node");
  }

  Node *nf = grid.createNode(parent->level()+1,parent->node(face_node)->coord());
  Node *nd = grid.createNode(parent->level()+1,parent->node(enoded)->coord());
  Node *n1 = grid.createNode(parent->level()+1,parent->node(enode1)->coord());
  Node *n2 = grid.createNode(parent->level()+1,parent->node(enode2)->coord());

  parent->node(face_node)->addChild(nf);
  parent->node(enoded)->addChild(nd);
  parent->node(enode1)->addChild(n1);
  parent->node(enode2)->addChild(n2);

  Node *midnode1 = grid.createNode(parent->level()+1,parent->edge(marked_edge[0])->midpoint());
  Node *midnode2 = grid.createNode(parent->level()+1,parent->edge(marked_edge[1])->midpoint());
  
  // Find element with common face (enoded,enode1,enode2) 
  // (search neighbors of parent)
  int face_neighbor;
  for (int i=0;i<parent->noCellNeighbors();i++){
    for (int j=0;j<parent->neighbor(i)->noNodes();j++){
      if (parent->neighbor(i)->node(j)->id() == parent->node(enoded)->id()){
	for (int k=0;k<parent->neighbor(i)->noNodes();k++){
	  if (k != j){
	    if (parent->neighbor(i)->node(k)->id() == parent->node(enode1)->id()){
	      for (int l=0;l<parent->neighbor(i)->noNodes();l++){
		if ( (l != j) && (l != k) && (parent->neighbor(i)->node(l)->id() == parent->node(enode2)->id()) ){
		  face_neighbor = i;
		}
	      }
	    }		  
	  }
	} 
      }
    }
  }   


  Cell *c1 = grid.createCell(parent->level()+1,nd,midnode1,midnode2,nf);
  Cell *c2 = grid.createCell(parent->level()+1,n1,midnode1,midnode2,nf);
  Cell *c3 = grid.createCell(parent->level()+1,n1,n2,midnode2,nf);
  
  int neighbor_face_node;
  for (int i=0;i<4;i++){
    if ( (nd->id() != parent->neighbor(face_neighbor)->node(i)->id()) && 
	 (n1->id() != parent->neighbor(face_neighbor)->node(i)->id()) && 
	 (n2->id() != parent->neighbor(face_neighbor)->node(i)->id()) ) neighbor_face_node = i;
  }

  Node *nnf = grid.createNode(parent->level()+1,parent->neighbor(face_neighbor)->node(neighbor_face_node)->coord());
  
  Cell *nc1 = grid.createCell(parent->level()+1,nd,midnode1,midnode2,nnf);
  Cell *nc2 = grid.createCell(parent->level()+1,n1,midnode1,midnode2,nnf);
  Cell *nc3 = grid.createCell(parent->level()+1,n1,n2,midnode2,nnf);

  parent->addChild(c1);
  parent->addChild(c2);
  parent->addChild(c3);
  parent->addChild(nc1);
  parent->addChild(nc2);
  parent->addChild(nc3);

  if (_create_edges){
    grid.createEdges(c1);
    grid.createEdges(c2);
    grid.createEdges(c3);
    grid.createEdges(nc1);
    grid.createEdges(nc2);
    grid.createEdges(nc3);
  }

  parent->neighbor(face_neighbor)->refineByFaceRule(true);

  parent->setStatus(Cell::REFINED_IRREGULAR_BY_3);
  if (parent->marker() == Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_3) 
    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}

void GridRefinement::irregTetRefByRule4(Cell* parent)
{
  // 2 edges are marked, opposite to each other: 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 4 new edges by connecting the new nodes to the 
  // endpoints of the opposite edges of the respectively new nodes. 

  if (parent->noMarkedEdges() != 2) dolfin_error("wrong size of refinement edges");
  if (parent->markedEdgesOnSameFace()) dolfin_error("marked edges on the same face");

  cout << "parent = " << parent->id() << endl;

  //parent->markEdge(0);
  //parent->markEdge(2);

  int cnt = 0;
  int marked_edge[2];
  for (int i=0;i<parent->noEdges();i++){
    if (parent->edge(i)->marked()){
      marked_edge[cnt++] = i;
    }
  }

  Node *e1n1 = grid.createNode(parent->level()+1,parent->edge(marked_edge[0])->node(0)->coord());
  Node *e1n2 = grid.createNode(parent->level()+1,parent->edge(marked_edge[0])->node(1)->coord());
  Node *e2n1 = grid.createNode(parent->level()+1,parent->edge(marked_edge[1])->node(0)->coord());
  Node *e2n2 = grid.createNode(parent->level()+1,parent->edge(marked_edge[1])->node(1)->coord());

  parent->edge(marked_edge[0])->node(0)->addChild(e1n1);
  parent->edge(marked_edge[0])->node(1)->addChild(e1n2);
  parent->edge(marked_edge[1])->node(0)->addChild(e2n1);
  parent->edge(marked_edge[1])->node(1)->addChild(e2n2);

  Node *midnode1 = grid.createNode(parent->level()+1,parent->edge(marked_edge[0])->midpoint());
  Node *midnode2 = grid.createNode(parent->level()+1,parent->edge(marked_edge[1])->midpoint());

  Cell *c1 = grid.createCell(parent->level()+1,e1n1,midnode1,midnode2,e2n1);
  Cell *c2 = grid.createCell(parent->level()+1,e1n1,midnode1,midnode2,e2n2);
  Cell *c3 = grid.createCell(parent->level()+1,e1n2,midnode1,midnode2,e2n1);
  Cell *c4 = grid.createCell(parent->level()+1,e1n2,midnode1,midnode2,e2n2);

  parent->addChild(c1);
  parent->addChild(c2);
  parent->addChild(c3);
  parent->addChild(c4);

  if (_create_edges){
    grid.createEdges(c1);
    grid.createEdges(c2);
    grid.createEdges(c3);
    grid.createEdges(c4);
  }

  parent->setStatus(Cell::REFINED_IRREGULAR_BY_4);
  if (parent->marker() == Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_4) 
    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}

*/
