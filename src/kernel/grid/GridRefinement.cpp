// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Grid.h>
#include <dolfin/Edge.h>
#include <dolfin/Cell.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void GridRefinement::refine(GridHierarchy& grids)
{
  // Write a message
  dolfin_start("Refining grid:");
  cout << grids.fine().rd->noMarkedCells()
       << " cells marked for refinement." << endl;

  // Refine grid here ...
  


  dolfin_end();
}
//-----------------------------------------------------------------------------

/*
Refinement::GlobalRefinement()
{
}



void GridRefinement::globalRegularRefinement()
{
  // Regular refinement: 
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  //  cout << "no elms = " << grid.noCells() << endl;
  //  cout << "no nodes = " << grid.noNodes() << endl;
  
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c)
    cells.add(c);
  
  for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
    regularRefinement((*c.pointer()));
  }
  
  //  cout << "new no elms = " << grid.noCells() << endl;
  //  cout << "new no nodes = " << grid.noNodes() << endl;
}


void GridRefinement::globalRefinement()
{
  for (int i=grid.finestGridLevel();i>=0;i--){
    evaluateMarks(i);
    closeGrid(i);
  }

  for (int i=0;i<=grid.finestGridLevel();i++){
    if (i>0) closeGrid(i);
    unrefineGrid(i);
    refineGrid(i);
  }

  //if ...()
  


}

void GridRefinement::evaluateMarks(int grid_level)
{
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c){
    if (c->level() == grid_level) cells.add(c);
  }
      
  bool sons_marked_for_coarsening,edge_of_son_marked;
  for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
    if ((*c.pointer())->status() == Cell::REFINED_REGULAR){
      sons_marked_for_coarsening = true;
      for (int i=0;i<(*c.pointer())->noChildren();i++){
	if ((*c.pointer())->child(i)->marker() != Cell::MARKED_FOR_COARSENING){ 
	  sons_marked_for_coarsening = false;
	  break;
	}
      }
      if (sons_marked_for_coarsening) (*c.pointer())->mark(Cell::MARKED_FOR_NO_REFINEMENT);
    }

    if ((*c.pointer())->status() == Cell::REFINED_IRREGULAR){
      edge_of_son_marked = false;
      for (int i=0;i<(*c.pointer())->noChildren();i++){
	for (int j=0;j<(*c.pointer())->child(i)->noEdges();j++){
	  if ((*c.pointer())->child(i)->edge(j)->marked()){
	    edge_of_son_marked = true;
	    break;
	  }
	}
	if (edge_of_son_marked) break;
      }
      if (edge_of_son_marked) (*c.pointer())->mark(Cell::MARKED_FOR_REGULAR_REFINEMENT);
      else (*c.pointer())->mark(Cell::MARKED_FOR_NO_REFINEMENT);
    }
           
  }
}


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

void GridRefinement::refineGrid(int grid_level)
{
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c){
    if (c->level() == grid_level) cells.add(c);
  }
      
  for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
    if ((*c.pointer())->marker() == Cell::MARKED_FOR_COARSENING){
      (*c.pointer())->mark(Cell::MARKED_FOR_NO_REFINEMENT);
    }
  }

  for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
    if ((*c.pointer())->marker() != Cell::MARKED_ACCORDING_TO_REFINEMENT){
      switch((*c.pointer())->marker()){ 
      case Cell::MARKED_FOR_REGULAR_REFINEMENT: 
	regularRefinement((*c.pointer()));
	break;
      case Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_1: 
	irregularRefinementBy1((*c.pointer()));
	break;
      case Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_2: 
	irregularRefinementBy2((*c.pointer()));
	break;
      case Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_3: 
	irregularRefinementBy3((*c.pointer()));
	break;
      case Cell::MARKED_FOR_IRREGULAR_REFINEMENT_BY_4: 
	irregularRefinementBy4((*c.pointer()));
	break;
      default:
	dolfin_error("wrong refinement rule");
      }

      for (int i=0;i<(*c.pointer())->noChildren();i++){
	(*c.pointer())->child(i)->mark(Cell::MARKED_FOR_NO_REFINEMENT);
	(*c.pointer())->child(i)->setLevel((*c.pointer())->level()+1);
	if (((*c.pointer())->level()+1) > grid.finestGridLevel()) 
	  grid.setFinestGridLevel((*c.pointer())->level()+1);
      }

    }
  }
 
}



void GridRefinement::closeGrid(int grid_level)
{
  bool marked_for_ref_by_cell;
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c){
    if ( (c->level() == grid_level) && (c->status() == Cell::REFINED_REGULAR) ){ 
      marked_for_ref_by_cell = false;
      for (int i=0;i<c->noEdges();i++){
	if (marked_for_ref_by_cell) break;
	if (c->edge(i)->marked()){
	  for (int j=0;j<c->edge(j)->refinedByCells();j++){
	    if (marked_for_ref_by_cell) break;
	    if (c->edge(i)->refinedByCell(j)->id() == c->id()){
	      marked_for_ref_by_cell = true;
	      cells.add(c);
	    }
	  }
	}
      }
    }
  }

  List<Cell *> new_cells;
  bool cell_is_member;
  while (cells.size() > 0){
    
    new_cells = closeCell((*cells.pointer(0)));
    //    cells.remove(cells(0));

    for (List<Cell *>::Iterator cn(new_cells); !cn.end(); ++cn){
      cell_is_member = false;      
      for (List<Cell *>::Iterator c(cells); !c.end(); ++c){
	if ((*cn.pointer())->id() == (*c.pointer())->id()){
	  cell_is_member = true;
	  break;
	}
      }
      if (!cell_is_member) cells.add((*cn.pointer()));
    }

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



void GridRefinement::regularRefinement(Cell* parent)
{
  // Regular refinement: 
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  switch (parent->type()) {
  case Cell::TETRAHEDRON: 
    regularRefinementTetrahedron(parent);
    break;
  case Cell::TRIANGLE: 
    regularRefinementTriangle(parent);
    break;
  default: 
    dolfin_error("Cell type not implemented.");
    exit(1);
  }

  parent->setStatus(Cell::REFINED_REGULAR);
  if (parent->marker() == Cell::MARKED_FOR_REGULAR_REFINEMENT) 
    parent->mark(Cell::MARKED_ACCORDING_TO_REFINEMENT);
}


void GridRefinement::regularRefinementTetrahedron(Cell* parent)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n0 = grid.createNode(parent->level()+1,parent->node(0)->coord());
  Node *n1 = grid.createNode(parent->level()+1,parent->node(1)->coord());
  Node *n2 = grid.createNode(parent->level()+1,parent->node(2)->coord());
  Node *n3 = grid.createNode(parent->level()+1,parent->node(3)->coord());

  parent->node(0)->setChild(n0);
  parent->node(1)->setChild(n1);
  parent->node(2)->setChild(n2);
  parent->node(3)->setChild(n3);
  
  Node *n01 = grid.createNode(parent->level()+1,parent->node(0)->coord().midpoint(parent->node(1)->coord()));
  Node *n02 = grid.createNode(parent->level()+1,parent->node(0)->coord().midpoint(parent->node(2)->coord()));
  Node *n03 = grid.createNode(parent->level()+1,parent->node(0)->coord().midpoint(parent->node(3)->coord()));
  Node *n12 = grid.createNode(parent->level()+1,parent->node(1)->coord().midpoint(parent->node(2)->coord()));
  Node *n13 = grid.createNode(parent->level()+1,parent->node(1)->coord().midpoint(parent->node(3)->coord()));
  Node *n23 = grid.createNode(parent->level()+1,parent->node(2)->coord().midpoint(parent->node(3)->coord()));

  Cell *t1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n0, n01,n02,n03);
  Cell *t2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n01,n1, n12,n13);
  Cell *t3 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n02,n12,n2, n23);
  Cell *t4 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n03,n13,n23,n3 );
  Cell *t5 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n01,n02,n03,n13);
  Cell *t6 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n01,n02,n12,n13);
  Cell *t7 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n02,n03,n13,n23);
  Cell *t8 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n02,n12,n13,n23);

  parent->addChild(t1);
  parent->addChild(t2);
  parent->addChild(t3);
  parent->addChild(t4);
  parent->addChild(t5);
  parent->addChild(t6);
  parent->addChild(t7);
  parent->addChild(t8);

  if (_create_edges){
    grid.createEdges(t1);
    grid.createEdges(t2);
    grid.createEdges(t3);
    grid.createEdges(t4);
    grid.createEdges(t5);
    grid.createEdges(t6);
    grid.createEdges(t7);
    grid.createEdges(t8);
  }
}

void GridRefinement::regularRefinementTriangle(Cell* parent)
{
  // Refine 1 triangle into 4 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n0 = grid.createNode(parent->level()+1,parent->node(0)->coord());
  Node *n1 = grid.createNode(parent->level()+1,parent->node(1)->coord());
  Node *n2 = grid.createNode(parent->level()+1,parent->node(2)->coord());

  parent->node(0)->setChild(n0);
  parent->node(1)->setChild(n1);
  parent->node(2)->setChild(n2);

  Node *n01 = grid.createNode(parent->level()+1,parent->node(0)->coord().midpoint(parent->node(1)->coord()));
  Node *n02 = grid.createNode(parent->level()+1,parent->node(0)->coord().midpoint(parent->node(2)->coord()));
  Node *n12 = grid.createNode(parent->level()+1,parent->node(1)->coord().midpoint(parent->node(2)->coord()));

  Cell *t1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n0, n01,n02);
  Cell *t2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n01,n1, n12);
  Cell *t3 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n02,n12,n2 );
  Cell *t4 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n01,n12,n02);

  parent->addChild(t1);
  parent->addChild(t2);
  parent->addChild(t3);
  parent->addChild(t4);

  if (_create_edges){
    grid.createEdges(t1);
    grid.createEdges(t2);
    grid.createEdges(t3);
    grid.createEdges(t4);
  }
}

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
    if (parent->markedEdgesOnSameFace()) irregularRefinementBy1(parent);
    else regularRefinementTetrahedron(parent);    
    break;
  case 2: 
    if (parent->markedEdgesOnSameFace()) irregularRefinementBy3(parent);
    else irregularRefinementBy4(parent);  
    break;
  case 1: 
    irregularRefinementBy2(parent);  
    break;
  default:
    dolfin_error("wrong number of marked edges");
  }
}


void GridRefinement::irregularRefinementBy1(Cell* parent)
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

  parent->node(face_node)->setChild(nf);
  parent->node(marked_nodes[0])->setChild(n0);
  parent->node(marked_nodes[1])->setChild(n1);
  parent->node(marked_nodes[2])->setChild(n2);

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
	  new_cell(i) = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n0,edge_nodes(1),edge_nodes(2),nf);
	}
	if (j == 1){
	  new_cell(i) = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n1,edge_nodes(0),edge_nodes(2),nf);
	}
	if (j == 2){
	  new_cell(i) = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n2,edge_nodes(0),edge_nodes(1),nf);
	}
      }
    }
  }

  new_cell(3) = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,edge_nodes(0),edge_nodes(1),edge_nodes(2),nf);
  
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


void GridRefinement::irregularRefinementBy2(Cell* parent)
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
      parent->edge(i)->node(0)->setChild(ne0);
      parent->edge(i)->node(1)->setChild(ne1);
      for (int j=0;j<parent->noNodes();j++){
	if ( (parent->edge(i)->node(0)->id() != j) && (parent->edge(i)->node(1)->id() != j) ){
	  nold(cnt) = grid.createNode(parent->level()+1,parent->node(j)->coord());
	  parent->node(j)->setChild(nold(cnt));
	  cnt++;
	}
      }
      cnew1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,nnew,ne0,nold(0),nold(1));
      cnew2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,nnew,ne1,nold(0),nold(1));
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

void GridRefinement::irregularRefinementBy3(Cell* parent)
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

  parent->node(face_node)->setChild(nf);
  parent->node(enoded)->setChild(nd);
  parent->node(enode1)->setChild(n1);
  parent->node(enode2)->setChild(n2);

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


  Cell *c1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,nd,midnode1,midnode2,nf);
  Cell *c2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n1,midnode1,midnode2,nf);
  Cell *c3 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n1,n2,midnode2,nf);
  
  int neighbor_face_node;
  for (int i=0;i<4;i++){
    if ( (nd->id() != parent->neighbor(face_neighbor)->node(i)->id()) && 
	 (n1->id() != parent->neighbor(face_neighbor)->node(i)->id()) && 
	 (n2->id() != parent->neighbor(face_neighbor)->node(i)->id()) ) neighbor_face_node = i;
  }

  Node *nnf = grid.createNode(parent->level()+1,parent->neighbor(face_neighbor)->node(neighbor_face_node)->coord());
  
  Cell *nc1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,nd,midnode1,midnode2,nnf);
  Cell *nc2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n1,midnode1,midnode2,nnf);
  Cell *nc3 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,n1,n2,midnode2,nnf);

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

void GridRefinement::irregularRefinementBy4(Cell* parent)
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

  parent->edge(marked_edge[0])->node(0)->setChild(e1n1);
  parent->edge(marked_edge[0])->node(1)->setChild(e1n2);
  parent->edge(marked_edge[1])->node(0)->setChild(e2n1);
  parent->edge(marked_edge[1])->node(1)->setChild(e2n2);

  Node *midnode1 = grid.createNode(parent->level()+1,parent->edge(marked_edge[0])->midpoint());
  Node *midnode2 = grid.createNode(parent->level()+1,parent->edge(marked_edge[1])->midpoint());

  Cell *c1 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,e1n1,midnode1,midnode2,e2n1);
  Cell *c2 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,e1n1,midnode1,midnode2,e2n2);
  Cell *c3 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,e1n2,midnode1,midnode2,e2n1);
  Cell *c4 = grid.createCell(parent->level()+1,Cell::TETRAHEDRON,e1n2,midnode1,midnode2,e2n2);

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
