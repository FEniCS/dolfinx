// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
#include <dolfin/Edge.h>
#include <dolfin/Cell.h>
#include <dolfin/RefineGrid.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void RefineGrid::refine()
{
  cout << "Refining grid: " << grid << endl;

  GlobalRegularRefinement();
}
//-----------------------------------------------------------------------------

/*
Refinement::GlobalRefinement()
{
}

Refinement::EvaluateMarks()
{
}

Refinement::CloseGrid()
{
}

Refinement::CloseElement()
{
}

Refinement::UnrefineGrid()
{
}

Refinement::RefineGrid()
{
}
*/


void RefineGrid::GlobalRegularRefinement()
{
  // Regular refinement: 
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  //  cout << "no elms = " << grid.noCells() << endl;
  //  cout << "no nodes = " << grid.noNodes() << endl;
  
  List<Cell *> cells;
  for (CellIterator c(grid); !c.end(); ++c)
    cells.add(c);
  
  for (List<Cell *>::Iterator c(&cells); !c.end(); ++c)
    RegularRefinement((*c.pointer()));
  
  //  cout << "new no elms = " << grid.noCells() << endl;
  //  cout << "new no nodes = " << grid.noNodes() << endl;
}


void RefineGrid::RegularRefinement(Cell* parent)
{
  // Regular refinement: 
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  switch (parent->type()) {
  case Cell::TETRAHEDRON: 
    RegularRefinementTetrahedron(parent);
    break;
  case Cell::TRIANGLE: 
    RegularRefinementTriangle(parent);
    break;
  default: 
    dolfin_error("Cell type not implemented.");
    exit(1);
  }
}


void RefineGrid::RegularRefinementTetrahedron(Cell* parent)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n01 = grid.createNode(parent->node(0)->coord().midpoint(parent->node(1)->coord()));
  Node *n02 = grid.createNode(parent->node(0)->coord().midpoint(parent->node(2)->coord()));
  Node *n03 = grid.createNode(parent->node(0)->coord().midpoint(parent->node(3)->coord()));
  Node *n12 = grid.createNode(parent->node(1)->coord().midpoint(parent->node(2)->coord()));
  Node *n13 = grid.createNode(parent->node(1)->coord().midpoint(parent->node(3)->coord()));
  Node *n23 = grid.createNode(parent->node(2)->coord().midpoint(parent->node(3)->coord()));

  Cell *t1 = grid.createCell(parent,Cell::TETRAHEDRON,parent->node(0),n01,n02,n03);
  Cell *t2 = grid.createCell(parent,Cell::TETRAHEDRON,n01,parent->node(1),n12,n13);
  Cell *t3 = grid.createCell(parent,Cell::TETRAHEDRON,n02,n12,parent->node(2),n23);
  Cell *t4 = grid.createCell(parent,Cell::TETRAHEDRON,n03,n13,n23,parent->node(3));
  Cell *t5 = grid.createCell(parent,Cell::TETRAHEDRON,n01,n02,n03,n13);
  Cell *t6 = grid.createCell(parent,Cell::TETRAHEDRON,n01,n02,n12,n13);
  Cell *t7 = grid.createCell(parent,Cell::TETRAHEDRON,n02,n03,n13,n23);
  Cell *t8 = grid.createCell(parent,Cell::TETRAHEDRON,n02,n12,n13,n23);
}

void RefineGrid::RegularRefinementTriangle(Cell* parent)
{
  // Refine 1 triangle into 4 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n01 = grid.createNode(parent->node(0)->coord().midpoint(parent->node(1)->coord()));
  Node *n02 = grid.createNode(parent->node(0)->coord().midpoint(parent->node(2)->coord()));
  Node *n12 = grid.createNode(parent->node(1)->coord().midpoint(parent->node(2)->coord()));

  Cell *t1 = grid.createCell(parent,Cell::TETRAHEDRON,parent->node(0),n01,n02);
  Cell *t2 = grid.createCell(parent,Cell::TETRAHEDRON,n01,parent->node(1),n12);
  Cell *t3 = grid.createCell(parent,Cell::TETRAHEDRON,n02,n12,parent->node(2));
  Cell *t4 = grid.createCell(parent,Cell::TETRAHEDRON,n01,n12,n02);
}

void RefineGrid::LocalIrregularRefinement(Cell *parent)
{
  switch(parent->noMarkedEdges()){ 
  case 6: 
    RegularRefinementTetrahedron(parent);
    break;
  case 5: 
    RegularRefinementTetrahedron(parent);
    break;
  case 4: 
    RegularRefinementTetrahedron(parent);
    break;
  case 3: 
    if (parent->markedEdgesOnSameFace()) IrrRef1(parent);
    else RegularRefinementTetrahedron(parent);    
    break;
  case 2: 
    if (parent->markedEdgesOnSameFace()) IrrRef3(parent);
    else IrrRef4(parent);  
    break;
  case 1: 
    IrrRef2(parent);  
    break;
  default:
    dolfin_error("wrong number of marked edges");
  }
}

void RefineGrid::IrrRef1(Cell* parent)
{
}

void RefineGrid::IrrRef2(Cell* parent)
{
}

void RefineGrid::IrrRef3(Cell* parent)
{
}

void RefineGrid::IrrRef4(Cell* parent)
{
}

/*

void RefineGrid::IrrRef1(Cell* parent, ShortList<Edge*> marked_edges)
{      
  // 3 edges are marked on the same face: 
  // insert 3 new nodes at the midpoints on the marked edges, connect the 
  // new nodes to each other, as well as to the node that is not on the 
  // marked face. This gives 4 new tetrahedrons. 

  if (marked_edges.size() != 3 ) dolfin_error("wrong size of refinement edges");

  Node* n01 = grid.createNode(marked_edges(0)->computeMidpoint());
  Node* n02 = grid.createNode(marked_edges(1)->computeMidpoint());
  Node* n12 = grid.createNode(marked_edges(2)->computeMidpoint());

  Node* nface; 
  for (int i=0; i<parent->noNodes(); i++){
    if ( (marked_edges(0)->node(0)->id() != parent->node(i)->id()) && 
	 (marked_edges(0)->node(1)->id() != parent->node(i)->id()) &&
	 (marked_edges(1)->node(0)->id() != parent->node(i)->id()) &&
	 (marked_edges(1)->node(1)->id() != parent->node(i)->id()) ){
      nface = parent->node(i);
      break;
    }
  }

  Cell *t1 = grid.createCell(level,Cell::TETRAHEDRON,parent->node(0),n01,n02,n03);


  Cell *t1 = grid.createCell(parent,Cell::TETRAHEDRON,parent->nodeGetFace(marked_face)->GetNode(0),n01,n02,nface);
  Cell *t2 = grid.createCell();
  Cell *t3 = grid.createCell();
  Cell *t4 = grid.createCell();

  t1->Set(parent,Cell::TETRAHEDRON,parent->nodeGetFace(marked_face)->GetNode(0),n01,n02,nface);
  t2->Set(parent->GetFace(marked_face)->GetNode(1),n01,n12,nface);
  t3->Set(parent->GetFace(marked_face)->GetNode(2),n02,n12,nface);
  t4->Set(n01,n02,n12,parent->GetNode(marked_face));

}
*/


/*
Refinement::IrrRef2(Cell *parent, int marked_edge)
{      
  // 1 edge is marked:
  // Insert 1 new node at the midpoint of the marked edge, then connect 
  // this new node to the 2 nodes not on the marked edge. This gives 2 new 
  // tetrahedrons. 
  
  int non_marked_nodes[2];
  int cnt = 0;

  Node *n = grid.createNode();
  
  n->Set(parent->GetEdge(marked_edge)->GetMidpoint());

  Cell *t1 = grid.createCell();
  Cell *t2 = grid.createCell();

  cnt = 0;
  for (int i=0; i<4; i++){
    if ( (parent->GetNode(i) != parent->GetEdge(marked_edge)->GetEndnode(0)) && 
	 (parent->GetNode(i) != parent->GetEdge(marked_edge)->GetEndnode(1)) ){
      non_marked_nodes[cnt] = i;
      cnt++;
    }
  }
  
  t1->Set(parent->GetEdge(marked_edge)->GetEndnode(0),n,parent->GetNode(non_marked_nodes[0]),parent->GetNode(non_marked_nodes[1]));
  t2->Set(parent->GetEdge(marked_edge)->GetEndnode(1),n,parent->GetNode(non_marked_nodes[0]),parent->GetNode(non_marked_nodes[1]));

}


Refinement::IrrRef3(Cell *parent, int marked_edge1, int marked_edge2)
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

  int non_marked_node;
  int double_marked_node;
  
  int once_marked_node[2];
  int node_marks[4];

  Node *n1 = grid.createNode();
  Node *n2 = grid.createNode();
  
  n1->Set(parent->GetEdge(marked_edge1)->GetMidpoint());
  n2->Set(parent->GetEdge(marked_edge2)->GetMidpoint());

  Cell *t1 = grid.createCell();
  Cell *t2 = grid.createCell();
  Cell *t3 = grid.createCell();

  for (int i=0; i<4; i++){
    node_marks[i] = 0;
    if ( (parent->GetNode(i) == parent->GetEdge(marked_edge1)->GetEndnode(0)) || 
	 (parent->GetNode(i) == parent->GetEdge(marked_edge1)->GetEndnode(1)) ||
	 (parent->GetNode(i) == parent->GetEdge(marked_edge2)->GetEndnode(0)) ||
	 (parent->GetNode(i) == parent->GetEdge(marked_edge2)->GetEndnode(1)) ){
      node_marks[i]++;
    }
  }  

  int cnt = 0;
  for (int i=0; i<4; i++){
    if (node_marks[i]==0) non_marked_node = i;
    else if (node_marks[i]==2) double_marked_node = i;
    else{ 
      once_marked_node[cnt] = i;
      cont++;
    }
  }
  
  t1->Set(parent->GetNode(double_marked_node),n1,n2,parent->GetNode(non_marked_node));

  t2->Set(parent->GetNode(once_marked_node[0]),n1,n2,parent->GetNode(non_marked_node));
  t3->Set(parent->GetNode(once_marked_node[1]),n1,n2,parent->GetNode(non_marked_node));

  t2->Set(parent->GetNode(double_marked_node),n1,n2,parent->GetNode(non_marked_node));
  t3->Set(parent->GetNode(double_marked_node),n1,n2,parent->GetNode(non_marked_node));
}


Refinement::IrrRef4(Cell *parent, int marked_edge1, int marked_edge2)
{      
  // 2 edges are marked, opposite to each other: 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 4 new edges by connecting the new nodes to the 
  // endpoints of the opposite edges of the respectively new nodes. 

  Node *n1 = grid.createNode();
  Node *n2 = grid.createNode();
  
  n1->Set(parent->GetEdge(marked_edge1)->GetMidpoint());
  n2->Set(parent->GetEdge(marked_edge2)->GetMidpoint());

  Cell *t1 = grid.createCell();
  Cell *t2 = grid.createCell();
  Cell *t3 = grid.createCell();
  Cell *t4 = grid.createCell();

  t1->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t2->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);
  t3->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t4->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);

}
*/
