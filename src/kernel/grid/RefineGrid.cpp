// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
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
  //
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  Grid grid_tmp;
  grid_tmp = grid;

  int cnt = 0;
  for (CellIterator c(grid_tmp); !c.end(); ++c){
    RegularRefinement(*c);
    cout << "cnt = " << cnt++ << endl;
  }

}


void RefineGrid::RegularRefinement(Cell &parent)
{
  // Regular refinement: 
  //
  // (1) Triangles: 1 -> 4 
  // (2) Tetrahedrons: 1 -> 8 

  switch (parent.type()) {
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


void RefineGrid::RegularRefinementTetrahedron(Cell &parent)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n01 = grid.createNode(parent.node(0)->coord().midpoint(parent.node(1)->coord()));
  Node *n02 = grid.createNode(parent.node(0)->coord().midpoint(parent.node(2)->coord()));
  Node *n03 = grid.createNode(parent.node(0)->coord().midpoint(parent.node(3)->coord()));
  Node *n12 = grid.createNode(parent.node(1)->coord().midpoint(parent.node(2)->coord()));
  Node *n13 = grid.createNode(parent.node(1)->coord().midpoint(parent.node(3)->coord()));
  Node *n23 = grid.createNode(parent.node(2)->coord().midpoint(parent.node(3)->coord()));

  Cell *t1 = grid.createCell(Cell::TETRAHEDRON,parent.node(0),n01,n02,n03);
  Cell *t2 = grid.createCell(Cell::TETRAHEDRON,n01,parent.node(1),n12,n13);
  Cell *t3 = grid.createCell(Cell::TETRAHEDRON,n02,n12,parent.node(2),n23);
  Cell *t4 = grid.createCell(Cell::TETRAHEDRON,n03,n13,n23,parent.node(3));
  Cell *t5 = grid.createCell(Cell::TETRAHEDRON,n01,n02,n03,n13);
  Cell *t6 = grid.createCell(Cell::TETRAHEDRON,n01,n02,n12,n13);
  Cell *t7 = grid.createCell(Cell::TETRAHEDRON,n02,n03,n13,n23);
  Cell *t8 = grid.createCell(Cell::TETRAHEDRON,n02,n12,n13,n23);
}

void RefineGrid::RegularRefinementTriangle(Cell &parent)
{
  // Refine 1 triangle into 4 new ones, introducing new nodes 
  // at the midpoints of the edges. 
  Node *n01 = grid.createNode(parent.node(0)->coord().midpoint(parent.node(1)->coord()));
  Node *n02 = grid.createNode(parent.node(0)->coord().midpoint(parent.node(2)->coord()));
  Node *n12 = grid.createNode(parent.node(1)->coord().midpoint(parent.node(2)->coord()));

  Cell *t1 = grid.createCell(Cell::TETRAHEDRON,parent.node(0),n01,n02);
  Cell *t2 = grid.createCell(Cell::TETRAHEDRON,n01,parent.node(1),n12);
  Cell *t3 = grid.createCell(Cell::TETRAHEDRON,n02,n12,parent.node(2));
  Cell *t4 = grid.createCell(Cell::TETRAHEDRON,n01,n12,n02);
}

/*
Refinement::LocalIrregularRefinement(Tetrahedron *parent)
{
  bool reg_ref = false;
  bool same_face = false;

  int ref_edges[2];
  int cnt = 0;
  
  switch(parent->GetNoMarkedEdges()){ 
  case 4: 
    LocalRegularRefinement(parent);
    break;
  case 3: 
    reg_ref = true;
    for (int i=0; i<4; i++){
      if (parent->GetFace(i)->GetNoMarkedEdges() == 3){ 
	IrrRef1(parent,i);
	reg_ref = false;
      }
    }
    if (reg_ref) LocalRegularRefinement(parent);    
    break;
  case 2: 
    for (int i=0; i<4; i++){
      if (parent->GetFace(i)->GetNoMarkedEdges() == 2){
	cnt = 0;
	for (int i=0; i<6; i++){
	  if (parent->GetEdge(i)->MarkedForRefinement()) ref_edges[cnt] = i;
	  cnt++;
	}
	IrrRef3(parent,ref_edges[0],ref_edges[1]);
	same_face = true;
      }
    }   
    if (!same_face){
      cnt = 0;
      for (int i=0; i<6; i++){
	if (parent->GetEdge(i)->MarkedForRefinement()) ref_edges[cnt] = i;
	cnt++;
      }
      IrrRef4(parent,ref_edges[0],ref_edges[1]);
    }    
    break;
  case 1: 
    for (int i=0; i<6; i++){
      if (parent->GetEdge(i)->MarkedForRefinement()){ 
	IrrRef2(parent,i);
      }
    }    
    break;
  case NONE:
    display->InternalError("Refinement::LocalIrregularRefinement()","Wrong no of marked edges");
    break;
  default:
    display->InternalError("Refinement::LocalIrregularRefinement()","Wrong no of marked edges");
  }

}


Refinement::IrrRef1(Tetrahedron *parent, int marked_face)
{      
  // 3 edges are marked on the same face: 
  // insert 3 new nodes at the midpoints on the marked edges, connect the 
  // new nodes to each other, as well as to the node that is not on the 
  // marked face. This gives 4 new tetrahedrons. 

  Node *n01 = grid.createNode();
  Node *n02 = grid.createNode();
  Node *n12 = grid.createNode();
  
  n01->Set(parent->GetFace(marked_face)->GetNode(0)->p.Midpoint(parent->GetFace(marked_face)->GetNode(1)->p));
  n02->Set(parent->GetFace(marked_face)->GetNode(0)->p.Midpoint(parent->GetFace(marked_face)->GetNode(2)->p));
  n12->Set(parent->GetFace(marked_face)->GetNode(1)->p.Midpoint(parent->GetFace(marked_face)->GetNode(2)->p));
	  
  Tetrahedron *t1 = grid.createCell();
  Tetrahedron *t2 = grid.createCell();
  Tetrahedron *t3 = grid.createCell();
  Tetrahedron *t4 = grid.createCell();

  t1->Set(parent->GetFace(marked_face)->GetNode(0),n01,n02,parent->GetNode(marked_face));
  t2->Set(parent->GetFace(marked_face)->GetNode(1),n01,n12,parent->GetNode(marked_face));
  t3->Set(parent->GetFace(marked_face)->GetNode(2),n02,n12,parent->GetNode(marked_face));
  t4->Set(n01,n02,n12,parent->GetNode(marked_face));

}


Refinement::IrrRef2(Tetrahedron *parent, int marked_edge)
{      
  // 1 edge is marked:
  // Insert 1 new node at the midpoint of the marked edge, then connect 
  // this new node to the 2 nodes not on the marked edge. This gives 2 new 
  // tetrahedrons. 
  
  int non_marked_nodes[2];
  int cnt = 0;

  Node *n = grid.createNode();
  
  n->Set(parent->GetEdge(marked_edge)->GetMidpoint());

  Tetrahedron *t1 = grid.createCell();
  Tetrahedron *t2 = grid.createCell();

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


Refinement::IrrRef3(Tetrahedron *parent, int marked_edge1, int marked_edge2)
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

  Tetrahedron *t1 = grid.createCell();
  Tetrahedron *t2 = grid.createCell();
  Tetrahedron *t3 = grid.createCell();

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


Refinement::IrrRef4(Tetrahedron *parent, int marked_edge1, int marked_edge2)
{      
  // 2 edges are marked, opposite to each other: 
  // insert 2 new nodes at the midpoints of the marked edges, 
  // insert 4 new edges by connecting the new nodes to the 
  // endpoints of the opposite edges of the respectively new nodes. 

  Node *n1 = grid.createNode();
  Node *n2 = grid.createNode();
  
  n1->Set(parent->GetEdge(marked_edge1)->GetMidpoint());
  n2->Set(parent->GetEdge(marked_edge2)->GetMidpoint());

  Tetrahedron *t1 = grid.createCell();
  Tetrahedron *t2 = grid.createCell();
  Tetrahedron *t3 = grid.createCell();
  Tetrahedron *t4 = grid.createCell();

  t1->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t2->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);
  t3->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t4->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);

}
*/
