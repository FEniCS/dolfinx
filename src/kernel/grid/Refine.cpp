// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Refinement.hh"



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



Refinement::LocalRegularRefinement(Tetrahedron *parent)
{
  Point p01 = parent->GetNode(0)->p.Midpoint(parent->GetNode(1)->p);
  Point p02 = parent->GetNode(0)->p.Midpoint(parent->GetNode(2)->p);
  Point p03 = parent->GetNode(0)->p.Midpoint(parent->GetNode(3)->p);
  Point p12 = parent->GetNode(1)->p.Midpoint(parent->GetNode(2)->p);
  Point p13 = parent->GetNode(1)->p.Midpoint(parent->GetNode(3)->p);
  Point p23 = parent->GetNode(2)->p.Midpoint(parent->GetNode(3)->p);

  Node *n01 = grid->createNode();
  Node *n02 = grid->createNode();
  Node *n03 = grid->createNode();
  Node *n11 = grid->createNode();
  Node *n12 = grid->createNode();
  Node *n23 = grid->createNode();

  n01->Set(p01);
  n02->Set(p02);
  n03->Set(p03);
  n11->Set(p11);
  n12->Set(p12);
  n23->Set(p23);

  Tetrahedron *t1 = grid->createCell();
  Tetrahedron *t2 = grid->createCell();
  Tetrahedron *t3 = grid->createCell();
  Tetrahedron *t4 = grid->createCell();
  Tetrahedron *t5 = grid->createCell();
  Tetrahedron *t6 = grid->createCell();
  Tetrahedron *t7 = grid->createCell();
  Tetrahedron *t8 = grid->createCell();

  t1->Set(parent->GetNode(0),n01,n02,n03);
  t2->Set(n01,parent->GetNode(1),n12,n13);
  t3->Set(n02,n12,parent->GetNode(2),n23);
  t4->Set(n03,n13,n23,parent->GetNode(3));
  t5->Set(n01,n02,n03,n13);
  t6->Set(n01,n02,n12,n13);
  t7->Set(n02,n03,n13,n23);
  t8->Set(n02,n12,n13,n23);

}


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

  Node *n01 = grid->createNode();
  Node *n02 = grid->createNode();
  Node *n12 = grid->createNode();
  
  n01->Set(parent->GetFace(marked_face)->GetNode(0)->p.Midpoint(parent->GetFace(marked_face)->GetNode(1)->p));
  n02->Set(parent->GetFace(marked_face)->GetNode(0)->p.Midpoint(parent->GetFace(marked_face)->GetNode(2)->p));
  n12->Set(parent->GetFace(marked_face)->GetNode(1)->p.Midpoint(parent->GetFace(marked_face)->GetNode(2)->p));
	  
  Tetrahedron *t1 = grid->createCell();
  Tetrahedron *t2 = grid->createCell();
  Tetrahedron *t3 = grid->createCell();
  Tetrahedron *t4 = grid->createCell();

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

  Node *n = grid->createNode();
  
  n->Set(parent->GetEdge(marked_edge)->GetMidpoint());

  Tetrahedron *t1 = grid->createCell();
  Tetrahedron *t2 = grid->createCell();

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

  Node *n1 = grid->createNode();
  Node *n2 = grid->createNode();
  
  n1->Set(parent->GetEdge(marked_edge1)->GetMidpoint());
  n2->Set(parent->GetEdge(marked_edge2)->GetMidpoint());

  Tetrahedron *t1 = grid->createCell();
  Tetrahedron *t2 = grid->createCell();
  Tetrahedron *t3 = grid->createCell();

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

  Node *n1 = grid->createNode();
  Node *n2 = grid->createNode();
  
  n1->Set(parent->GetEdge(marked_edge1)->GetMidpoint());
  n2->Set(parent->GetEdge(marked_edge2)->GetMidpoint());

  Tetrahedron *t1 = grid->createCell();
  Tetrahedron *t2 = grid->createCell();
  Tetrahedron *t3 = grid->createCell();
  Tetrahedron *t4 = grid->createCell();

  t1->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t2->Set(parent->GetEdge(marked_edge1)->GetEndnode(0),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);
  t3->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(0),n2);
  t4->Set(parent->GetEdge(marked_edge1)->GetEndnode(1),n1,parent->GetEdge(marked_edge2)->GetEndnode(1),n2);

}

