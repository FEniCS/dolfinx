// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TETRAHEDRON_H
#define __TETRAHEDRON_H

#include <dolfin/Array.h>
#include <dolfin/GenericCell.h>

namespace dolfin {

  class Node;
  class Cell;
  
  class Tetrahedron : public GenericCell {
  public:

    Tetrahedron(Node* n0, Node* n1, Node* n2, Node* n3);
	 
    int noNodes() const;
    int noEdges() const;
    int noFaces() const;

    int noBoundaries() const;
    
    Cell::Type type() const;

    real volume() const;
    real diameter() const;
    
  private:
    
    void createEdges();
    void createFaces();

  };

}

#endif
