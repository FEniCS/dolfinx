// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005-12-01

#ifndef __TETRAHEDRON_H
#define __TETRAHEDRON_H

#include <dolfin/PArray.h>
#include <dolfin/GenericCell.h>

namespace dolfin
{

  class Vertex;
  class Cell;
  
  class Tetrahedron : public GenericCell
  {
  public:

    Tetrahedron(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3);
	 
    int noVertices() const;
    int noEdges() const;
    int noFaces() const;

    int noBoundaries() const;
    
    Cell::Type type() const;
    Cell::Orientation orientation() const;

    real volume() const;
    real diameter() const;

    uint edgeAlignment(uint i) const;
    uint faceAlignment(uint i) const;
    
  private:
    
    void createEdges();
    void createFaces();
    void sort();
    Edge* findEdge(uint n0, uint n1) const;
    Face* findFace(uint n) const;

  };

}

#endif
