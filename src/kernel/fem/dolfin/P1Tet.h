// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005-11-29

#ifndef __P1_TET_H
#define __P1_TET_H

#include <dolfin/FiniteElement.h>

namespace dolfin
{

  /// This class represents the standard scalar- or vector-valued
  /// linear finite element on a tetrahedron. Note that finite
  /// elements are normally generated automatically by FFC, but this
  /// class might be useful for simple computations with standard
  /// linear elements.

  class P1Tet : public dolfin::FiniteElement
  {
  public:
  
    P1Tet() : dolfin::FiniteElement(), tensordims(0), subelements(0)
    {
      // Element is scalar, don't need to initialize tensordims
  
      // Element is simple, don't need to initialize subelements
    }
  
    ~P1Tet()
    {
      if ( tensordims ) delete [] tensordims;
      if ( subelements )
      {
        for (unsigned int i = 0; i < elementdim(); i++)
          delete subelements[i];
        delete [] subelements;
      }
    }
  
    inline unsigned int spacedim() const
    {
      return 4;
    }
  
    inline unsigned int shapedim() const
    {
      return 3;
    }
  
    inline unsigned int tensordim(unsigned int i) const
    {
      dolfin_error("Element is scalar.");
      return 0;
    }
  
    inline unsigned int elementdim() const
    {
      return 1;
    }
  
    inline unsigned int rank() const
    {
      return 0;
    }
  
    void dofmap(int dofs[], const Cell& cell, const Mesh& mesh) const
    {
      dofs[0] = cell.nodeID(0);
      dofs[1] = cell.nodeID(1);
      dofs[2] = cell.nodeID(2);
      dofs[3] = cell.nodeID(3);
    }
  
    void pointmap(Point points[], unsigned int components[], const AffineMap& map) const
    {
      points[0] = map(0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00);
      points[1] = map(1.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00);
      points[2] = map(0.000000000000000e+00, 1.000000000000000e+00, 0.000000000000000e+00);
      points[3] = map(0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00);
      components[0] = 0;
      components[1] = 0;
      components[2] = 0;
      components[3] = 0;
    }
  
    void vertexeval(real values[], unsigned int vertex, const Vector& x, const Mesh& mesh) const
    {
      // FIXME: Temporary fix for Lagrange elements
      values[0] = x(vertex);
    }
  
    const FiniteElement& operator[] (unsigned int i) const
    {
      return *this;
    }
    
  private:
  
    unsigned int* tensordims;
    FiniteElement** subelements;
  
  };

}

#endif
