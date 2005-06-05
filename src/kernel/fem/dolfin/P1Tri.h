// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __P1_TRI_H
#define __P1_TRI_H

#include <dolfin/FiniteElement.h>

namespace dolfin
{

  /// This class represents the standard scalar- or vector-valued
  /// linear finite element on a triangle. Note that finite elements
  /// are normally generated automatically by FFC, but this class
  /// might be useful for simple computations with standard linear
  /// elements.

  class P1Tri : public FiniteElement
  {
  public:
        
    P1Tri(uint vectordim = 0) : FiniteElement(), vectordim(vectordim) {}
    
    ~P1Tri() {}

    inline unsigned int spacedim() const
    {
      return 3;
    }

    inline unsigned int shapedim() const
    {
      return 2;
    }

    inline unsigned int tensordim(unsigned int i) const
    {
      dolfin_error("Element is scalar.");
      return 0;
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
    }

    void pointmap(Point points[], unsigned int components[], const AffineMap& map) const
    {
      points[0] = map(0.000000000000000e+00, 0.000000000000000e+00);
      points[1] = map(1.000000000000000e+00, 0.000000000000000e+00);
      points[2] = map(0.000000000000000e+00, 1.000000000000000e+00);
      components[0] = 0;
      components[1] = 0;
      components[2] = 0;
    }

  private:

    uint vectordim;
    
  };
  
}

#endif
