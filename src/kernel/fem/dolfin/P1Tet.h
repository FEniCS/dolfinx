// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __P1_TET_H
#define __P1_TET_H

#include <dolfin/FiniteElement.h>

namespace dolfin
{

  /// This class represents the standard scalar- or vector-valued linear
  /// finite element on a tetrahedron. Note that finite elements are
  /// normally generated automatically by FFC, but this class might be
  /// useful for simple computations with standard linear elements.

  class P1Tet : public FiniteElement
  {
  public:
        
    P1Tet(uint vectordim = 0) : FiniteElement(), vectordim(vectordim) {}
    
    ~P1Tet() {}
    
    inline uint spacedim() const
    {
      return 4 * vectordim;
    }
    
    inline uint shapedim() const
    {
      return 3;
    }
    
    inline uint tensordim(uint i) const
    {
      if ( vectordim == 0 )
	dolfin_error("Element is scalar.");

      return vectordim;
    }
    
    inline uint rank() const
    {
      if ( vectordim == 0 )
	return 0;

      return 1;
    }
    
    inline uint dof(uint i, const Cell& cell, const Mesh& mesh) const
    {
      return (i/4) * mesh.noNodes() + cell.nodeID(i % 4);
    }
    
    inline const Point coord(uint i, const Cell& cell, const Mesh& mesh) const
    {
      return cell.node(i % 4).coord();
    }

  private:

    uint vectordim;
    
  };
  
}

#endif
