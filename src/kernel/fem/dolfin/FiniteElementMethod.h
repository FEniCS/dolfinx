// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FINITE_ELEMENT_METHOD
#define __FINITE_ELEMENT_METHOD

#include <dolfin/Mesh.h>
#include <dolfin/FiniteElement.h>

namespace dolfin
{
  
  class Map;
  class Quadrature;

  /// A FiniteElementMethod contains the specification of a Galerkin
  /// finite element method, including
  ///
  /// (1) A finite element
  /// (2) A map from the reference cell
  /// (3) A quadrature rule on the reference cell

  class FiniteElementMethod
  {
  public:
    
    /// Create default finite element method for given mesh and system size
    FiniteElementMethod(Mesh::Type type, unsigned int noeq);
    
    /// Create user-specified finite element method
    FiniteElementMethod(FiniteElement::Vector& element, 
			Map& map, 
			Quadrature& quadrature);
    
    /// Destructor
    ~FiniteElementMethod();

    /// The assembler FEM is a friend
    friend class FEM;

  private:

    // The finite element
    FiniteElement::Vector* element;

    // The map from the reference cell
    Map* map;
    
    // The quadrature rule on the reference cell
    Quadrature* quadrature;
    
    // True if user specifies method
    bool user;

  };

}

#endif
