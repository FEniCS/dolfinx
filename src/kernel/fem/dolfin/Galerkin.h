// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GALERKIN_H
#define __GALERKIN_H

namespace dolfin {

  class Vector;
  class Matrix;
  class Grid;
  class PDE;
  class FiniteElement;
  class Mapping;
  class Quadrature;
  
  /// Assembling from variational formulation. It is assumed that
  /// each degree of freedom is associated with a node. The id of
  /// the node is used as an index for the degrees of freedom.

  class Galerkin {
  public:
    
    // Default constructor (default method will be used)
    Galerkin();
    
    // Constructor allowing specification of method
    Galerkin(FiniteElement& element, Mapping& mapping, Quadrature& quadrature);
    
    // Destructor
    ~Galerkin();
    
    // Assemble and set boundary conditions
    void assemble(PDE& pde, Grid& grid, Matrix& A, Vector& b);
    
    // Assemble A and b individually and set boundary conditions
    void assemble(PDE& pde, Grid& grid, Matrix& A);
    void assemble(PDE& pde, Grid& grid, Vector& b);

    // Assemble A and b individually, without setting boundary conditions
    void assembleLHS(PDE& pde, Grid& grid, Matrix& A);
    void assembleRHS(PDE& pde, Grid& grid, Vector& b);
    
    // Set boundary conditions
    void setBC(Grid& grid, Matrix& A);
    void setBC(Grid& grid, Vector& b);
    
  private:
    
    void init(Grid& grid);
    
    void alloc(Matrix& A, Grid& grid);
    void alloc(Vector& b, Grid& grid);
    
    // Method data
    FiniteElement* element;    // The finite element
    Mapping*       mapping;    // Mapping from reference cell
    Quadrature*    quadrature; // Quadrature on reference cell
    
    // True if user specifies method
    bool user;
    
  };

}

#endif
