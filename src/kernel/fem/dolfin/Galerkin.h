// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GALERKIN_H
#define __GALERKIN_H

namespace dolfin {

  class Vector;
  class Matrix;
  class Mesh;
  class PDE;
  class FiniteElement;
  class Map;
  class Quadrature;
  
  /// Assembling from variational formulation. It is assumed that
  /// each degree of freedom is associated with a node. The id of
  /// the node is used as an index for the degrees of freedom.

  class Galerkin {
  public:
    
    // Default constructor (default method will be used)
    Galerkin();
    
    // Constructor allowing specification of method
    Galerkin(FiniteElement& element, Map& map, Quadrature& quadrature);
    
    // Destructor
    ~Galerkin();
    
    // Assemble and set boundary conditions
    void assemble(PDE& pde, Mesh& mesh, Matrix& A, Vector& b);
    
    // Assemble A and b individually and set boundary conditions
    void assemble(PDE& pde, Mesh& mesh, Matrix& A);
    void assemble(PDE& pde, Mesh& mesh, Vector& b);

    // Assemble A and b individually, without setting boundary conditions
    void assembleLHS(PDE& pde, Mesh& mesh, Matrix& A);
    void assembleRHS(PDE& pde, Mesh& mesh, Vector& b);
    
    // Set boundary conditions
    void setBC(Mesh& mesh, Matrix& A);
    void setBC(Mesh& mesh, Vector& b);
    
  private:
    
    void init(Mesh& mesh);
    
    void alloc(Matrix& A, Mesh& mesh);
    void alloc(Vector& b, Mesh& mesh);
    
    // Method data
    FiniteElement* element;    // The finite element
    Map*       map;    // Map from reference cell
    Quadrature*    quadrature; // Quadrature on reference cell
    
    // True if user specifies method
    bool user;
    
  };

}

#endif
