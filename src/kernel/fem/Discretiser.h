// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DISCRETISER_HH
#define __DISCRETISER_HH

#include "Grid.hh"
#include <SparseMatrix.hh>
#include <Vector.hh>
#include <dolfin.h>

class Grid;
class Equation;
class SparseMatrix;
class Vector;

class Discretiser{
public:

  Discretiser(Grid *grid, Equation *equation);
  ~Discretiser();

  // Assemble and set boundary conditions
  void Assemble(SparseMatrix *A, Vector *b);
  
  // Assemble A and b individually, without setting boundary conditions
  void AssembleLHS (SparseMatrix *A);
  void AssembleRHS (Vector *b);

  // Set boundary conditions
  void SetBoundaryConditions(SparseMatrix *A, Vector *b);
  
private:

  void Allocate (SparseMatrix *A);
  void Allocate (Vector *b);
  
  Grid *grid;
  Equation *equation;

  int no_eq;
  int no_nodes;
  int size;
  int space_dimension;
  
  dolfin_bc (*bc_function) (real x, real y, real z, int node, int component);
};

#endif
