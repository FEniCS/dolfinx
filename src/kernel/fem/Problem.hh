// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PROBLEM_HH
#define __PROBLEM_HH

#include <Vector.hh>
#include <SparseMatrix.hh>
#include <DenseMatrix.hh>
#include <Grid.hh>
#include <Discretiser.hh>
#include <Display.hh>
#include <KrylovSolver.hh>
#include <SISolver.hh>
#include <DirectSolver.hh>
#include <Output.hh>
#include <Settings.hh>

class Problem{
public:
  
  Problem(Grid *grid);
  ~Problem();

  /// Problem description
  virtual const char *Description() = 0;
  
  /// Solve problem
  virtual void Solve() = 0;
  
protected:
  
  Grid *grid;

  int space_dimension;
  int no_nodes;
  
};

#endif
