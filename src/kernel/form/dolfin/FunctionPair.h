// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_PAIR_H
#define __FUNCTION_PAIR_H

#include <dolfin/NewArray.h>
#include <dolfin/constants.h>

namespace dolfin
{
  
  class Function;
  class NewPDE;

  /// FunctionPair represents a pair of functions (w,f),
  /// with the local function w being the restriction of
  /// the global function f to a cell.
  ///
  /// The local function w is represented by a list of degrees
  /// of freedom, while the global function f is represented
  /// by an object of the class Function.

  class FunctionPair
  {
  public:
    
    /// Create empty function pair
    FunctionPair();
    
    /// Create function pair of given functions
    FunctionPair(NewArray<real>& w, Function& f);
    
    /// Destructor
    ~FunctionPair();

    /// Update local values on given cell at given time t
    void update(const Cell& cell, const NewPDE& pde);
    
  private:
    
    // Degrees of freedom of the local function
    NewArray<real>* w;

    // The global function
    Function* f;
    
  };

}
  
#endif
