// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DOF_FUNCTION_H
#define __DOF_FUNCTION_H

#include <dolfin/ElementFunction.h>
#include <dolfin/GenericFunction.h>

namespace dolfin {

  class Cell;
  class FiniteElement;
  class Vector;
  class Mesh;
  class NewPDE;
  
  class DofFunction : public GenericFunction {
  public:
    
    DofFunction(Mesh& mesh, Vector& dofs, int dim, int size);
    ~DofFunction();
	 
    // Evaluation of function
    real operator() (const Node&  n, real t) const;
    real operator() (const Node&  n, real t);

    // Update function to given time
    void update(real t);

    // Return current time
    real time() const;

    // Return the mesh
    Mesh& mesh() const;

    // Update values of element function
    void update(FunctionSpace::ElementFunction &v,
		const FiniteElement &element,
		const Cell &cell, real t) const;

    // Update local function (restriction to given cell)
    void update(NewArray<real>& w, const Cell& cell, const NewPDE& pde) const;
    
  private:
    
    Mesh& _mesh;
    Vector& x;
    real t;

    unsigned int dim;
    unsigned int size;
    
  };

}

#endif
