// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Mapping from reference cell to actual cell, including
// derivatives of the mapping and the inverse of derivatives.
//
// It is assumed that the mapping is linear and the reference
// cells are given by
//
//   (0) - (1)                             in 1D
//
//   (0,0) - (1,0) - (0,1)                 in 2D
//
//   (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1) in 3D
//
// It is also assumed that x = y = 0 in 1D and z = 0 in 2D.

#ifndef __MAPPING_H
#define __MAPPING_H

#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {

  class Cell;
  
  class Mapping {
  public:

	 Mapping();

	 real det() const;

	 // Derivative of constant
	 real dx(real a) const;
	 real dy(real a) const;
	 real dz(real a) const;
	 real dt(real a) const;

	 // Derivative of ShapeFunction
	 virtual const FunctionSpace::ElementFunction dx(const FunctionSpace::ShapeFunction &v) const = 0;
	 virtual const FunctionSpace::ElementFunction dy(const FunctionSpace::ShapeFunction &v) const = 0;
	 virtual const FunctionSpace::ElementFunction dz(const FunctionSpace::ShapeFunction &v) const = 0;
	 virtual const FunctionSpace::ElementFunction dt(const FunctionSpace::ShapeFunction &v) const = 0;

	 // Derivative of Product
	 const FunctionSpace::ElementFunction dx(const FunctionSpace::Product &v) const;
	 const FunctionSpace::ElementFunction dy(const FunctionSpace::Product &v) const;
	 const FunctionSpace::ElementFunction dz(const FunctionSpace::Product &v) const;
	 const FunctionSpace::ElementFunction dt(const FunctionSpace::Product &v) const;

	 // Derivative of ElementFunction
	 const FunctionSpace::ElementFunction dx(const FunctionSpace::ElementFunction &v) const;
	 const FunctionSpace::ElementFunction dy(const FunctionSpace::ElementFunction &v) const;
	 const FunctionSpace::ElementFunction dz(const FunctionSpace::ElementFunction &v) const;
	 const FunctionSpace::ElementFunction dt(const FunctionSpace::ElementFunction &v) const;

	 // Update mapping
	 virtual void update(const Cell &cell) = 0;
	 
  protected:

	 void reset();
	 
	 int dim;  // Dimension

	 // Jacobian of mapping from reference cell
	 real f11, f12, f13;
	 real f21, f22, f23;
	 real f31, f32, f33;

	 // Inverse of F
	 real g11, g12, g13;
	 real g21, g22, g23;
	 real g31, g32, g33;

	 // Determinant of Jacobian (F)
	 real d;
	 
  };

}

#endif
