// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from reference cell to actual cell, including
// derivatives of the map and the inverse of derivatives.
//
// It is assumed that the map is linear and the reference
// cells are given by
//
//   (0) - (1)                             in 1D
//
//   (0,0) - (1,0) - (0,1)                 in 2D
//
//   (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1) in 3D
//
// It is also assumed that x = y = 0 in 1D and z = 0 in 2D.

#ifndef __MAP_H
#define __MAP_H

#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {
  
  class Cell;
  
  class Map {
  public:
    
    Map();
    
    real det() const;
    
    // Derivative of constant
    real ddx(real a) const;
    real ddy(real a) const;
    real ddz(real a) const;
    real ddt(real a) const;
    
    // Derivative of ShapeFunction
    virtual const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction &v) const = 0;
    virtual const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction &v) const = 0;
    virtual const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction &v) const = 0;
    virtual const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction &v) const = 0;
    
    // Derivative of Product
    const FunctionSpace::ElementFunction ddx(const FunctionSpace::Product &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::Product &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::Product &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::Product &v) const;
    
    // Derivative of ElementFunction
    const FunctionSpace::ElementFunction ddx(const FunctionSpace::ElementFunction &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::ElementFunction &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::ElementFunction &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::ElementFunction &v) const;
    
    // Update map
    virtual void update(const Cell &cell) = 0;
    
  protected:
    
    void reset();
    
    int dim;  // Dimension
    
    // Jacobian of map from reference cell
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
