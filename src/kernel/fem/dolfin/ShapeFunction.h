// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHAPE_FUNCTION_H
#define __SHAPE_FUNCTION_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/function.h>
#include <dolfin/Point.h>
#include <dolfin/FunctionSpace.h>
#include <dolfin/Integral.h>

namespace dolfin {

  class FunctionSpace::Product;
  class FunctionSpace::ElementFunction;
  
  class FunctionSpace::ShapeFunction {
  public:
    
    // Empty constructor for v = 0
    ShapeFunction();
    
    // Constructor for v = 1
    ShapeFunction(int id);
    
    // Initialisation
    ShapeFunction(function f);
    ShapeFunction(const ShapeFunction &v);
    
    // Specification of derivatives
    void set(ElementFunction dX, ElementFunction dY, ElementFunction dZ, ElementFunction dT);
    
    // Get id
    int id() const;
    
    // True if equal to zero
    bool zero() const;
    
    // True if equal to unity
    bool one() const;
    
    // Derivatives
    const ElementFunction& dx() const;
    const ElementFunction& dy() const;
    const ElementFunction& dz() const;
    const ElementFunction& dt() const;
    
    const ElementFunction& dX() const;
    const ElementFunction& dY() const;
    const ElementFunction& dZ() const;
    const ElementFunction& dT() const;
    
    // Update
    void update(const Mapping& mapping);
    
    //--- Operators ---
    
    // Evaluation
    real operator() (real x, real y, real z, real t) const;
    real operator() (Point p) const;
    
    // Assignment
    ShapeFunction& operator= (const ShapeFunction &v);
    
    // Addition
    ElementFunction operator+ (const ShapeFunction   &v) const;
    ElementFunction operator+ (const Product         &v) const;
    ElementFunction operator+ (const ElementFunction &v) const;
    
    // Subtraction
    ElementFunction operator- (const ShapeFunction   &v) const;
    ElementFunction operator- (const Product         &v) const;
    ElementFunction operator- (const ElementFunction &v) const;
    
    // Multiplication
    ElementFunction operator* (const real a)             const;
    Product         operator* (const ShapeFunction   &v) const;
    Product         operator* (const Product         &v) const;
    ElementFunction operator* (const ElementFunction &v) const;
    
    // Integration
    real operator* (Integral::Measure &dm) const;
    
    // Needed for ShortList
    void operator= (int zero);
    bool operator! () const;
    
    // Output
    friend LogStream& operator<<(LogStream& stream, const ShapeFunction &v);
    
  private:
    
    int _id;
    
  };
  
  FunctionSpace::ElementFunction operator* (real a, const FunctionSpace::ShapeFunction &v);
  
}

#endif
