// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PRODUCT_H
#define __PRODUCT_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Point.h>
#include <dolfin/FunctionSpace.h>
#include <dolfin/Integral.h>

namespace dolfin {

  class FunctionSpace::ShapeFunction;
  class FunctionSpace::ElementFunction;
  
  class FunctionSpace::Product {
  public:
	 
    // Empty constructor for v = 1
    Product();
    
    // Initialisation
    Product(const ShapeFunction &v);
    Product(const Product       &v);
    
    // Constructors for v0 * v1
    Product(const ShapeFunction &v0, const ShapeFunction &v1);
    Product(const Product       &v0, const Product       &v1);
    Product(const ShapeFunction &v0, const Product       &v1);
    
    // Destructor
    ~Product();
    
    // Assignment (v0 * v1)
    void set(const ShapeFunction &v0, const ShapeFunction &v1);
    void set(const Product       &v0, const Product       &v1);
    void set(const ShapeFunction &v0, const Product       &v1);
    
    // Get id
    int* id() const;
    
    // True if equal to zero
    bool zero() const;
    
    // True if equal to unity
    bool one() const;
    
    // Get number of factors
    int size() const;
    
    //--- Operators ---
    
    // Evaluation
    real operator() (real x, real y, real z, real t) const;
    real operator() (Point p) const;
    
    // Assignment
    Product& operator= (const ShapeFunction &v);
    Product& operator= (const Product       &v);
    
    // Addition
    ElementFunction operator+ (const ShapeFunction   &v) const;
    ElementFunction operator+ (const Product         &v) const;
    ElementFunction operator+ (const ElementFunction &v) const;
    
    // Subtraction
    ElementFunction operator- (const ShapeFunction   &v) const;
    ElementFunction operator- (const Product         &v) const;
    ElementFunction operator- (const ElementFunction &v) const;
    
    // Multiplication
    ElementFunction operator* (real a)                   const;
    Product         operator* (const ShapeFunction   &v) const;
    Product         operator* (const Product         &v) const;
    ElementFunction operator* (const ElementFunction &v) const;
    
    // Integration
    real operator* (Integral::Measure &dm) const;
    
    // Output
    friend LogStream& operator<<(LogStream& stream, const Product &v);
    
  private:
    
    int n;    // Number of factors
    int *_id; // Shape function id:s
    
  };
  
  FunctionSpace::ElementFunction operator* (real a, const FunctionSpace::Product &v);
  
}

#endif
