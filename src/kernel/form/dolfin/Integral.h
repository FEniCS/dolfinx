// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __INTEGRAL_H
#define __INTEGRAL_H

#include <dolfin/constants.h>
#include <dolfin/Tensor.h>
#include <dolfin/FunctionSpace.h>

namespace dolfin {

  class Quadrature;
  class Map;
  
  class Integral
  {
  public:

    // Forward declarations of nested classes
    class Measure;
    class InteriorMeasure;
    class BoundaryMeasure;
    
    /// Integral measure (base class)
    class Measure
    {
    public:
      
      Measure();
      Measure(const Map& map, const Quadrature& quadrature);
      virtual ~Measure();
      
      // Update map and quadrature
      virtual void update(const Map& map, const Quadrature& quadrature);
      
      // Integration operators dm * v
      real operator* (real a) const;
      virtual real operator* (const FunctionSpace::ShapeFunction& v) = 0;
      virtual real operator* (const FunctionSpace::Product& v) = 0;
      real operator* (const FunctionSpace::ElementFunction& v);
      
    protected:
      
      // Return determinant of map
      virtual real det() const = 0;
            
      // Integral data
      class Value
      {
      public:
	
	Value() { value = 0.0; computed = false; }
	real operator() () { return value; }
	bool ok() { return computed; }
	void set(real value) { this->value = value; computed = true; }
	void operator= (int a) { value = 0.0; computed = false;	}
	
      private:
	
	real value;
	bool computed;
	
      };
      
      // Map from reference cell
      const Map* m;

      // Quadrature rule on reference cell
      const Quadrature* q;
      
      // True if the measure is active
      bool active;

    };
    
    // Integral measure for the interior of an element
    class InteriorMeasure : public Measure
    {
    public:
      
      /// Constructor
      InteriorMeasure();

      /// Constructor
      InteriorMeasure(Map& m, Quadrature& q);

      // Destructor
      ~InteriorMeasure();
      
      // Update map and quadrature
      void update(const Map& map, const Quadrature& quadrature);

      // Return integral of shape function
      real operator* (const FunctionSpace::ShapeFunction& v);

      // Return integral of product of shape functions
      real operator* (const FunctionSpace::Product& v);

    private:
      
      // Evaluate integral of shape function
      real integral(const FunctionSpace::ShapeFunction& v);

      // Evaluate integral of product of shape functions
      real integral(const FunctionSpace::Product& v);
      
      // Return determinant of map
      real det() const;
      
      // Init table
      virtual void init();
      
      // Resize table
      virtual void resize(unsigned int new_order, unsigned int new_n);

      // A lookup table for integrals
      Tensor<Value>* table;

      // Maximum number of factors
      unsigned int order;

      // Number of different shape functions
      unsigned int n;
            
    };
    
    // Integral measure for the boundary of an element
    class BoundaryMeasure : public Measure
    {
    public:
      
      // Constructor
      BoundaryMeasure();

      // Constructor
      BoundaryMeasure(Map& m, Quadrature& q);
      
      // Destructor
      ~BoundaryMeasure();

      // Update map and quadrature
      void update(const Map& map, const Quadrature& quadrature);

      // Return integral of shape function
      real operator* (const FunctionSpace::ShapeFunction& v);

      // Return integral of product of shape functions
      real operator* (const FunctionSpace::Product& v);

    private:
      
      // Evaluate integral of shape function
      real integral(const FunctionSpace::ShapeFunction& v);

      // Evaluate integral of product of shape functions
      real integral(const FunctionSpace::Product& v);
      
      // Return determinant of map
      real det() const;

      // Init table
      virtual void init();
      
      // Resize table
      virtual void resize(unsigned int new_order, unsigned int new_n);

      // A lookup table for integrals
      Tensor<Value>** table;
      
      // Maximum number of factors
      unsigned int order;

      // Number of different shape functions
      unsigned int n;

      // Current boundary
      int boundary;
      
      // Maximum number of different boundaries
      unsigned int bndmax;

    };
    
  };
  
  real operator* (real a, const Integral::Measure& dm);
  
}

#endif
