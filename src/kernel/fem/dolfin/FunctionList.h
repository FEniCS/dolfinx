// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_LIST_H
#define __FUNCTION_LIST_H

#include <dolfin/function.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {
  
  class Mapping;
  
  class FunctionList {
  public:
    
    FunctionList();
    
    // Addition of functions
    static int add(function f);
    
    // Specification of derivatives wrt reference coordinates
    static void set(int id,
		    FunctionSpace::ElementFunction dX,
		    FunctionSpace::ElementFunction dY,
		    FunctionSpace::ElementFunction dZ,
		    FunctionSpace::ElementFunction dT);
    
    // Update derivatives with current mapping
    static void update(const FunctionSpace::ShapeFunction& v, const Mapping& mapping);
	 
    // Size of list
    static int size();
    
    // Evaluation
    static real eval(int id, real x, real y, real z, real t);
    
    // Derivatives wrt real coordinates
    static const FunctionSpace::ElementFunction& dx(int id);
    static const FunctionSpace::ElementFunction& dy(int id);
    static const FunctionSpace::ElementFunction& dz(int id);
    static const FunctionSpace::ElementFunction& dt(int id);
    
    // Derivatives wrt reference coordinates
    static const FunctionSpace::ElementFunction& dX(int id);
    static const FunctionSpace::ElementFunction& dY(int id);
    static const FunctionSpace::ElementFunction& dZ(int id);
    static const FunctionSpace::ElementFunction& dT(int id);
    
  private:
    
    class FunctionData {
    public:
      
      FunctionData();
      FunctionData(function f);
      
      // Needed for ShortList
      void operator= (int zero);
      bool operator! () const;
      
      function f;                        // Function pointer
      
      FunctionSpace::ElementFunction dx; // Derivative wrt dx
      FunctionSpace::ElementFunction dy; // Derivative wrt dy
      FunctionSpace::ElementFunction dz; // Derivative wrt dz
      FunctionSpace::ElementFunction dt; // Derivative wrt dt
      
      FunctionSpace::ElementFunction dX; // Derivative wrt dX
      FunctionSpace::ElementFunction dY; // Derivative wrt dY
      FunctionSpace::ElementFunction dZ; // Derivative wrt dZ
      FunctionSpace::ElementFunction dT; // Derivative wrt dT
    };
    
    static void init();
    
    static ShortList<FunctionData> list;
    static int _size;
    static bool initialised;
    
  };
  
}

#endif
