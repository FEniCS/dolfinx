#ifndef __FUNCTION_LIST_H
#define __FUNCTION_LIST_H

#include <dolfin/function.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {

  class FunctionList {
  public:

	 // Addition of functions
	 static int add(function f);

	 // Specification of derivatives
	 static void set(int id,
						  FunctionSpace::ElementFunction dx,
						  FunctionSpace::ElementFunction dy,
						  FunctionSpace::ElementFunction dz,
						  FunctionSpace::ElementFunction dt);
	 
	 // Evaluation
	 static real eval(int id, real x, real y, real z, real t);

	 // Derivatives
	 static FunctionSpace::ElementFunction dx(int id);
	 static FunctionSpace::ElementFunction dy(int id);
	 static FunctionSpace::ElementFunction dz(int id);
	 static FunctionSpace::ElementFunction dt(int id);
	 
  private:

	 class FunctionData {
	 public:

		FunctionData();
		FunctionData(function f);

		bool operator! ();
		void operator= (int i);
		
		function f;                        // Function pointer
		FunctionSpace::ElementFunction dx; // Derivative wrt dx
		FunctionSpace::ElementFunction dy; // Derivative wrt dy
		FunctionSpace::ElementFunction dz; // Derivative wrt dz
		FunctionSpace::ElementFunction dt; // Derivative wrt dt
	 };

	 static void init();
	 
	 static ShortList<FunctionData> list;
	 static bool initialised;
	 
  };

}

#endif

