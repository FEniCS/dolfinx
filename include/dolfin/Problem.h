#ifndef __PROBLEM_HH
#define __PROBLEM_HH

namespace dolfin {

  class Grid;
  class Solver;
  
  class Problem {
  public:

	 Problem(const char *problem);
	 Problem(const char *problem, Grid &grid);
	 
	 void set(const char *property, ...);
	 void solve();

  private:
				
	 Grid *grid;
	 Solver *solver;

  };

}

#endif
