#ifndef __PROBLEM_H
#define __PROBLEM_H

namespace dolfin {
  
  class Grid;
  class Solver;
  class Settings;
  
  class Problem {
  public:

	 Problem(const char *problem);
	 Problem(const char *problem, Grid &grid);

	 ~Problem();
	 
	 void set(const char *property, ...);
	 void solve();

  private:
				
	 Solver *solver;
	 Settings *settings;

  };

}

#endif
