// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PROBLEM_H
#define __PROBLEM_H

namespace dolfin {
  
  class Grid;
  class Solver;
  
  class Problem {
  public:
    
    Problem(const char* problem);
    Problem(const char* problem, Grid& grid);
    
    ~Problem();
    
    void set(const char* property, ...);
    void solve();
    
  private:
    
    Solver* solver;

  };

}

#endif
