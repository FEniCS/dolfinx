// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PROBLEM_POISSON_HH
#define __PROBLEM_POISSON_HH

#include <Problem.hh>

class ProblemPoisson : public Problem {
public:

  ProblemPoisson(Grid *grid) : Problem(grid) {}
  ~ProblemPoisson(){}

  const char *Description();
  void Solve();
    
};

#endif
