// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PROBLEM_CONVDIFF_HH
#define __PROBLEM_CONVDIFF_HH

#include <Problem.hh>

class GlobalField;

class ProblemConvDiff : public Problem {
public:
  
  ProblemConvDiff(Grid *grid) : Problem(grid) {}
  ~ProblemConvDiff();
  
  const char *Description();
  void Solve();
  
};

#endif
