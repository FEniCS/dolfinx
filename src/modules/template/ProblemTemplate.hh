// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#ifndef __PROBLEM_TEMPLATE_HH
#define __PROBLEM_TEMPLATE_HH

#include <Problem.hh>

class ProblemTemplate : public Problem {
public:

  ProblemTemplate(Grid *grid) : Problem(grid) {}
  ~ProblemTemplate(){}

  const char *Description();
  void Solve();
    
};

#endif
