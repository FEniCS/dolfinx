// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.

#ifndef __VECTOR_HH
#define __VECTOR_HH

#include "kw_constants.h"

class DirectSolver;

class Vector{
public:

  Vector  ();
  Vector  (int n);
  ~Vector ();

  void Resize(int n);
  
  void CopyTo(Vector* vec);
  void CopyFrom(Vector* vec);
  void SetToConstant(real val);

  int Size();
  
  void Set(int i, real val);
  real Get(int i);
  
  void Add  (int i, real val);
  void Add  (real a, Vector *v);
  void Mult (int i, real val);
  real Dot  (Vector *v);
  real Norm ();
  real Norm (int i);

  real operator()(int i);

  void Display();
  void Write(const char *filename);
  
  friend class DirectSolver;
  
private:

  int n;

  real *values;
};

#endif
