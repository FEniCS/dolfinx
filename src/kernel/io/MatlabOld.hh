// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MATLAB_OLD_HH
#define __MATLAB_OLD_HH

#include "kw_constants.h"
#include "Value.hh"

/// Saves scalar data to Matlab/Octave format
class MatlabOld{
public:

  MatlabOld(int size);
  ~MatlabOld();

  void   Set     (int pos, real val);
  void   SetTime (real t);
  void   SetLabel(int pos, const char *string);

  void   Reset   ();
  void   Save    ();
  
  int    Size    ();
  double Time    ();
  double Get     (int pos);
  char  *Label   (int pos);

private:

  void ClearData  ();
  void SaveData   ();
  void SaveScript ();
  
  char script_file[DOLFIN_LINELENGTH];
  char data_file[DOLFIN_LINELENGTH];

  Value *values;
  bool first_frame;
  
};

#endif
