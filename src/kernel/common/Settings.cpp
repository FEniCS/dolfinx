// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Settings.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void Settings::Initialize()
{
  char *string = new char[DOLFIN_LINELENGTH];
  char *prefix = new char[DOLFIN_LINELENGTH];

  // Get the file path prefix
  Get("output file prefix",prefix);
  
  // File name for the primal solution
  if ( !Changed("output file primal") ){
	 sprintf(string,"%s-primal.dx",prefix);
	 Set("output file primal",string);
  }
	 
  // File name for the dual solution
  if ( !Changed("output file dual") ){
	 sprintf(string,"%s-dual.dx",prefix);
	 Set("output file dual",string);
  }

  // File name for settings
  if ( !Changed("output file settings") ){
	 sprintf(string,"%s-settings.rc",prefix);
	 Set("output file settings",string);
  }
  
  // File name for the primal solution
  if ( !Changed("output file residual") ){
	 sprintf(string,"%s-residual.dx",prefix);
	 Set("output file residual",string);
  }

  // File name for Matlab file
  if ( !Changed("output file matlab") ){
	 sprintf(string,"%s_scalars.m",prefix);
	 Set("output file matlab",string);
  }
  
  // File name for Matlab file
  if ( !Changed("output file matlab data") ){
	 sprintf(string,"%s_scalars.txt",prefix);
	 Set("output file matlab data",string);
  }
  
  // Delete temporary variables
  delete [] string;
  delete [] prefix;
}
//----------------------------------------------------------------------------
