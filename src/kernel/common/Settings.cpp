// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Settings.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void Settings::init()
{
  char *string = new char[DOLFIN_LINELENGTH];
  char *prefix = new char[DOLFIN_LINELENGTH];

  // Get the file path prefix
  get("output file prefix",prefix);
  
  // File name for the primal solution
  if ( !changed("output file primal") ){
	 sprintf(string,"%s-primal.dx",prefix);
	 set("output file primal",string);
  }
	 
  // File name for the dual solution
  if ( !changed("output file dual") ){
	 sprintf(string,"%s-dual.dx",prefix);
	 set("output file dual",string);
  }

  // File name for settings
  if ( !changed("output file settings") ){
	 sprintf(string,"%s-settings.rc",prefix);
	 set("output file settings",string);
  }
  
  // File name for the primal solution
  if ( !changed("output file residual") ){
	 sprintf(string,"%s-residual.dx",prefix);
	 set("output file residual",string);
  }

  // File name for Matlab file
  if ( !changed("output file matlab") ){
	 sprintf(string,"%s_scalars.m",prefix);
	 set("output file matlab",string);
  }
  
  // File name for Matlab file
  if ( !changed("output file matlab data") ){
	 sprintf(string,"%s_scalars.txt",prefix);
	 set("output file matlab data",string);
  }
  
  // Delete temporary variables
  delete [] string;
  delete [] prefix;
}
//----------------------------------------------------------------------------
