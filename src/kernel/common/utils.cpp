// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <time.h>
#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
bool dolfin::suffix(const char *string, const char *suffix)
{
  // Step to end of string
  int i=0;
  for (;string[i];i++);

  // Step to end of suffix
  int j=0;
  for (;suffix[j];j++);

  // String can not be shorter than suffix
  if ( i<j )
         return false;
  
  // Compare
  for (int k=i-j;k<i;k++)
         if ( string[k] != suffix[k-i+j] )
                return false;

  return true;
}
//-----------------------------------------------------------------------------
void dolfin::remove_newline(char *string)
{
  for (int i=0;string[i];i++)
	 if ( string[i] == '\n' ){
		string[i] = '\0';
		return;
	 }
}
//-----------------------------------------------------------------------------
int dolfin::length(const char *string)
{
  int n = 0;
  for (; string[n]; n++);
  return n;
}
//-----------------------------------------------------------------------------
void dolfin::delay(real seconds)
{
  if ( seconds < 0 ) {
    dolfin_warning("Delay must be positive.");
    return;
  }
  
  struct timespec req;
  req.tv_sec  = (int) floor(seconds);
  req.tv_nsec = (int) (1000000000.0 * (seconds - floor(seconds)));
  
  nanosleep(&req, 0);
}
//-----------------------------------------------------------------------------
