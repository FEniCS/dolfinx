// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <time.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/sysinfo.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::sysinfo()
{
  char string[DOLFIN_WORDLENGTH];

  cout << "- System info:" << endl;
  
  sysinfo_user(string);
  cout << "- User:    " << string << endl;
  
  sysinfo_date(string);
  cout << "- Date:    " << string << endl;
  
  sysinfo_host(string);
  cout << "- Host:    " << string << endl;
  
  sysinfo_mach(string);
  cout << "- Machine: " << string << endl;
  
  sysinfo_name(string);
  cout << "- System:  " << string << endl;
  
  sysinfo_vers(string);
  cout << "- Version: " << string << endl;
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_user(char* string)
{
  sprintf(string,"%s",getenv("USER"));
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_date(char* string)
{
  time_t t;
  time(&t);

  sprintf(string,"%s",ctime(&t));
  remove_newline(string);
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_host(char* string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.nodename);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_mach(char* string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.machine);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_name(char* string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.sysname);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void dolfin::sysinfo_vers(char* string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.release);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
