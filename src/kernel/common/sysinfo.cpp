#include <stdio.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/sysinfo.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::sysinfo()
{
  char string[DOLFIN_WORDLENGTH];

  std::cout << "- System info:" << std::endl;
  
  sysinfo_user(string);
  std::cout << "- User:    " << string << std::endl;
  
  sysinfo_date(string);
  std::cout << "- Date:    " << string << std::endl;
  
  sysinfo_host(string);
  std::cout << "- Host:    " << string << std::endl;
  
  sysinfo_mach(string);
  std::cout << "- Machine: " << string << std::endl;
  
  sysinfo_name(string);
  std::cout << "- System:  " << string << std::endl;
  
  sysinfo_vers(string);
  std::cout << "- Version: " << string << std::endl;
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
