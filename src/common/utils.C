// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <time.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include <Display.hh>

char buffer[DOLFIN_LINELENGTH];

//-----------------------------------------------------------------------------
bool suffix(const char *string, const char *suffix)
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
void remove_newline(char *string)
{
  for (int i=0;string[i];i++)
	 if ( string[i] == '\n' ){
		string[i] = '\0';
		return;
	 }
}
//-----------------------------------------------------------------------------
bool end_of_file(FILE *fp)
{
  char c = fgetc(fp);

  if ( c == EOF )
	 return true;

  ungetc(c,fp);
  return false;
}
//-----------------------------------------------------------------------------
bool keyword_in_line(FILE *fp, const char *keyword)
{
  // Check that we didn't reach end of file
  if ( end_of_file(fp) )
	 display->Error("Reached end of file while looking for keyword \"%s\".",keyword);
  
  // Read next line of the file
  fgets(buffer,DOLFIN_LINELENGTH,fp);

  // Check if the line contains the keyword
  if ( strstr(buffer,keyword) )
	 return true;

  return false;
}
//-----------------------------------------------------------------------------
bool keyword_in_line(FILE *fp, const char *keyword, char *line)
{
  // Check that we didn't reach end of file
  if ( end_of_file(fp) )
	 display->Error("Reached end of file while looking for keyword \"%s\".",keyword);
  
  // Read next line of the file
  fgets(line,DOLFIN_LINELENGTH,fp);

  // Check if the line contains the keyword
  if ( strstr(line,keyword) )
	 return true;

  return false;
}
//-----------------------------------------------------------------------------
void skip_line(FILE *fp)
{
  // Read next line of the file
  fgets(buffer,DOLFIN_LINELENGTH,fp);
}
//-----------------------------------------------------------------------------
int factorial(int n)
{
  if (n<0) display->InternalError("utils::factorial","n! for n<0 not implemented"); 
  int fact = 1;
  for (int i=2;i<n+1;i++) fact *= i; 
  return fact;
}
//-----------------------------------------------------------------------------
real sqr(real x)
{
  return ( x*x );
}
//-----------------------------------------------------------------------------
real max(real x, real y)
{
  return ( x > y ? x : y );
}
//-----------------------------------------------------------------------------
real min(real x, real y)
{
  return ( x < y ? x : y );
}
//-----------------------------------------------------------------------------
bool contains(int *list, int size, int number)
{
  for (int i=0;i<size;i++)
	 if ( list[i] == number )
		return true;
  
  return false;
}
//-----------------------------------------------------------------------------
void env_get_user(char *string)
{
  sprintf(string,"%s",getenv("USER"));
}
//-----------------------------------------------------------------------------
void env_get_time(char *string)
{
  time_t t;
  time(&t);

  sprintf(string,"%s",ctime(&t));
  remove_newline(string);
}
//-----------------------------------------------------------------------------
void env_get_host(char *string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.nodename);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void env_get_mach(char *string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.machine);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void env_get_name(char *string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.sysname);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
void env_get_vers(char *string)
{
  struct utsname buf;
  if ( uname(&buf) == 0 )
	 sprintf(string,"%s",buf.release);
  else
	 sprintf(string,"<unknown>");
}
//-----------------------------------------------------------------------------
