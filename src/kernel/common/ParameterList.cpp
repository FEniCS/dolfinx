#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <iostream.h>
#include <time.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include "ParameterList.h"
#include <dolfin/Display.h>

using namespace dolfin;

//----------------------------------------------------------------------------
ParameterList::ParameterList()
{
  // Default size for list of parameters, will be reallocated if necessary
  alloc_size = DOLFIN_PARAMSIZE;
  
  // Initialize the parameters
  parameters = new Parameter[alloc_size];
  current    = 0;
}
//----------------------------------------------------------------------------
ParameterList::~ParameterList()
{
  if ( parameters )
	 delete [] parameters;
  parameters = 0;
}
//----------------------------------------------------------------------------
void ParameterList::add(Parameter::Type type, const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);

  double val_double;
  int    val_int;
  bool   val_bool;
  char  *val_string;

  // Check if we need more memory for the list
  if ( current >= alloc_size )
	 realloc();
  
  // Set the data
  parameters[current].set(type, identifier, aptr);

  // Step to the next parameter
  current += 1;
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::set(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);
  
  set_aptr(identifier,aptr);
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::set_aptr(const char *identifier, va_list aptr)
{
  // Go through the list of parameters and see which one matches

  int index = getIndex(identifier);

  if ( index >= 0 ){
	 parameters[index].set(identifier,aptr);
	 return;
  }
  
  // If we get here, then we didn't find the parameter in the list
  display->Warning("Trying to set value of unknown parameter \"%s\".",
						 identifier);
}
//----------------------------------------------------------------------------
void ParameterList::get(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);

  get_aptr(identifier,aptr);

  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::get_aptr(const char *identifier, va_list aptr)
{
  // Go through the list of settings and look for the requested parameter.
  
  int index = getIndex(identifier);
  
  if ( index >= 0 ){
	 parameters[index].get(aptr);
	 return;
  }
  
  // If we get here we didn't find the parameter
  display->Warning("Unknown parameter: \"%s\".",identifier);
  
}
//----------------------------------------------------------------------------
void ParameterList::save()
{
  char filename[DOLFIN_LINELENGTH];
  get("output file settings",&filename);
  save(filename);
}
//----------------------------------------------------------------------------
void ParameterList::save(const char *filename)
{
  display->Message(0,"Saving parameters to %s...",filename);
  
  FILE *fp;

  // Open the file
  fp = fopen(filename,"w");
  if ( !fp ){
	 display->Warning("Unable to save parameters to file \"s\".",filename);
	 return;
  }

  time_t t;
  time(&t);
  
  struct utsname buf;

  char description[DOLFIN_LINELENGTH];
  get("problem description",description);
  
  // Write a nice header
  fprintf(fp,"# Parameters for DOLFIN, version %s\n",DOLFIN_VERSION);
  fprintf(fp,"#\n");
  fprintf(fp,"# Saved by %s at %s",getenv("USER"),ctime(&t));
  if ( uname(&buf) == 0 )
	 fprintf(fp,"# on %s (%s) running %s version %s.\n",
				buf.nodename,buf.machine,buf.sysname,buf.release);
  fprintf(fp,"#\n");
  fprintf(fp,"# %s\n",description);
  fprintf(fp,"\n");
  
  // Find the maximum length of all parameters
  int length = 0;
  int thislength;
  for (int i=0;i<current;i++)
	 if ( (thislength = parameters[i].StringLength()) > length )
		length = thislength;
  
  // Write the parameters
  for (int i=0;i<current;i++)
	 parameters[i].WriteToFile(fp,length-parameters[i].StringLength());

  // Close the file
  fclose(fp);
	 
}
//----------------------------------------------------------------------------
void ParameterList::load()
{
  char filename[DOLFIN_LINELENGTH];
  get("output file settings",&filename);
  load(filename);
}
//----------------------------------------------------------------------------
void ParameterList::load(const char *filename)
{
  cout << "Loading of parameters not implemented." << endl;
}
//----------------------------------------------------------------------------
bool ParameterList::changed(const char *identifier)
{
  int index = getIndex(identifier);

  if ( index >= 0 )
	 return parameters[index].changed();

  cout << "Error: Status for unknown parameter <" << identifier << "> not available." << endl;
}
//----------------------------------------------------------------------------
int ParameterList::getIndex(const char *identifier)
{
  for (int i=0;i<current;i++)
	 if ( parameters[i].matches(identifier) )
		return i;
  
  return -1;
}
//----------------------------------------------------------------------------
void ParameterList::realloc()
{
  Parameter tmp[alloc_size];

  // Place all data in the temporary storage
  for (int i=0;i<alloc_size;i++)
	 tmp[i] = parameters[i];
  
  // Double the size
  delete [] parameters;
  alloc_size *= 2;
  parameters = new Parameter[alloc_size];
  
  // Put the data back
  for (int i=0;i<alloc_size;i++)
	 parameters[i] = tmp[i];
}
//----------------------------------------------------------------------------
