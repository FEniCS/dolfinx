#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <iostream.h>
#include <time.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include "ParameterList.hh"
#include <Display.hh>

//----------------------------------------------------------------------------
ParameterList::ParameterList()
{
  // Default size for list of parameters, will be reallocated if necessary
  alloc_size = DOLFIN_PARAMSIZE;
  alloc_size_functions = DOLFIN_PARAMSIZE;
  
  // Initialize the parameters
  parameters       = new Parameter[alloc_size];
  functions        = new Function[alloc_size_functions];
  current          = 0;
  current_function = 0;
}
//----------------------------------------------------------------------------
ParameterList::~ParameterList()
{
  if ( parameters )
	 delete [] parameters;
  parameters = 0;
  
  if ( functions )
	 delete [] functions;
  functions = 0;
}
//----------------------------------------------------------------------------
void ParameterList::Initialize()
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
	 
  // Delete temporary variables
  delete [] string;
  delete [] prefix;
}
//----------------------------------------------------------------------------
void ParameterList::Add(const char *identifier, Type type, ...)
{
  va_list aptr;
  va_start(aptr,type);

  double val_double;
  int    val_int;
  bool   val_bool;
  char  *val_string;

  // Check if we need more memory for the list
  if ( current >= alloc_size )
	 Realloc();
  
  // Set the data
  parameters[current].Set(identifier,type,aptr);

  // Step to the next parameter
  current += 1;
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::AddFunction(const char *identifier)
{
  // Check if we need more memory for the list
  if ( current_function >= alloc_size_functions )
	 ReallocFunctions();

  // Set the data
  functions[current_function].Set(identifier);

  // Step to the next function
  current_function += 1;
}
//----------------------------------------------------------------------------
void ParameterList::SetByArgumentList(const char *identifier, va_list aptr)
{
  // Go through the list of parameters and see which one matches

  int index = ParameterIndex(identifier);

  if ( index >= 0 ){
	 parameters[index].Set(identifier,aptr);
	 return;
  }
	 
  // If we get here, then we didn't find the parameter in the list
  display->Warning("Trying to set value of unknown parameter \"%s\".",
						 identifier);
}
//----------------------------------------------------------------------------
void ParameterList::Set(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);
  
  SetByArgumentList(identifier,aptr);
  
  va_end(aptr);
}
//----------------------------------------------------------------------------
void ParameterList::SetFunction(const char *identifier, FunctionPointer f)
{
  // Go through the list of functions and see which one matches
  
    int index = FunctionIndex(identifier);

  if ( index >= 0 ){
	 functions[index].Set(identifier,f);
	 return;
  }
	 
  // If we get here, then we didn't find the function in the list
  display->Warning("Unknown function: \"%s\".",identifier);
}
//----------------------------------------------------------------------------
void ParameterList::GetByArgumentList(const char *identifier, va_list aptr)
{
  // Go through the list of settings and look for the requested parameter.
  
  int index = ParameterIndex(identifier);
  
  if ( index >= 0 ){
	 parameters[index].Get(aptr);
	 return;
  }
  
  // If we get here we didn't find the parameter
  display->Warning("Unknown parameter: \"%s\".",identifier);
  
}
//----------------------------------------------------------------------------
void ParameterList::Get(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr,identifier);

  GetByArgumentList(identifier,aptr);

  va_end(aptr);
}
//----------------------------------------------------------------------------
FunctionPointer ParameterList::GetFunction(const char *identifier)
{
  // Go through the list of functions and see which one matches
  
  int index = FunctionIndex(identifier);
  
  if ( index >= 0 )
	 return functions[index].f;

  // If we get here, then we didn't find the function in the list
  display->Warning("Unknown function: \"%s\".",identifier);  
  
  return 0;
}
//----------------------------------------------------------------------------
void ParameterList::Save()
{
  char filename[DOLFIN_LINELENGTH];
  Get("output file settings",&filename);
  Save(filename);
}
//----------------------------------------------------------------------------
void ParameterList::Save(const char *filename)
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
  Get("problem description",description);
  
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
void ParameterList::Load()
{
  char filename[DOLFIN_LINELENGTH];
  Get("output file settings",&filename);
  Load(filename);
}
//----------------------------------------------------------------------------
void ParameterList::Load(const char *filename)
{
  cout << "Loading of parameters not implemented." << endl;
}
//----------------------------------------------------------------------------
bool ParameterList::Changed(const char *identifier)
{
  int index = ParameterIndex(identifier);

  if ( index >= 0 )
	 return parameters[index].Changed();

  cout << "Error: Status for unknown parameter <" << identifier << "> not available." << endl;
}
//----------------------------------------------------------------------------
int ParameterList::ParameterIndex(const char *identifier)
{
  for (int i=0;i<current;i++)
	 if ( parameters[i].Matches(identifier) )
		return i;

  return -1;
}
//----------------------------------------------------------------------------
int ParameterList::FunctionIndex(const char *identifier)
{
  for (int i=0;i<current;i++)
	 if ( functions[i].Matches(identifier) )
		return i;
  
  return -1;
}
//----------------------------------------------------------------------------
void ParameterList::Realloc()
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
void ParameterList::ReallocFunctions()
{
  Function tmp[alloc_size_functions];
  
  // Place all data in the temporary storage
  for (int i=0;i<alloc_size_functions;i++)
	 tmp[i] = functions[i];
  
  // Double the size
  delete [] functions;
  alloc_size_functions *= 2;
  functions = new Function[alloc_size_functions];
  
  // Put the data back
  for (int i=0;i<alloc_size_functions;i++)
	 functions[i] = tmp[i];
}
//----------------------------------------------------------------------------
