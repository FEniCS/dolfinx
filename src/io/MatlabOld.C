// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <time.h>
#include <sys/utsname.h>

#include "MatlabOld.hh"
#include "Settings.hh"
#include <dolfin/Display.hh>

extern Settings *settings;

//-----------------------------------------------------------------------------
MatlabOld::MatlabOld(int size)
{
  // Get filenames for datafiles
  settings->Get("output file matlab",      script_file);
  settings->Get("output file matlab data", data_file);

  // Initialize values
  values = new Value(size);

  first_frame = true;
}
//-----------------------------------------------------------------------------
MatlabOld::~MatlabOld()
{
  delete values;
}
//-----------------------------------------------------------------------------
void MatlabOld::Set(int pos, real val)
{
  values->Set(pos,val);
}
//-----------------------------------------------------------------------------
void MatlabOld::SetTime(real t)
{
  values->SetTime(t);
}
//-----------------------------------------------------------------------------
void MatlabOld::SetLabel(int pos, const char *string)
{
  values->SetLabel(pos,string);
}
//-----------------------------------------------------------------------------
void MatlabOld::Save()
{
  // Clear data file first time
  if ( first_frame )
	 ClearData();
  
  // Save data
  SaveData();

  // Save script file first time
  if ( first_frame )
	 SaveScript();

  first_frame = false;

}
//-----------------------------------------------------------------------------
void MatlabOld::Reset()
{
  first_frame = true;
}
//-----------------------------------------------------------------------------
int MatlabOld::Size()
{
  return ( values->Size() );
}
//-----------------------------------------------------------------------------
double MatlabOld::Time()
{
  return ( values->Time() );
}
//-----------------------------------------------------------------------------
double MatlabOld::Get(int pos)
{
  return ( values->Get(pos) );
}
//-----------------------------------------------------------------------------
char *MatlabOld::Label(int pos)
{
  return ( values->Label(pos) );
}
//-----------------------------------------------------------------------------
void MatlabOld::ClearData()
{
  FILE *fp = fopen(data_file,"w");
  if ( !fp )
	 display->Error("Unable to write to data file %s.",data_file);
  
  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabOld::SaveData()
{
  FILE *fp = fopen(data_file,"a");
  if ( !fp )
	 display->Error("Unable to write to data file %s.",data_file);
  
  values->Save(fp);

  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabOld::SaveScript()
{
  FILE *fp = fopen(script_file,"w");
  if ( !fp )
	 display->Error("Unable to write to data file %s.",script_file);

  time_t t;
  time(&t);
  
  struct utsname buf;

  // Get the problem description
  char description[DOLFIN_LINELENGTH];
  settings->Get("problem description",description);

  // Write a nice header
  fprintf(fp,"%% Data from DOLFIN, version %s\n",DOLFIN_VERSION);
  fprintf(fp,"%%\n");
  fprintf(fp,"%% Saved by %s at %s",getenv("USER"),ctime(&t));
  if ( uname(&buf) == 0 )
	 fprintf(fp,"%% on %s (%s) running %s version %s.\n",
				buf.nodename,buf.machine,buf.sysname,buf.release);
  fprintf(fp,"%%\n");
  fprintf(fp,"%% %s\n",description);
  fprintf(fp,"%% Description of data:\n");
  fprintf(fp,"%%\n");
  fprintf(fp,"%%    t      = time values.\n");
  for (int i=0;i<values->Size();i++)
	 fprintf(fp,"%%    v(:,%d) = %s\n",i+1,values->Label(i));
  fprintf(fp,"\n");

  // Read file
  fprintf(fp,"%% Read data from file\n");
  fprintf(fp,"disp('Reading data from file...')\n");
  fprintf(fp,"data = load('%s');\n",data_file);
  fprintf(fp,"disp('Done.')\n");
  fprintf(fp,"disp('Processing data...')\n");
  fprintf(fp,"\n");
  
  // Get dimensions
  fprintf(fp,"%% Get dimensions\n");
  fprintf(fp,"[m,n] = size(data);\n");
  fprintf(fp,"\n");

  // Get the values
  fprintf(fp,"%% Get the values\n");
  fprintf(fp,"t = data(:,1);\n");
  fprintf(fp,"v = data(:,2:n);\n");

  // Print a message
  fprintf(fp,"disp('All data successfully loaded.')\n");
  fprintf(fp,"disp('Type \"help %s\" for more information.')\n",script_file);
  
  // Close the file
  fclose(fp);
}
//-----------------------------------------------------------------------------
