#include "Output.hh"
#include <Display.hh>
#include <Settings.hh>
#include <utils.h>

#include "matlab.h"
#include "opendx.h"
#include "gid.h"

extern Settings *settings;

// Data is assumed to be stored in the following way
//
// Assume three variables: u = (u1,u2,u3), p and w = (w1,w2)
//
// If no_vectors = 1: Then the vector should contain u,p,w in the order
//
//     [ u1 u2 u3 p w1 w2 u1 u2 u3 p w1 w2 ... ]
//
// If no_vectors > 2: Then no_vectors must be = 3 and the 3 vectors should be
//
//     [ u1 u2 u3 u1 u2 u3 ... ], [ p p ... ], [ w1 w2 w1 w2 ... ]

//-----------------------------------------------------------------------------
Output::Output(int no_data, ...)
{
  // Parse input
  va_list aptr;
  va_start(aptr,no_data);

  // If no filename is specified, generate a default file name
  GenerateFileName();

  // Common initialisations
  InitCommon(filename,no_data,aptr);

  // Clean up
  va_end(aptr);
}
//-----------------------------------------------------------------------------
Output::Output(int no_data, va_list aptr)
{
  // If no filename is specified, generate a default file name
  GenerateFileName();
    
  // Common initialisations
  InitCommon(filename,no_data,aptr);
}
//-----------------------------------------------------------------------------
Output::Output(const char *filename, int no_data, ...)
{
  // Parse input
  va_list aptr;
  va_start(aptr,no_data);

  // Common initialisations
  InitCommon(filename,no_data,aptr);

  // FIXME: gid file name generation
  if ( filetype == FILE_GID )
	 display->Error("You cannot specify file name for output to GiD format (only prefix).");
  
  // Clean up
  va_end(aptr);
}
//-----------------------------------------------------------------------------
Output::~Output()
{
  if ( time_values )
	 delete time_values;
  time_values = 0;

  if ( dimensions )
	 delete [] dimensions;
  dimensions = 0;

  if ( datainfo )
	 delete datainfo;
  datainfo = 0;
}
//-----------------------------------------------------------------------------
void Output::SetLabel(int i, const char *name, const char *label)
{
  datainfo->SetLabel(i,name,label);
}
//----------------------------------------------------------------------------
void Output::AddFrame(Grid *grid, Vector **u, real t, int no_vectors = 1)
{
  FILE *fp = 0;

  // Open the file
  fp = fopen(filename,"r+");
  if ( !fp )
	 display->Error("Unable to write to output file \"%s\".",filename);
  
  // Check that no_vectors is ok
  if ( no_vectors != 1 )
	 if ( no_vectors != datainfo->Size() )
		display->InternalError("Output::AddFrame",
									  "Number of vectors (%d) does not match size of data set (%d).",
									  no_vectors,datainfo->Size());

  // Update system variables
  sysinfo.Update();

  // Save the time value
  SaveTimeValue(t);
  
  switch ( filetype ){
  case FILE_OPENDX:
	 OpenDXAddFrame(fp,grid,u,t,no_vectors);
	 break;
  case FILE_MATLAB:
	 MatlabAddFrame(fp,grid,u,t,no_vectors);
	 break;
  case FILE_GID:
	 GiDAddFrame(fp,grid,u,t,no_vectors);
	 break;
  default:
	 display->InternalError("Output::AddFrame()","Unknown file type.");
  }

  // Increase frame counter
  current_frame += 1;

  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void Output::Reset()
{
  // Clear the file
  FILE *fp = fopen(filename,"w");
  if ( !fp )
	 display->Error("Unable to write to output file \"%s\".",filename);
  fclose(fp);
  
  // Reset frame counter
  current_frame = 0;

  // No header yet
  done_header = false;
  
  // No grid yet
  done_grid = false;
}
//----------------------------------------------------------------------------
void Output::InitCommon(const char *filename, int no_data, va_list aptr)
{
  // Determine file type
  if ( suffix(filename,".dx") ){
	 filetype = FILE_OPENDX;
  }
  else if ( suffix(filename,".m") ){
	 filetype = FILE_MATLAB;
  }
  else if ( suffix(filename,".res") ){
	 filetype = FILE_GID;
  }
  else
	 display->Error("Unable to save data to file %s. Unknown file type.",filename);
  
  // Save the file name
  sprintf(this->filename,filename);
  
  // Get the problem description
  settings->Get("problem description",problem_description);
  
  // Allocate memory for the dimension array
  dimensions = new int[no_data];
  
  // Get dimensions of the data
  for (int i=0;i<no_data;i++)
	 dimensions[i] = va_arg(aptr,int);

  // Initialise information about variable labels and names
  datainfo = new DataInfo(problem_description,no_data,dimensions);
  
  // Compute total data size
  datasize = 0;
  for (int i=0;i<no_data;i++)
	 datasize += dimensions[i];

  // Check if the file exists
  FILE *fp = fopen(filename,"r");
  if ( fp ){
	 fclose(fp);
	 display->Error("Output file \"%s\" already exists. Remove or rename the file first.",
						 filename);
  }

  // Allocate memory for time values
  time_values = new Vector(DOLFIN_PARAMSIZE);
  
  // Reset: clear file and reset counter
  Reset();
}
//----------------------------------------------------------------------------
void Output::GenerateFileName()
{
  char prefix[DOLFIN_LINELENGTH];
  char filetype[DOLFIN_LINELENGTH];
  int space_dimension;
  
  settings->Get("output file prefix",prefix);
  settings->Get("output file type",filetype);
  settings->Get("space dimension",&space_dimension);

  // Default file type is MATLAB for 2d problems and OpenDX for 3d problems
  
  if ( strcasecmp(filetype,"opendx") == 0 )
	 sprintf(filename,"%s.dx",prefix);
  else if ( strcasecmp(filetype,"matlab") == 0 )
	 sprintf(filename,"%s.m",prefix);
  else if ( strcasecmp(filetype,"gid") == 0 ){
	 sprintf(filename,"%s.flavia.res",prefix);
	 sprintf(grid_filename,"%s.flavia.msh",prefix);
  }
  else if ( strcasecmp(filetype,"") == 0 ){
	 if ( space_dimension == 2 )
		sprintf(filename,"%s.m",prefix);
	 else
		sprintf(filename,"%s.dx",prefix);
  }
  else
	 display->Error("Unknown file type: \"%s\".",filetype);	 
	 
  display->Message(5,"Setting filename: \"%s\".",filename);
}
//----------------------------------------------------------------------------
void Output::SaveTimeValue(real t)
{
  // Reallocate if necessary
  
  if ( current_frame > (time_values->Size()-1) ){
	 Vector *tmp = time_values;
	 time_values = new Vector(time_values->Size()*2);
	 for (int i=0;i<tmp->Size();i++)
		time_values->Set(i,tmp->Get(i));
	 delete tmp;
  }

  // Save the time value
  time_values->Set(current_frame,t);
}
//----------------------------------------------------------------------------
void Output::MatlabAddFrame(FILE *fp, Grid *grid, Vector **u, real t,
									 int no_vectors)
{
  display->Message(5,"Adding frame in MATLAB format.");
  
  // Step to the end of the file
  fseek(fp,0L,SEEK_END);
  
  // Write header
  if ( !done_header ){
	 matlab_write_header(fp,datainfo,&sysinfo);
	 done_header = true;
  }
  
  // Write grid
  if ( !done_grid ){
	 matlab_write_grid(fp,grid);
	 done_grid = true;
  }

  // Write field
  matlab_write_field(fp,datainfo,u,t,current_frame,no_vectors);
}
//----------------------------------------------------------------------------
void Output::OpenDXAddFrame(FILE *fp, Grid *grid, Vector **u, real t,
									 int no_vectors)
{
  display->Message(5,"Adding frame in OpenDX format.");
  
  // Step to the end of the file
  fseek(fp,0L,SEEK_END);
  
  // Write header the first time
  if ( !done_header ){
	 opendx_write_header(fp,datainfo,&sysinfo);
	 done_header = true;
  }
  
  // Write grid the first time
  if ( !done_grid ){
	 opendx_write_grid(fp,grid);
	 done_grid = true;
  }

  // Remove the previous time series
  if ( current_frame > 0 )
	 opendx_remove_series(fp,datainfo);

  // Write field
  opendx_write_field(fp,datainfo,grid,current_frame,u,no_vectors);

  // Write the time series
  opendx_write_series(fp,datainfo,current_frame,time_values);
}
//----------------------------------------------------------------------------
void Output::GiDAddFrame(FILE *fp, Grid *grid, Vector **u, real t,
								 int no_vectors)
{
  display->Message(5,"Adding frame in GiD format.");
  
  // Step to the end of the file
  fseek(fp,0L,SEEK_END);
  
  // Write header the first time
  if ( !done_header ){
	 gid_write_header(fp,datainfo,&sysinfo);
	 done_header = true;
  }
  
  // Write grid the first time
  if ( !done_grid ){
	 FILE *fp_grid = fopen(grid_filename,"w");
	 if ( !fp_grid )
		display->Error("Unable to write to output file \"%s\".",filename);
	 
	 gid_write_grid(fp_grid,grid);
	 done_grid = true;

	 fclose(fp_grid);
  }

  // Write field
  gid_write_field(fp,grid,datainfo,u,t,current_frame,no_vectors);

}
//----------------------------------------------------------------------------
