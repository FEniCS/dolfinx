// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.

#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <stdarg.h>
#include <stdio.h>
#include <dolfin/constants.h>
#include <SysInfo.h>
#include <DataInfo.h>
#include <dolfin/Grid.h>
#include <dolfin/Vector.h>
#include <FileType.h>

namespace dolfin {

  class Output{
  public:
	 
	 Output(int no_data, ...);
	 Output(int no_data, va_list aptr);
	 Output(const char *filename, int no_data, ...);
	 
	 ~Output();
	 
	 /// Set label for data
	 void SetLabel(int i, const char *name, const char *label);
	 /// Add frame to the file
	 //  void AddFrame(Grid *grid, Vector **u, real t, int no_vectors = 1);
	 /// Reset the file, overwriting previously saved frames
	 void Reset();
	 
  protected:
	 
	 void InitCommon(const char *filename, int no_data, va_list aptr);
	 void GenerateFileName();
	 void SaveTimeValue(real t);
	 
	 //  void MatlabAddFrame (FILE *fp, Grid *grid, Vector **u, real t, int no_vectors);
	 // void OpenDXAddFrame (FILE *fp, Grid *grid, Vector **u, real t, int no_vectors);
	 //void GiDAddFrame    (FILE *fp, Grid *grid, Vector **u, real t, int no_vectors);
	 
	 // Filename for saving output
	 char filename[DOLFIN_LINELENGTH];
	 char grid_filename[DOLFIN_LINELENGTH]; // Used when needed
	 
	 // Data dimensions total size of data
	 int *dimensions;
	 int datasize;
	 
	 // Indicators for saving
	 int current_frame;
	 bool done_grid;
	 bool done_header; 
	 
	 // Time values
	 Vector *time_values;
	 
	 // Problem description
	 char problem_description[DOLFIN_LINELENGTH];
	 
	 // System info
	 SysInfo sysinfo;
	 
	 // Info about data labels and variable names
	 DataInfo *datainfo;
	 
	 // File type
	 FileType filetype;
	 
  };

}

#endif
