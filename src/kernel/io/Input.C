#include "Input.hh"
#include <dolfin/Display.hh>

//#include "inp.h"
//#include "gid.h"

// Input works in the following way:
//
//   1. ReadHeader() reads the number of nodes, cells and cell type
//   2. ReadNodes()  then reads all nodes
//   3. ReadCells()  then reads all cells
//
// The file is open by ReadHeader() (if it is not already opened).
// The file pointer is not rewound between calls, unless the format specific
// functions take care of that themselves. (For the inp-format we would like
// to read sequentially and for the gid-format we need to rewind the file
// since we first need to count the number of nodes and cells.)

//-----------------------------------------------------------------------------
Input::Input(const char *filename)
{
  sprintf(this->filename,"%s",filename);

  // Check if the file exists
  fp = fopen(filename,"r");
  if ( !fp )
	 display->Error("Unable to read grid file \"%s\".",filename);
  fclose(fp);
  
  // Determine the file type
  filetype = GetFileType(filename);
}
//-----------------------------------------------------------------------------
//void Input::loadGrid(GridData *gd)
//{
//  display->Message(0,"Not implemented");
//}
//-----------------------------------------------------------------------------
Input::~Input()
{
  
}
//-----------------------------------------------------------------------------
/*
void Input::ReadHeader(int *no_nodes, int *no_cells, CellType *celltype)
{
  // Read header
  switch ( filetype ){
  case FILE_INP:
	 inp_read_header(fp,no_nodes,no_cells,celltype);
	 break;
  case FILE_OPENDX:
	 display->Error("Input from OpenDX is not implemented.");
	 break;
  case FILE_MATLAB:
	 display->Error("Input from MATLAB is not implemented. Use the 'writeinp.m' script.");
	 break;
  case FILE_GID:
  	 gid_read_header(fp,no_nodes,no_cells,celltype);
	 break;
  default:
	 display->Error("Unknown file type for grid file \"%s\".",filename);
  }
}
//-----------------------------------------------------------------------------
void Input::ReadNodes(Grid *grid, int no_nodes)
{
  switch ( filetype ){
  case FILE_INP:
	 inp_read_nodes(fp,grid,no_nodes);
	 break;
  case FILE_OPENDX:
	 display->Error("Input from OpenDX is not implemented.");
	 break;
  case FILE_MATLAB:
	 display->Error("Input from MATLAB is not implemented. Use the 'writeinp.m' script.");
	 break;
  case FILE_GID:
  	 gid_read_nodes(fp,grid,no_nodes);
	 break;
  default:
	 display->Error("Unknown file type for grid file \"%s\".",filename);
  }
}
//-----------------------------------------------------------------------------
void Input::ReadCells(Grid *grid, int no_cells, CellType celltype)
{
  switch ( filetype ){
  case FILE_INP:
	 inp_read_cells(fp,grid,no_cells,celltype);
	 break;
  case FILE_OPENDX:
	 display->Error("Input from OpenDX is not implemented.");
	 break;
  case FILE_MATLAB:
	 display->Error("Input from MATLAB is not implemented. Use the 'writeinp.m' script.");
	 break;
  case FILE_GID:
  	 gid_read_cells(fp,grid,no_cells,celltype);
	 break;
  default:
	 display->Error("Unknown file type for grid file \"%s\".",filename);
  }
}
//-----------------------------------------------------------------------------
void Input::CloseFile()
{
  // Close the file
  if ( fp )
	 fclose(fp);
  fp = 0;
}
//-----------------------------------------------------------------------------
*/
