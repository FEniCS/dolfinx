// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.

#ifndef __INTPUT_HH
#define __INTPUT_HH

#include <dolfin/dolfin_grid.h>
//#include "GridData.hh"
#include <FileType.hh>
#include <dolfin/Display.hh>
#include <utils.h>

#include <string.h>

class Input{
public:

  Input(const char *filename);
  ~Input();

  //void loadGrid(GridData *gd);
  
  //  void ReadHeader (int *no_nodes, int *no_cells, CellType *celltype);
  // void ReadNodes  (Grid *grid, int no_nodes);
  //void ReadCells  (Grid *grid, int no_cells, CellType celltype);

  // Should not need to be used (file is close in the destructor).
  void CloseFile();
  
private:

  FILE *fp;
  
  FileType filetype;
  char filename[DOLFIN_LINELENGTH];
  
};

#endif
