#ifndef __INP_H
#define __INP_H

#include "Input.hh"

void inp_read_header (FILE *fp, int *no_nodes, int *no_cells, CellType *celltype);
void inp_read_nodes  (FILE *fp, Grid *grid, int no_nodes);
void inp_read_cells  (FILE *fp, Grid *grid, int no_cells, CellType celltype);

#endif
