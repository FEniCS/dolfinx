// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.

// FIXME: GiD output does not seem to work in the way it is described in the manual.
// FIXME: Temporary solution works to some extent.

#ifndef __GID_H
#define __GID_H

#include "Input.hh"
#include "Output.hh"

void gid_read_header  (FILE *fp, int *no_nodes, int *no_cells, CellType *celltype);
void gid_read_nodes   (FILE *fp, Grid *grid, int no_nodes);
void gid_read_cells   (FILE *fp, Grid *grid, int no_cells, CellType celltype);

void gid_write_header (FILE *fp, DataInfo *datainfo, SysInfo *sysinfo);
void gid_write_grid   (FILE *fp, Grid *grid);
void gid_write_field  (FILE *fp, Grid *grid, DataInfo *datainfo, Vector **u, real t, int frame, int no_vectors);

#endif
