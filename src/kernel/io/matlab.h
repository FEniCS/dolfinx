#ifndef __MATLAB_H
#define __MATLAB_H

#include "Output.hh"

void matlab_write_header (FILE *fp, DataInfo *datainfo, SysInfo *sysinfo);
void matlab_write_grid   (FILE *fp, Grid *grid);
void matlab_write_field  (FILE *fp, DataInfo *datainfo, Vector **u, real t, int frame, int no_vectors);

#endif
