#ifndef __OPENDX_H
#define __OPENDX_H

#include "Output.hh"

void opendx_write_header  (FILE *fp, DataInfo *datainfo, SysInfo *sysinfo);
void opendx_write_grid    (FILE *fp, Grid *grid);
void opendx_write_time    (FILE *fp, real t);
void opendx_write_field   (FILE *fp, DataInfo *datainfo, Grid *grid, int frame, Vector **u, int no_vectors);
void opendx_write_series  (FILE *fp, DataInfo *datainfo, int frame, Vector *t);
void opendx_remove_series (FILE *fp, DataInfo *datainfo);

#endif
