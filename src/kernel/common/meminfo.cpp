// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/meminfo.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::meminfo()
{
  // Get process id
  int pid = (int) getpid();

  // Write values from /proc to temporary file
  char command[DOLFIN_WORDLENGTH];
  
  system("/bin/rm -f meminfo.tmp");
  
  sprintf(command, "cat /proc/%d/status | grep VmSize | /usr/bin/awk '{print \" \"$2\" \"$3}' >> meminfo.tmp", pid);
  system(command);
  
  sprintf(command, "cat /proc/%d/status | grep VmRSS  | /usr/bin/awk '{print \" \"$2\" \"$3}' >> meminfo.tmp", pid);
  system(command);

  // Get variables and print
  FILE *fp = fopen("meminfo.tmp", "r");

  dolfin_info("Memory usage for process %d:", pid);

  fgets(command, DOLFIN_WORDLENGTH, fp); remove_newline(command);
  dolfin_info("Size:     %s", command);

  fgets(command, DOLFIN_WORDLENGTH, fp); remove_newline(command);
  dolfin_info("Resident: %s", command);

  fclose(fp);
}
//-----------------------------------------------------------------------------
