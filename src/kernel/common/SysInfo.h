#ifndef __SYSINFO_HH
#define __SYSINFO_HH

#include <dolfin/constants.h>

class SysInfo{
public:

  SysInfo();
  ~SysInfo();

  void Update();
  
  char user[DOLFIN_LINELENGTH];
  char time[DOLFIN_LINELENGTH];
  char host[DOLFIN_LINELENGTH];
  char mach[DOLFIN_LINELENGTH];
  char name[DOLFIN_LINELENGTH];
  char vers[DOLFIN_LINELENGTH];
  
};

#endif
