// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SYSINFO_H
#define __SYSINFO_H

namespace dolfin {

  void sysinfo();
  
  void sysinfo_user(char* string);
  void sysinfo_date(char* string);
  void sysinfo_host(char* string);
  void sysinfo_mach(char* string);
  void sysinfo_name(char* string);
  void sysinfo_vers(char* string);

  void sysinfo_dolfin(char* string);

}
  
#endif
