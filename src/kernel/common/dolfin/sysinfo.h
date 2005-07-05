// Copyright (C) 2002-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005

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
