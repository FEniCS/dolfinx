// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SYSTEM_H
#define __SYSTEM_H

#include <dolfin/constants.h>

namespace dolfin {

  class System {
  public:
    
    System();
    ~System();
    
    void update();
    
    const char* user() const;
    const char* date() const;
    const char* host() const;
    const char* mach() const;
    const char* name() const;
    const char* vers() const;
    
  private:
    
    char _user[DOLFIN_LINELENGTH];
    char _date[DOLFIN_LINELENGTH];
    char _host[DOLFIN_LINELENGTH];
    char _mach[DOLFIN_LINELENGTH];
    char _name[DOLFIN_LINELENGTH];
    char _vers[DOLFIN_LINELENGTH];
    
  };

}
  
#endif
