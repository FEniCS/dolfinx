// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/sysinfo.h>
#include <dolfin/System.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
System::System()
{
  update();
}
//-----------------------------------------------------------------------------
void System::update()
{
  sysinfo_user(_user);
  sysinfo_date(_date);
  sysinfo_host(_host);
  sysinfo_mach(_mach);
  sysinfo_name(_name);
  sysinfo_vers(_vers);
  
  cout << "Reading system info:"  << endl;
  cout << "  user    = " << _user << endl;
  cout << "  date    = " << _date << endl;
  cout << "  host    = " << _host << endl;
  cout << "  machine = " << _mach << endl;
  cout << "  sysname = " << _name << endl;
  cout << "  version = " << _vers << endl;
}
//-----------------------------------------------------------------------------
const char* System::user() const
{
  return _user;
}
//-----------------------------------------------------------------------------
const char* System::date() const
{
  return _date;
}
//-----------------------------------------------------------------------------
const char* System::host() const
{
  return _host;
}
//-----------------------------------------------------------------------------
const char* System::mach() const
{
  return _mach;
}
//-----------------------------------------------------------------------------
const char* System::name() const
{
  return _name;

}
//-----------------------------------------------------------------------------
const char* System::vers() const
{
  return _vers;
}
//-----------------------------------------------------------------------------
