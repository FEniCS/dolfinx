#include <iostream>

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
  
  cout << "Reading system info:" << endl;
  cout << "  user    = " << _user;
  cout << "  date    = " << _date;
  cout << "  host    = " << _host;
  cout << "  machine = " << _mach;
  cout << "  sysname = " << _name;
  cout << "  version = " << _vers;
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
