#include "SysInfo.hh"

#include "utils.h"
#include <Display.hh>

//-----------------------------------------------------------------------------
SysInfo::SysInfo()
{
  Update();
}
//-----------------------------------------------------------------------------
SysInfo::~SysInfo()
{

}
//-----------------------------------------------------------------------------
void SysInfo::Update()
{
  env_get_user(user);
  env_get_time(time);
  env_get_host(host);
  env_get_mach(mach);
  env_get_name(name);
  env_get_vers(vers);
  
  display->Message(10,"SysInfo: Reading system info:");
  display->Message(10,"  user    = %s",user);
  display->Message(10,"  time    = %s",time);
  display->Message(10,"  host    = %s",host);
  display->Message(10,"  machine = %s",mach);
  display->Message(10,"  sysname = %s",name);
  display->Message(10,"  version = %s",vers);
}
//-----------------------------------------------------------------------------
