// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005

#include <dolfin/File.h>
#include <dolfin/SettingsManager.h>
#include <dolfin/SettingsMacros.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::dolfin_parameter(Parameter::Type type,const char *identifier,...)
{
  va_list aptr;
  va_start(aptr, identifier);
  
  SettingsManager::settings.add_aptr(type, identifier, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_set(const char *identifier, ...)
{
  va_list aptr;
  va_start(aptr, identifier);

  SettingsManager::settings.set_aptr(identifier, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_set_aptr(const char *identifier, va_list aptr)
{
  SettingsManager::settings.set_aptr(identifier, aptr);
}
//-----------------------------------------------------------------------------
Parameter dolfin::dolfin_get(const char *identifier)
{
  return SettingsManager::settings.get(identifier);
}
//-----------------------------------------------------------------------------
bool dolfin::dolfin_parameter_changed(const char* identifier)
{
  return SettingsManager::settings.changed(identifier);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_load(const char* filename)
{
  File file(filename);
  file >> SettingsManager::settings;
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_save(const char* filename)
{
  File file(filename);
  file << SettingsManager::settings;
}
//-----------------------------------------------------------------------------
