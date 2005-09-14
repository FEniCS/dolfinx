// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-09-14

#include <dolfin/File.h>
#include <dolfin/SettingsManager.h>
#include <dolfin/SettingsMacros.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::dolfin_parameter(Parameter::Type type, const char *key,...)
{
  va_list aptr;
  va_start(aptr, key);
  
  SettingsManager::settings.add_aptr(type, key, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_set(const char *key, ...)
{
  va_list aptr;
  va_start(aptr, key);

  SettingsManager::settings.set_aptr(key, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_set_aptr(const char *key, va_list aptr)
{
  SettingsManager::settings.set_aptr(key, aptr);
}
//-----------------------------------------------------------------------------
Parameter dolfin::dolfin_get(const char *key)
{
  return SettingsManager::settings.get(key);
}
//-----------------------------------------------------------------------------
bool dolfin::dolfin_parameter_changed(const char* key)
{
  return SettingsManager::settings.changed(key);
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
