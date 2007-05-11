// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-06-18
// Last changed: 2007-05-11

#include <dolfin/SilentLogger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SilentLogger::SilentLogger() : GenericLogger()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SilentLogger::~SilentLogger()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::info(const char* msg)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::debug(const char* msg, const char* location)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::warning(const char* msg, const char* location)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::error(const char* msg, const char* location)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::dassert(const char* msg, const char* location)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SilentLogger::progress(const char* title, const char* label, real p)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
