// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/SilentLogger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SilentLogger::SilentLogger() : GenericLogger()
{
  
}
//-----------------------------------------------------------------------------
SilentLogger::~SilentLogger()
{
  
}
//-----------------------------------------------------------------------------
void SilentLogger::info(const char* msg)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::debug(const char* msg, const char* location)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::warning(const char* msg, const char* location)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::error(const char* msg, const char* location)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::dassert(const char* msg, const char* location)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::progress(const char* title, const char* label, real p)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::update()
{

}
//-----------------------------------------------------------------------------
void SilentLogger::quit()
{

}
//-----------------------------------------------------------------------------
bool SilentLogger::finished()
{

}
//-----------------------------------------------------------------------------
void SilentLogger::progress_add(Progress* p)
{

}
//-----------------------------------------------------------------------------
void SilentLogger::progress_remove(Progress *p)
{

}
//-----------------------------------------------------------------------------
