// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-01-03
// Last changed: 2005

#include <dolfin/log.h>
#include <dolfin/Event.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Event::Event(const std::string message, unsigned int maxcount) :
  message(message), _maxcount(maxcount), _count(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Event::~Event()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Event::operator() ()
{
  if ( _count < _maxcount)
    dolfin_info(message.c_str());
  
  _count++;
  
  if ( _count == _maxcount && _maxcount > 1 )
    dolfin_info("Last message repeated %d times. Not displaying again.", _count);
}
//-----------------------------------------------------------------------------
unsigned int Event::count() const
{
  return _count;
}
//-----------------------------------------------------------------------------
unsigned int Event::maxcount() const
{
  return _maxcount;
}
//-----------------------------------------------------------------------------
