// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-01-03
// Last changed: 2005

#include "log.h"
#include "Event.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Event::Event(const std::string msg, unsigned int maxcount) :
  msg(msg), _maxcount(maxcount), _count(0)
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
    message(msg);
  
  _count++;
  
  if ( _count == _maxcount && _maxcount > 1 )
    message("Last message repeated %d times. Not displaying again.", _count);
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
