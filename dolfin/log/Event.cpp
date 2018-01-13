// Copyright (C) 2004-2005 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Event.h"
#include "log.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Event::Event(const std::string msg, unsigned int maxcount)
    : _msg(msg), _maxcount(maxcount), _count(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Event::~Event()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Event::operator()()
{
  if (_count < _maxcount)
    info(_msg);

  _count++;

  if (_count == _maxcount && _maxcount > 1)
    info("Last message repeated %d times. Not displaying again.", _count);
}
//-----------------------------------------------------------------------------
unsigned int Event::count() const { return _count; }
//-----------------------------------------------------------------------------
unsigned int Event::maxcount() const { return _maxcount; }
//-----------------------------------------------------------------------------
