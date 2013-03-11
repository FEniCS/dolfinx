// Copyright (C) 2004-2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2004-01-03
// Last changed: 2005

#include "log.h"
#include "Event.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Event::Event(const std::string msg, unsigned int maxcount) :
  _msg(msg), _maxcount(maxcount), _count(0)
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
    info(_msg);

  _count++;

  if ( _count == _maxcount && _maxcount > 1 )
    info("Last message repeated %d times. Not displaying again.", _count);
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
