// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/LoggerMacros.h>
#include <dolfin/Message.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Message::Message(const std::string message, unsigned int maxcount) :
  message(message), _maxcount(maxcount), _count(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Message::~Message()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Message::display()
{
  if ( _count < _maxcount)
    dolfin_info(message.c_str());
  
  _count++;
  
  if ( _count == _maxcount && _maxcount > 1 )
    dolfin_info("Last message repeated %d times. Not displaying again.", _count);
}
//-----------------------------------------------------------------------------
unsigned int Message::count() const
{
  return _count;
}
//-----------------------------------------------------------------------------
unsigned int Message::maxcount() const
{
  return _maxcount;
}
//-----------------------------------------------------------------------------
