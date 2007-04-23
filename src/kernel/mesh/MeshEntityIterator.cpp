// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-12
// Last changed: 2007-04-12

#include <dolfin/MeshEntityIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntityIterator& it)
{
  stream << "[ Mesh entity iterator at position "
	 << it._pos
	 << " stepping from 0 to "
	 << it.pos_end - 1
	 << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
