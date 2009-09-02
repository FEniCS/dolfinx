// Copyright (C) 2006-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2009-09-02

//-----------------------------------------------------------------------------
// Global renames
//-----------------------------------------------------------------------------
%rename(__setitem__) *::setitem;
%rename(__getitem__) *::getitem;
%rename(__float__) *::operator double;
