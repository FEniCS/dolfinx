/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-05-10
// Last changed: 2010-10-07

//=============================================================================
// SWIG directives for the DOLFIN log kernel module (post)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Make progress available from Python through the __iadd__ interface
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Progress::_add "Missing docstring";
%feature("docstring") dolfin::Progress::_set "Missing docstring";
%extend dolfin::Progress {

void _add(int incr) {
    for (int j=0;j<incr; ++j)
        (*self)++;
}

void _set(double value) {
    *self = value;
}

%pythoncode
%{
def __iadd__(self, other):
    if isinstance(other, int):
        self._add(other)
    elif isinstance(other, float):
        self._set(other)
    return self

def update(self, other):
    "Update the progress with given number"
    if isinstance(other, float):
        self._set(other)
%}

}

//-----------------------------------------------------------------------------
// Use traceback in debug message
// Reimplement info
//-----------------------------------------------------------------------------
%pythoncode %{
def debug(message):
    import traceback
    file, line, func, txt = traceback.extract_stack(None, 2)[0]
    __debug(file, line, func, message)

def info(*args):
    if args and isinstance(args[0], int):
        if args[0] < get_log_level():
            return
        args = args[1:]

    if len(args) > 0 and isinstance(args[0],(Variable,Parameters)):
        if len(args) > 1:
            _info(args[0].str(*args[1:]))
        else:
            _info(args[0].str(False))
    else:
        _info(*args)
%}
