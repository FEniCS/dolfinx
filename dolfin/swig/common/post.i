/* -*- C -*- */
// Copyright (C) 2008-2011 Johan Hake
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

//-----------------------------------------------------------------------------
// MPI communicator wrappers (deliberately very lightweight)
//-----------------------------------------------------------------------------
//#ifdef HAS_MPI
typedef struct {
} MPI_Comm;
//#endif

// Lightweight wrappers for MPI_COMM_WORLD and MPI_COMM_SELF
%inline %{
  MPI_Comm mpi_comm_world()
  { return MPI_COMM_WORLD; }

  MPI_Comm mpi_comm_self()
  { return MPI_COMM_SELF; }
%}


//-----------------------------------------------------------------------------
// Wrapper of std::terminate, installed into Python exception hook
//-----------------------------------------------------------------------------
%noexception dolfin::dolfin_terminate();
%inline %{
  #include <signal.h>

  namespace dolfin {
    void dolfin_terminate() noexcept
    {
      // Uninstall OpenMPI signal handlers cluttering stderr (only on POSIX)
      #ifndef __MINGW32__
      struct sigaction act;
      memset(&act, 0, sizeof(act));
      act.sa_handler = SIG_IGN;
      sigaction(SIGABRT, &act, NULL);
      #endif

      // We don't bother with MPI_Abort. This would require taking care of
      // MPI state. We just assume mpirun catches SIGABRT and sends SIGTERM
      // to other ranks.
      std::abort();
    }
  }
%}

%pythoncode %{
# Install C++ terminate handler into Python (if not interactive)
# This ensures that std::abort (which is likely to be appropriately
# interpreted by MPI implementations) is called by sys.excepthook thus
# avoiding parallel deadlocks when one process raises
import sys
def _is_interactive():
    return hasattr(sys, "ps1") or sys.flags.interactive
if not _is_interactive():
    def _new_excepthook(*args):
        import sys
        sys.__excepthook__(*args)
        dolfin_terminate()
    sys.excepthook = _new_excepthook
    del _new_excepthook
del _is_interactive, sys
%}


//-----------------------------------------------------------------------------
// Instantiate some DOLFIN MPI templates
//-----------------------------------------------------------------------------

%template(max) dolfin::MPI::max<double>;
%template(min) dolfin::MPI::min<double>;
%template(sum) dolfin::MPI::sum<double>;
%template(max) dolfin::MPI::max<int>;
%template(min) dolfin::MPI::min<int>;
%template(sum) dolfin::MPI::sum<int>;
%template(max) dolfin::MPI::max<unsigned int>;
%template(min) dolfin::MPI::min<unsigned int>;
%template(sum) dolfin::MPI::sum<unsigned int>;
%template(max) dolfin::MPI::max<std::size_t>;
%template(min) dolfin::MPI::min<std::size_t>;
%template(sum) dolfin::MPI::sum<std::size_t>;
%template(max) dolfin::MPI::max<dolfin::Table>;
%template(min) dolfin::MPI::min<dolfin::Table>;
%template(sum) dolfin::MPI::sum<dolfin::Table>;
%template(avg) dolfin::MPI::avg<dolfin::Table>;

//-----------------------------------------------------------------------------
// Ignore const array interface (Used if the Array type is a const)
//-----------------------------------------------------------------------------
%define CONST_ARRAY_IGNORES(TYPE)
%ignore dolfin::Array<const TYPE>::Array(std::size_t N);
%ignore dolfin::Array<const TYPE>::array();
%ignore dolfin::Array<const TYPE>::resize(std::size_t N);
%ignore dolfin::Array<const TYPE>::zero();
%ignore dolfin::Array<const TYPE>::update();
%ignore dolfin::Array<const TYPE>::__setitem__;
%enddef

//-----------------------------------------------------------------------------
// Macro for instantiating the Array templates
//
// TYPE          : The Array template type
// TEMPLATE_NAME : The Template name
// TYPE_NAME     : The name of the pointer type, 'double' for 'double',
//                 'uint' for 'dolfin::uint'
//-----------------------------------------------------------------------------

%define ARRAY_EXTENSIONS(TYPE, TEMPLATE_NAME, TYPE_NAME)

// Construct value wrapper for dolfin::Array<TYPE>
// Valuewrapper is used so a return by value Array does not make an
// extra copy in any typemaps
%feature("valuewrapper") dolfin::Array<TYPE>;

// Cannot construct an Array from another Array.
// Use NumPy Array instead
%ignore dolfin::Array<TYPE>::Array(const Array& other);
%template(TEMPLATE_NAME ## Array) dolfin::Array<TYPE>;
%feature("docstring") dolfin::Array::__getitem__ "Missing docstring";
%feature("docstring") dolfin::Array::__setitem__ "Missing docstring";
%feature("docstring") dolfin::Array::array "Missing docstring";
%extend dolfin::Array<TYPE>
{
  TYPE _getitem(unsigned int i) const { return (*self)[i]; }
  void _setitem(unsigned int i, const TYPE& val) { (*self)[i] = val; }

  PyObject * _array()
  {
    return %make_numpy_array(1, TYPE_NAME)(self->size(), self->data(), true);
  }

  %pythoncode%{
def array(self):
    """
    Return a NumPy array view of object
    """
    data = self._array()
    _attach_base_to_numpy_array(data, self)
    return data

def __getitem__(self, index):
    if not isinstance(index, int):
        raise TypeError("expected an int as index argument")
    while index < 0:
        index += self.size()
    if index >= self.size():
        raise IndexError("index out of range")
    return self._getitem(index)

def __setitem__(self, index, value):
    if not isinstance(index, int):
        raise TypeError("expected an int as index argument")
    while index < 0:
        index += self.size()
    if index >= self.size():
        raise IndexError("index out of range")
    self._setitem(index, value)

def __len__(self):
    return self.size()
    %}
}
%enddef

//-----------------------------------------------------------------------------
// Run Array macros, which also instantiate the templates
//-----------------------------------------------------------------------------
CONST_ARRAY_IGNORES(double)
ARRAY_EXTENSIONS(double, Double, double)
ARRAY_EXTENSIONS(unsigned int, UInt, uint)
ARRAY_EXTENSIONS(int, Int, int)

//-----------------------------------------------------------------------------
// Add pretty print for Variables
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Variable::__str__ "Missing docstring";
%extend dolfin::Variable
{
  std::string __str__() const
  {
    return self->str(false);
  }
}

//-----------------------------------------------------------------------------
// Fixup docstrings
//-----------------------------------------------------------------------------
%pythoncode
%{
for f in [timings, list_timings, dump_timings_to_xml, timing]:
    doc = f.__doc__
    doc = doc.replace("TimingType::", "TimingType_")
    doc = doc.replace("TimingClear::", "TimingClear_")
    doc = doc.replace("std::set<TimingType>", "list")
    doc = doc.replace("{ ", "[")
    doc = doc.replace(" }", "]")
    f.__doc__ = doc
    del doc
del f
%}
