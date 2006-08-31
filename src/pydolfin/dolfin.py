# This file was created automatically by SWIG 1.3.29.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _dolfin
import new
new_instancemethod = new.instancemethod
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


try:
    import weakref
    weakref_proxy = weakref.proxy
except:
    weakref_proxy = lambda x: x



def new_realArray(*args):
  """new_realArray(size_t nelements) -> real"""
  return _dolfin.new_realArray(*args)

def delete_realArray(*args):
  """delete_realArray(real ary)"""
  return _dolfin.delete_realArray(*args)

def realArray_getitem(*args):
  """realArray_getitem(real ary, size_t index) -> real"""
  return _dolfin.realArray_getitem(*args)

def realArray_setitem(*args):
  """realArray_setitem(real ary, size_t index, real value)"""
  return _dolfin.realArray_setitem(*args)

def new_intArray(*args):
  """new_intArray(size_t nelements) -> int"""
  return _dolfin.new_intArray(*args)

def delete_intArray(*args):
  """delete_intArray(int ary)"""
  return _dolfin.delete_intArray(*args)

def intArray_getitem(*args):
  """intArray_getitem(int ary, size_t index) -> int"""
  return _dolfin.intArray_getitem(*args)

def intArray_setitem(*args):
  """intArray_setitem(int ary, size_t index, int value)"""
  return _dolfin.intArray_setitem(*args)

def dolfin_init(*args):
  """dolfin_init(int argc, char argv)"""
  return _dolfin.dolfin_init(*args)

def sqr(*args):
  """sqr(real x) -> real"""
  return _dolfin.sqr(*args)

def ipow(*args):
  """ipow(uint a, uint n) -> uint"""
  return _dolfin.ipow(*args)

def rand(*args):
  """rand() -> real"""
  return _dolfin.rand(*args)

def seed(*args):
  """seed(unsigned int s)"""
  return _dolfin.seed(*args)
class TimeDependent(_object):
    """Proxy of C++ TimeDependent class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependent, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependent, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> TimeDependent
        __init__(self, real t) -> TimeDependent
        """
        this = _dolfin.new_TimeDependent(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_TimeDependent
    __del__ = lambda self : None;
    def sync(*args):
        """sync(self, real t)"""
        return _dolfin.TimeDependent_sync(*args)

    def time(*args):
        """time(self) -> real"""
        return _dolfin.TimeDependent_time(*args)

TimeDependent_swigregister = _dolfin.TimeDependent_swigregister
TimeDependent_swigregister(TimeDependent)

class Variable(_object):
    """Proxy of C++ Variable class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Variable, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Variable, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Variable
        __init__(self, string name, string label) -> Variable
        __init__(self, Variable variable) -> Variable
        """
        this = _dolfin.new_Variable(*args)
        try: self.this.append(this)
        except: self.this = this
    def rename(*args):
        """rename(self, string name, string label)"""
        return _dolfin.Variable_rename(*args)

    def name(*args):
        """name(self) -> string"""
        return _dolfin.Variable_name(*args)

    def label(*args):
        """label(self) -> string"""
        return _dolfin.Variable_label(*args)

    def number(*args):
        """number(self) -> int"""
        return _dolfin.Variable_number(*args)

Variable_swigregister = _dolfin.Variable_swigregister
Variable_swigregister(Variable)


def suffix(*args):
  """suffix(char string, char suffix) -> bool"""
  return _dolfin.suffix(*args)

def remove_newline(*args):
  """remove_newline(char string)"""
  return _dolfin.remove_newline(*args)

def length(*args):
  """length(char string) -> int"""
  return _dolfin.length(*args)

def date(*args):
  """date() -> string"""
  return _dolfin.date(*args)

def delay(*args):
  """delay(real seconds)"""
  return _dolfin.delay(*args)

def tic(*args):
  """tic()"""
  return _dolfin.tic(*args)

def toc(*args):
  """toc() -> real"""
  return _dolfin.toc(*args)

def tocd(*args):
  """tocd() -> real"""
  return _dolfin.tocd(*args)

def dolfin_update(*args):
  """dolfin_update()"""
  return _dolfin.dolfin_update(*args)

def dolfin_quit(*args):
  """dolfin_quit()"""
  return _dolfin.dolfin_quit(*args)

def dolfin_finished(*args):
  """dolfin_finished() -> bool"""
  return _dolfin.dolfin_finished(*args)

def dolfin_segfault(*args):
  """dolfin_segfault()"""
  return _dolfin.dolfin_segfault(*args)

def dolfin_output(*args):
  """dolfin_output(char destination)"""
  return _dolfin.dolfin_output(*args)

def dolfin_log(*args):
  """dolfin_log(bool state)"""
  return _dolfin.dolfin_log(*args)
class Parameter(_object):
    """Proxy of C++ Parameter class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Parameter, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Parameter, name)
    __repr__ = _swig_repr
    type_real = _dolfin.Parameter_type_real
    type_int = _dolfin.Parameter_type_int
    type_bool = _dolfin.Parameter_type_bool
    type_string = _dolfin.Parameter_type_string
    def __init__(self, *args): 
        """
        __init__(self, int value) -> Parameter
        __init__(self, uint value) -> Parameter
        __init__(self, real value) -> Parameter
        __init__(self, bool value) -> Parameter
        __init__(self, string value) -> Parameter
        __init__(self, char value) -> Parameter
        __init__(self, Parameter parameter) -> Parameter
        """
        this = _dolfin.new_Parameter(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Parameter
    __del__ = lambda self : None;
    def type(*args):
        """type(self) -> int"""
        return _dolfin.Parameter_type(*args)

Parameter_swigregister = _dolfin.Parameter_swigregister
Parameter_swigregister(Parameter)

def dolfin_begin(*args):
  """
    dolfin_begin()
    dolfin_begin(char msg, v(...) ?)
    """
  return _dolfin.dolfin_begin(*args)

def dolfin_end(*args):
  """
    dolfin_end()
    dolfin_end(char msg, v(...) ?)
    """
  return _dolfin.dolfin_end(*args)

class File(_object):
    """Proxy of C++ File class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, File, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, File, name)
    __repr__ = _swig_repr
    xml = _dolfin.File_xml
    matlab = _dolfin.File_matlab
    matrixmarket = _dolfin.File_matrixmarket
    octave = _dolfin.File_octave
    opendx = _dolfin.File_opendx
    gid = _dolfin.File_gid
    tecplot = _dolfin.File_tecplot
    vtk = _dolfin.File_vtk
    python = _dolfin.File_python
    def __init__(self, *args): 
        """
        __init__(self, string filename) -> File
        __init__(self, string filename, Type type) -> File
        """
        this = _dolfin.new_File(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_File
    __del__ = lambda self : None;
    def __rshift__(*args):
        """
        __rshift__(self, Vector x)
        __rshift__(self, Matrix A)
        __rshift__(self, Mesh mesh)
        __rshift__(self, NewMesh mesh)
        __rshift__(self, Function u)
        __rshift__(self, Sample sample)
        __rshift__(self, FiniteElementSpec spec)
        __rshift__(self, ParameterList parameters)
        __rshift__(self, BLASFormData blas)
        """
        return _dolfin.File___rshift__(*args)

    def __lshift__(*args):
        """
        __lshift__(self, Vector x)
        __lshift__(self, Matrix A)
        __lshift__(self, Mesh mesh)
        __lshift__(self, NewMesh mesh)
        __lshift__(self, Function u)
        __lshift__(self, Sample sample)
        __lshift__(self, FiniteElementSpec spec)
        __lshift__(self, ParameterList parameters)
        __lshift__(self, BLASFormData blas)
        """
        return _dolfin.File___lshift__(*args)

File_swigregister = _dolfin.File_swigregister
File_swigregister(File)

class ublas_dense_matrix(_object):
    """Proxy of C++ ublas_dense_matrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ublas_dense_matrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ublas_dense_matrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
ublas_dense_matrix_swigregister = _dolfin.ublas_dense_matrix_swigregister
ublas_dense_matrix_swigregister(ublas_dense_matrix)

class ublas_sparse_matrix(_object):
    """Proxy of C++ ublas_sparse_matrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ublas_sparse_matrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ublas_sparse_matrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
ublas_sparse_matrix_swigregister = _dolfin.ublas_sparse_matrix_swigregister
ublas_sparse_matrix_swigregister(ublas_sparse_matrix)

class GenericMatrix(_object):
    """Proxy of C++ GenericMatrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GenericMatrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GenericMatrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_GenericMatrix
    __del__ = lambda self : None;
    def init(*args):
        """
        init(self, uint M, uint N)
        init(self, uint M, uint N, uint nzmax)
        """
        return _dolfin.GenericMatrix_init(*args)

    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.GenericMatrix_size(*args)

    def get(*args):
        """get(self, uint i, uint j) -> real"""
        return _dolfin.GenericMatrix_get(*args)

    def set(*args):
        """
        set(self, uint i, uint j, real value)
        set(self, real block, int rows, int m, int cols, int n)
        """
        return _dolfin.GenericMatrix_set(*args)

    def add(*args):
        """add(self, real block, int rows, int m, int cols, int n)"""
        return _dolfin.GenericMatrix_add(*args)

    def apply(*args):
        """apply(self)"""
        return _dolfin.GenericMatrix_apply(*args)

    def zero(*args):
        """zero(self)"""
        return _dolfin.GenericMatrix_zero(*args)

    def ident(*args):
        """ident(self, int rows, int m)"""
        return _dolfin.GenericMatrix_ident(*args)

    def nzmax(*args):
        """nzmax(self) -> uint"""
        return _dolfin.GenericMatrix_nzmax(*args)

GenericMatrix_swigregister = _dolfin.GenericMatrix_swigregister
GenericMatrix_swigregister(GenericMatrix)

class GenericVector(_object):
    """Proxy of C++ GenericVector class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GenericVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GenericVector, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_GenericVector
    __del__ = lambda self : None;
    def init(*args):
        """init(self, uint N)"""
        return _dolfin.GenericVector_init(*args)

    def size(*args):
        """size(self) -> uint"""
        return _dolfin.GenericVector_size(*args)

    def get(*args):
        """get(self, uint i) -> real"""
        return _dolfin.GenericVector_get(*args)

    def set(*args):
        """
        set(self, uint i, real value)
        set(self, real block, int pos, int n)
        """
        return _dolfin.GenericVector_set(*args)

    def add(*args):
        """add(self, real block, int pos, int n)"""
        return _dolfin.GenericVector_add(*args)

    def apply(*args):
        """apply(self)"""
        return _dolfin.GenericVector_apply(*args)

    def zero(*args):
        """zero(self)"""
        return _dolfin.GenericVector_zero(*args)

GenericVector_swigregister = _dolfin.GenericVector_swigregister
GenericVector_swigregister(GenericVector)

class uBlasVector(GenericVector,Variable):
    """Proxy of C++ uBlasVector class"""
    __swig_setmethods__ = {}
    for _s in [GenericVector,Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasVector, name, value)
    __swig_getmethods__ = {}
    for _s in [GenericVector,Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasVector, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> uBlasVector
        __init__(self, uint N) -> uBlasVector
        """
        this = _dolfin.new_uBlasVector(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasVector
    __del__ = lambda self : None;
    def init(*args):
        """init(self, uint N)"""
        return _dolfin.uBlasVector_init(*args)

    def size(*args):
        """size(self) -> uint"""
        return _dolfin.uBlasVector_size(*args)

    def __call__(*args):
        """
        __call__(self, uint i) -> real
        __call__(self, uint i) -> real
        """
        return _dolfin.uBlasVector___call__(*args)

    def set(*args):
        """
        set(self, uint i, real value)
        set(self, real block, int pos, int n)
        """
        return _dolfin.uBlasVector_set(*args)

    def add(*args):
        """add(self, real block, int pos, int n)"""
        return _dolfin.uBlasVector_add(*args)

    def get(*args):
        """
        get(self, uint i) -> real
        get(self, real block, int pos, int n)
        """
        return _dolfin.uBlasVector_get(*args)

    l1 = _dolfin.uBlasVector_l1
    l2 = _dolfin.uBlasVector_l2
    linf = _dolfin.uBlasVector_linf
    def norm(*args):
        """
        norm(self, NormType type=l2) -> real
        norm(self) -> real
        """
        return _dolfin.uBlasVector_norm(*args)

    def sum(*args):
        """sum(self) -> real"""
        return _dolfin.uBlasVector_sum(*args)

    def apply(*args):
        """apply(self)"""
        return _dolfin.uBlasVector_apply(*args)

    def zero(*args):
        """zero(self)"""
        return _dolfin.uBlasVector_zero(*args)

    def disp(*args):
        """
        disp(self, uint precision=2)
        disp(self)
        """
        return _dolfin.uBlasVector_disp(*args)

    def copy(*args):
        """
        copy(self, real a) -> uBlasVector
        copy(self, PETScVector y)
        copy(self, uBlasVector y)
        """
        return _dolfin.uBlasVector_copy(*args)

uBlasVector_swigregister = _dolfin.uBlasVector_swigregister
uBlasVector_swigregister(uBlasVector)

class GMRES(_object):
    """Proxy of C++ GMRES class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GMRES, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GMRES, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def solve(*args):
        """
        solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
            uBlasVector b, Preconditioner pc=default_pc) -> uint
        solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
            uBlasVector b) -> uint
        solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
            uBlasVector b, Preconditioner pc=default_pc) -> uint
        solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
            uBlasVector b) -> uint
        solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b, 
            Preconditioner pc=default_pc) -> uint
        solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
            uBlasVector b, uBlasPreconditioner pc) -> uint
        solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
            uBlasVector b, uBlasPreconditioner pc) -> uint
        solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b, 
            uBlasPreconditioner pc) -> uint
        """
        return _dolfin.GMRES_solve(*args)

    if _newclass:solve = staticmethod(solve)
    __swig_getmethods__["solve"] = lambda x: solve
GMRES_swigregister = _dolfin.GMRES_swigregister
GMRES_swigregister(GMRES)

def GMRES_solve(*args):
  """
    solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
        uBlasVector b, Preconditioner pc=default_pc) -> uint
    solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
        uBlasVector b) -> uint
    solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
        uBlasVector b, Preconditioner pc=default_pc) -> uint
    solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
        uBlasVector b) -> uint
    solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b, 
        Preconditioner pc=default_pc) -> uint
    solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b) -> uint
    solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
        uBlasVector b, uBlasPreconditioner pc) -> uint
    solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
        uBlasVector b, uBlasPreconditioner pc) -> uint
    GMRES_solve(uBlasKrylovMatrix A, uBlasVector x, uBlasVector b, 
        uBlasPreconditioner pc) -> uint
    """
  return _dolfin.GMRES_solve(*args)

class LU(_object):
    """Proxy of C++ LU class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, LU, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, LU, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def solve(*args):
        """
        solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
            uBlasVector b)
        solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
            uBlasVector b)
        """
        return _dolfin.LU_solve(*args)

    if _newclass:solve = staticmethod(solve)
    __swig_getmethods__["solve"] = lambda x: solve
LU_swigregister = _dolfin.LU_swigregister
LU_swigregister(LU)

def LU_solve(*args):
  """
    solve(uBlasMatrix<(dolfin::ublas_dense_matrix)> A, uBlasVector x, 
        uBlasVector b)
    LU_solve(uBlasMatrix<(dolfin::ublas_sparse_matrix)> A, uBlasVector x, 
        uBlasVector b)
    """
  return _dolfin.LU_solve(*args)

class uBlasDummyPreconditioner(_object):
    """Proxy of C++ uBlasDummyPreconditioner class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasDummyPreconditioner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasDummyPreconditioner, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> uBlasDummyPreconditioner"""
        this = _dolfin.new_uBlasDummyPreconditioner(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasDummyPreconditioner
    __del__ = lambda self : None;
    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.uBlasDummyPreconditioner_solve(*args)

uBlasDummyPreconditioner_swigregister = _dolfin.uBlasDummyPreconditioner_swigregister
uBlasDummyPreconditioner_swigregister(uBlasDummyPreconditioner)

class uBlasKrylovMatrix(_object):
    """Proxy of C++ uBlasKrylovMatrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasKrylovMatrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasKrylovMatrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_uBlasKrylovMatrix
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.uBlasKrylovMatrix_size(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.uBlasKrylovMatrix_mult(*args)

uBlasKrylovMatrix_swigregister = _dolfin.uBlasKrylovMatrix_swigregister
uBlasKrylovMatrix_swigregister(uBlasKrylovMatrix)

class uBlasKrylovSolver(_object):
    """Proxy of C++ uBlasKrylovSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasKrylovSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasKrylovSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, KrylovMethod method=default_method, Preconditioner pc=default_pc) -> uBlasKrylovSolver
        __init__(self, KrylovMethod method=default_method) -> uBlasKrylovSolver
        __init__(self) -> uBlasKrylovSolver
        __init__(self, Preconditioner pc) -> uBlasKrylovSolver
        __init__(self, uBlasPreconditioner pc) -> uBlasKrylovSolver
        __init__(self, KrylovMethod method, uBlasPreconditioner preconditioner) -> uBlasKrylovSolver
        """
        this = _dolfin.new_uBlasKrylovSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasKrylovSolver
    __del__ = lambda self : None;
    def solve(*args):
        """
        solve(self, uBlasDenseMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(self, uBlasSparseMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(self, uBlasKrylovMatrix A, uBlasVector x, uBlasVector b) -> uint
        """
        return _dolfin.uBlasKrylovSolver_solve(*args)

uBlasKrylovSolver_swigregister = _dolfin.uBlasKrylovSolver_swigregister
uBlasKrylovSolver_swigregister(uBlasKrylovSolver)

class uBlasLinearSolver(_object):
    """Proxy of C++ uBlasLinearSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasLinearSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasLinearSolver, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_uBlasLinearSolver
    __del__ = lambda self : None;
    def solve(*args):
        """
        solve(self, uBlasDenseMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(self, uBlasSparseMatrix A, uBlasVector x, uBlasVector b) -> uint
        """
        return _dolfin.uBlasLinearSolver_solve(*args)

uBlasLinearSolver_swigregister = _dolfin.uBlasLinearSolver_swigregister
uBlasLinearSolver_swigregister(uBlasLinearSolver)

class uBlasLUSolver(uBlasLinearSolver):
    """Proxy of C++ uBlasLUSolver class"""
    __swig_setmethods__ = {}
    for _s in [uBlasLinearSolver]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasLUSolver, name, value)
    __swig_getmethods__ = {}
    for _s in [uBlasLinearSolver]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasLUSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> uBlasLUSolver"""
        this = _dolfin.new_uBlasLUSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasLUSolver
    __del__ = lambda self : None;
    def solve(*args):
        """
        solve(self, uBlasDenseMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(self, uBlasSparseMatrix A, uBlasVector x, uBlasVector b) -> uint
        solve(self, uBlasKrylovMatrix A, uBlasVector x, uBlasVector b)
        """
        return _dolfin.uBlasLUSolver_solve(*args)

    def solveInPlaceUBlas(*args):
        """solveInPlaceUBlas(self, uBlasDenseMatrix A, uBlasVector x, uBlasVector b) -> uint"""
        return _dolfin.uBlasLUSolver_solveInPlaceUBlas(*args)

uBlasLUSolver_swigregister = _dolfin.uBlasLUSolver_swigregister
uBlasLUSolver_swigregister(uBlasLUSolver)

class uBlasILUPreconditioner(_object):
    """Proxy of C++ uBlasILUPreconditioner class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasILUPreconditioner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasILUPreconditioner, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> uBlasILUPreconditioner
        __init__(self, uBlasSparseMatrix A) -> uBlasILUPreconditioner
        """
        this = _dolfin.new_uBlasILUPreconditioner(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasILUPreconditioner
    __del__ = lambda self : None;
    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.uBlasILUPreconditioner_solve(*args)

uBlasILUPreconditioner_swigregister = _dolfin.uBlasILUPreconditioner_swigregister
uBlasILUPreconditioner_swigregister(uBlasILUPreconditioner)

class uBlasPreconditioner(_object):
    """Proxy of C++ uBlasPreconditioner class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasPreconditioner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasPreconditioner, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_uBlasPreconditioner
    __del__ = lambda self : None;
    def init(*args):
        """
        init(self, uBlasDenseMatrix A)
        init(self, uBlasSparseMatrix A)
        init(self, uBlasKrylovMatrix A)
        """
        return _dolfin.uBlasPreconditioner_init(*args)

    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.uBlasPreconditioner_solve(*args)

uBlasPreconditioner_swigregister = _dolfin.uBlasPreconditioner_swigregister
uBlasPreconditioner_swigregister(uBlasPreconditioner)

class uBlasSparseMatrix(Variable,GenericMatrix,ublas_sparse_matrix):
    """Proxy of C++ uBlasSparseMatrix class"""
    __swig_setmethods__ = {}
    for _s in [Variable,GenericMatrix,ublas_sparse_matrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasSparseMatrix, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable,GenericMatrix,ublas_sparse_matrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasSparseMatrix, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> uBlasSparseMatrix
        __init__(self, uint M, uint N) -> uBlasSparseMatrix
        """
        this = _dolfin.new_uBlasSparseMatrix(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasSparseMatrix
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.uBlasSparseMatrix_size(*args)

    def get(*args):
        """get(self, uint i, uint j) -> real"""
        return _dolfin.uBlasSparseMatrix_get(*args)

    def getRow(*args):
        """getRow(self, uint i, int ncols, dolfin::Array<(int)> columns, dolfin::Array<(dolfin::real)> values)"""
        return _dolfin.uBlasSparseMatrix_getRow(*args)

    def lump(*args):
        """lump(self, uBlasVector m)"""
        return _dolfin.uBlasSparseMatrix_lump(*args)

    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.uBlasSparseMatrix_solve(*args)

    def invert(*args):
        """invert(self)"""
        return _dolfin.uBlasSparseMatrix_invert(*args)

    def apply(*args):
        """apply(self)"""
        return _dolfin.uBlasSparseMatrix_apply(*args)

    def zero(*args):
        """zero(self)"""
        return _dolfin.uBlasSparseMatrix_zero(*args)

    def ident(*args):
        """ident(self, int rows, int m)"""
        return _dolfin.uBlasSparseMatrix_ident(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.uBlasSparseMatrix_mult(*args)

    def disp(*args):
        """
        disp(self, uint precision=2)
        disp(self)
        """
        return _dolfin.uBlasSparseMatrix_disp(*args)

    def init(*args):
        """
        init(self, uint M, uint N)
        init(self, uint M, uint N, uint nzmax)
        """
        return _dolfin.uBlasSparseMatrix_init(*args)

    def set(*args):
        """
        set(self, uint i, uint j, real value)
        set(self, real block, int rows, int m, int cols, int n)
        """
        return _dolfin.uBlasSparseMatrix_set(*args)

    def add(*args):
        """add(self, real block, int rows, int m, int cols, int n)"""
        return _dolfin.uBlasSparseMatrix_add(*args)

    def nzmax(*args):
        """nzmax(self) -> uint"""
        return _dolfin.uBlasSparseMatrix_nzmax(*args)

uBlasSparseMatrix_swigregister = _dolfin.uBlasSparseMatrix_swigregister
uBlasSparseMatrix_swigregister(uBlasSparseMatrix)

class uBlasDenseMatrix(Variable,GenericMatrix,ublas_dense_matrix):
    """Proxy of C++ uBlasDenseMatrix class"""
    __swig_setmethods__ = {}
    for _s in [Variable,GenericMatrix,ublas_dense_matrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, uBlasDenseMatrix, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable,GenericMatrix,ublas_dense_matrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, uBlasDenseMatrix, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> uBlasDenseMatrix
        __init__(self, uint M, uint N) -> uBlasDenseMatrix
        """
        this = _dolfin.new_uBlasDenseMatrix(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_uBlasDenseMatrix
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.uBlasDenseMatrix_size(*args)

    def get(*args):
        """get(self, uint i, uint j) -> real"""
        return _dolfin.uBlasDenseMatrix_get(*args)

    def getRow(*args):
        """getRow(self, uint i, int ncols, dolfin::Array<(int)> columns, dolfin::Array<(dolfin::real)> values)"""
        return _dolfin.uBlasDenseMatrix_getRow(*args)

    def lump(*args):
        """lump(self, uBlasVector m)"""
        return _dolfin.uBlasDenseMatrix_lump(*args)

    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.uBlasDenseMatrix_solve(*args)

    def invert(*args):
        """invert(self)"""
        return _dolfin.uBlasDenseMatrix_invert(*args)

    def apply(*args):
        """apply(self)"""
        return _dolfin.uBlasDenseMatrix_apply(*args)

    def zero(*args):
        """zero(self)"""
        return _dolfin.uBlasDenseMatrix_zero(*args)

    def ident(*args):
        """ident(self, int rows, int m)"""
        return _dolfin.uBlasDenseMatrix_ident(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.uBlasDenseMatrix_mult(*args)

    def disp(*args):
        """
        disp(self, uint precision=2)
        disp(self)
        """
        return _dolfin.uBlasDenseMatrix_disp(*args)

    def init(*args):
        """
        init(self, uint M, uint N)
        init(self, uint M, uint N, uint nzmax)
        """
        return _dolfin.uBlasDenseMatrix_init(*args)

    def set(*args):
        """
        set(self, uint i, uint j, real value)
        set(self, real block, int rows, int m, int cols, int n)
        """
        return _dolfin.uBlasDenseMatrix_set(*args)

    def add(*args):
        """add(self, real block, int rows, int m, int cols, int n)"""
        return _dolfin.uBlasDenseMatrix_add(*args)

    def nzmax(*args):
        """nzmax(self) -> uint"""
        return _dolfin.uBlasDenseMatrix_nzmax(*args)

uBlasDenseMatrix_swigregister = _dolfin.uBlasDenseMatrix_swigregister
uBlasDenseMatrix_swigregister(uBlasDenseMatrix)

# Explicit typedefs
Vector = uBlasVector
Matrix = uBlasSparseMatrix
KrylovSolver = uBlasKrylovSolver
LUSolver = uBlasLUSolver

# Explicit typedefs
DenseVector = uBlasVector

def __getitem__(self, i):
    return self.get(i)
def __setitem__(self, i, val):
    self.set(i, val)

GenericVector.__getitem__ = __getitem__
GenericVector.__setitem__ = __setitem__

def __getitem__(self, i):
    return self.get(i[0], i[1])
def __setitem__(self, i, val):
    self.set(i[0], i[1], val)

GenericMatrix.__getitem__ = __getitem__
GenericMatrix.__setitem__ = __setitem__

class Function(Variable,TimeDependent):
    """Proxy of C++ Function class"""
    __swig_setmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Function, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Function, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, real value) -> Function
        __init__(self, uint vectordim=1) -> Function
        __init__(self) -> Function
        __init__(self, FunctionPointer fp, uint vectordim=1) -> Function
        __init__(self, FunctionPointer fp) -> Function
        __init__(self, Vector x) -> Function
        __init__(self, Vector x, Mesh mesh) -> Function
        __init__(self, Vector x, Mesh mesh, FiniteElement element) -> Function
        __init__(self, Mesh mesh, FiniteElement element) -> Function
        __init__(self, Function f) -> Function
        """
        if self.__class__ == Function:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_Function(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Function
    __del__ = lambda self : None;
    def eval(*args):
        """
        eval(self, Point p, uint i=0) -> real
        eval(self, Point p) -> real
        """
        return _dolfin.Function_eval(*args)

    def __call__(*args):
        """
        __call__(self, Point p, uint i=0) -> real
        __call__(self, Point p) -> real
        __call__(self, Vertex vertex, uint i=0) -> real
        __call__(self, Vertex vertex) -> real
        """
        return _dolfin.Function___call__(*args)

    def __getitem__(*args):
        """__getitem__(self, uint i) -> Function"""
        return _dolfin.Function___getitem__(*args)

    def interpolate(*args):
        """interpolate(self, real coefficients, AffineMap map, FiniteElement element)"""
        return _dolfin.Function_interpolate(*args)

    def vectordim(*args):
        """vectordim(self) -> uint"""
        return _dolfin.Function_vectordim(*args)

    def vector(*args):
        """vector(self) -> Vector"""
        return _dolfin.Function_vector(*args)

    def mesh(*args):
        """mesh(self) -> Mesh"""
        return _dolfin.Function_mesh(*args)

    def element(*args):
        """element(self) -> FiniteElement"""
        return _dolfin.Function_element(*args)

    def attach(*args):
        """
        attach(self, Vector x, bool local=False)
        attach(self, Vector x)
        attach(self, Mesh mesh, bool local=False)
        attach(self, Mesh mesh)
        attach(self, FiniteElement element, bool local=False)
        attach(self, FiniteElement element)
        """
        return _dolfin.Function_attach(*args)

    def init(*args):
        """init(self, Mesh mesh, FiniteElement element)"""
        return _dolfin.Function_init(*args)

    constant = _dolfin.Function_constant
    user = _dolfin.Function_user
    functionpointer = _dolfin.Function_functionpointer
    discrete = _dolfin.Function_discrete
    def type(*args):
        """type(self) -> int"""
        return _dolfin.Function_type(*args)

    def __disown__(self):
        self.this.disown()
        _dolfin.disown_Function(self)
        return weakref_proxy(self)
Function_swigregister = _dolfin.Function_swigregister
Function_swigregister(Function)

class Form(_object):
    """Proxy of C++ Form class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Form, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Form, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint num_functions) -> Form"""
        this = _dolfin.new_Form(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Form
    __del__ = lambda self : None;
    def update(*args):
        """update(self, AffineMap map)"""
        return _dolfin.Form_update(*args)

    def function(*args):
        """function(self, uint i) -> Function"""
        return _dolfin.Form_function(*args)

    def element(*args):
        """element(self, uint i) -> FiniteElement"""
        return _dolfin.Form_element(*args)

    __swig_setmethods__["num_functions"] = _dolfin.Form_num_functions_set
    __swig_getmethods__["num_functions"] = _dolfin.Form_num_functions_get
    if _newclass:num_functions = property(_dolfin.Form_num_functions_get, _dolfin.Form_num_functions_set)
Form_swigregister = _dolfin.Form_swigregister
Form_swigregister(Form)

class BilinearForm(Form):
    """Proxy of C++ BilinearForm class"""
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BilinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BilinearForm, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_BilinearForm
    __del__ = lambda self : None;
    def eval(*args):
        """
        eval(self, real block, AffineMap map)
        eval(self, real block, AffineMap map, uint segment)
        """
        return _dolfin.BilinearForm_eval(*args)

    def test(*args):
        """
        test(self) -> FiniteElement
        test(self) -> FiniteElement
        """
        return _dolfin.BilinearForm_test(*args)

    def trial(*args):
        """
        trial(self) -> FiniteElement
        trial(self) -> FiniteElement
        """
        return _dolfin.BilinearForm_trial(*args)

BilinearForm_swigregister = _dolfin.BilinearForm_swigregister
BilinearForm_swigregister(BilinearForm)

class LinearForm(Form):
    """Proxy of C++ LinearForm class"""
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, LinearForm, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_LinearForm
    __del__ = lambda self : None;
    def eval(*args):
        """
        eval(self, real block, AffineMap map)
        eval(self, real block, AffineMap map, uint segment)
        """
        return _dolfin.LinearForm_eval(*args)

    def test(*args):
        """
        test(self) -> FiniteElement
        test(self) -> FiniteElement
        """
        return _dolfin.LinearForm_test(*args)

LinearForm_swigregister = _dolfin.LinearForm_swigregister
LinearForm_swigregister(LinearForm)

class Mesh(Variable):
    """Proxy of C++ Mesh class"""
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Mesh, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Mesh, name)
    __repr__ = _swig_repr
    triangles = _dolfin.Mesh_triangles
    tetrahedra = _dolfin.Mesh_tetrahedra
    def __init__(self, *args): 
        """
        __init__(self) -> Mesh
        __init__(self, char filename) -> Mesh
        __init__(self, Mesh mesh) -> Mesh
        """
        this = _dolfin.new_Mesh(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Mesh
    __del__ = lambda self : None;
    def merge(*args):
        """merge(self, Mesh mesh2)"""
        return _dolfin.Mesh_merge(*args)

    def init(*args):
        """init(self)"""
        return _dolfin.Mesh_init(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.Mesh_clear(*args)

    def numSpaceDim(*args):
        """numSpaceDim(self) -> int"""
        return _dolfin.Mesh_numSpaceDim(*args)

    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.Mesh_numVertices(*args)

    def numCells(*args):
        """numCells(self) -> int"""
        return _dolfin.Mesh_numCells(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.Mesh_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.Mesh_numFaces(*args)

    def createVertex(*args):
        """
        createVertex(self, Point p) -> Vertex
        createVertex(self, real x, real y, real z) -> Vertex
        """
        return _dolfin.Mesh_createVertex(*args)

    def createCell(*args):
        """
        createCell(self, int n0, int n1, int n2) -> Cell
        createCell(self, int n0, int n1, int n2, int n3) -> Cell
        createCell(self, Vertex n0, Vertex n1, Vertex n2) -> Cell
        createCell(self, Vertex n0, Vertex n1, Vertex n2, Vertex n3) -> Cell
        """
        return _dolfin.Mesh_createCell(*args)

    def createEdge(*args):
        """
        createEdge(self, int n0, int n1) -> Edge
        createEdge(self, Vertex n0, Vertex n1) -> Edge
        """
        return _dolfin.Mesh_createEdge(*args)

    def createFace(*args):
        """
        createFace(self, int e0, int e1, int e2) -> Face
        createFace(self, Edge e0, Edge e1, Edge e2) -> Face
        """
        return _dolfin.Mesh_createFace(*args)

    def remove(*args):
        """
        remove(self, Vertex vertex)
        remove(self, Cell cell)
        remove(self, Edge edge)
        remove(self, Face face)
        """
        return _dolfin.Mesh_remove(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.Mesh_type(*args)

    def vertex(*args):
        """vertex(self, uint id) -> Vertex"""
        return _dolfin.Mesh_vertex(*args)

    def cell(*args):
        """cell(self, uint id) -> Cell"""
        return _dolfin.Mesh_cell(*args)

    def edge(*args):
        """edge(self, uint id) -> Edge"""
        return _dolfin.Mesh_edge(*args)

    def face(*args):
        """face(self, uint id) -> Face"""
        return _dolfin.Mesh_face(*args)

    def boundary(*args):
        """boundary(self) -> Boundary"""
        return _dolfin.Mesh_boundary(*args)

    def refine(*args):
        """refine(self)"""
        return _dolfin.Mesh_refine(*args)

    def refineUniformly(*args):
        """
        refineUniformly(self)
        refineUniformly(self, int i)
        """
        return _dolfin.Mesh_refineUniformly(*args)

    def parent(*args):
        """parent(self) -> Mesh"""
        return _dolfin.Mesh_parent(*args)

    def child(*args):
        """child(self) -> Mesh"""
        return _dolfin.Mesh_child(*args)

    def __eq__(*args):
        """__eq__(self, Mesh mesh) -> bool"""
        return _dolfin.Mesh___eq__(*args)

    def __ne__(*args):
        """__ne__(self, Mesh mesh) -> bool"""
        return _dolfin.Mesh___ne__(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.Mesh_disp(*args)

Mesh_swigregister = _dolfin.Mesh_swigregister
Mesh_swigregister(Mesh)

class Boundary(_object):
    """Proxy of C++ Boundary class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Boundary, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Boundary, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Boundary
        __init__(self, Mesh mesh) -> Boundary
        """
        this = _dolfin.new_Boundary(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Boundary
    __del__ = lambda self : None;
    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.Boundary_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.Boundary_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.Boundary_numFaces(*args)

    def numFacets(*args):
        """numFacets(self) -> int"""
        return _dolfin.Boundary_numFacets(*args)

Boundary_swigregister = _dolfin.Boundary_swigregister
Boundary_swigregister(Boundary)

class Point(_object):
    """Proxy of C++ Point class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Point, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Point, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Point
        __init__(self, real x) -> Point
        __init__(self, real x, real y) -> Point
        __init__(self, real x, real y, real z) -> Point
        __init__(self, Point p) -> Point
        """
        this = _dolfin.new_Point(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Point
    __del__ = lambda self : None;
    def dist(*args):
        """
        dist(self, Point p) -> real
        dist(self, real x, real y=0.0, real z=0.0) -> real
        dist(self, real x, real y=0.0) -> real
        dist(self, real x) -> real
        """
        return _dolfin.Point_dist(*args)

    def norm(*args):
        """norm(self) -> real"""
        return _dolfin.Point_norm(*args)

    def midpoint(*args):
        """midpoint(self, Point p) -> Point"""
        return _dolfin.Point_midpoint(*args)

    def __add__(*args):
        """__add__(self, Point p) -> Point"""
        return _dolfin.Point___add__(*args)

    def __sub__(*args):
        """__sub__(self, Point p) -> Point"""
        return _dolfin.Point___sub__(*args)

    def __mul__(*args):
        """__mul__(self, Point p) -> real"""
        return _dolfin.Point___mul__(*args)

    def __iadd__(*args):
        """__iadd__(self, Point p) -> Point"""
        return _dolfin.Point___iadd__(*args)

    def __isub__(*args):
        """__isub__(self, Point p) -> Point"""
        return _dolfin.Point___isub__(*args)

    def __imul__(*args):
        """__imul__(self, real a) -> Point"""
        return _dolfin.Point___imul__(*args)

    def __idiv__(*args):
        """__idiv__(self, real a) -> Point"""
        return _dolfin.Point___idiv__(*args)

    def cross(*args):
        """cross(self, Point p) -> Point"""
        return _dolfin.Point_cross(*args)

    __swig_setmethods__["x"] = _dolfin.Point_x_set
    __swig_getmethods__["x"] = _dolfin.Point_x_get
    if _newclass:x = property(_dolfin.Point_x_get, _dolfin.Point_x_set)
    __swig_setmethods__["y"] = _dolfin.Point_y_set
    __swig_getmethods__["y"] = _dolfin.Point_y_get
    if _newclass:y = property(_dolfin.Point_y_get, _dolfin.Point_y_set)
    __swig_setmethods__["z"] = _dolfin.Point_z_set
    __swig_getmethods__["z"] = _dolfin.Point_z_get
    if _newclass:z = property(_dolfin.Point_z_get, _dolfin.Point_z_set)
Point_swigregister = _dolfin.Point_swigregister
Point_swigregister(Point)

class Vertex(_object):
    """Proxy of C++ Vertex class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vertex, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vertex, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Vertex
        __init__(self, real x) -> Vertex
        __init__(self, real x, real y) -> Vertex
        __init__(self, real x, real y, real z) -> Vertex
        """
        this = _dolfin.new_Vertex(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Vertex
    __del__ = lambda self : None;
    def clear(*args):
        """clear(self)"""
        return _dolfin.Vertex_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.Vertex_id(*args)

    def numVertexNeighbors(*args):
        """numVertexNeighbors(self) -> int"""
        return _dolfin.Vertex_numVertexNeighbors(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> int"""
        return _dolfin.Vertex_numCellNeighbors(*args)

    def numEdgeNeighbors(*args):
        """numEdgeNeighbors(self) -> int"""
        return _dolfin.Vertex_numEdgeNeighbors(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.Vertex_vertex(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.Vertex_cell(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.Vertex_edge(*args)

    def parent(*args):
        """parent(self) -> Vertex"""
        return _dolfin.Vertex_parent(*args)

    def child(*args):
        """child(self) -> Vertex"""
        return _dolfin.Vertex_child(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.Vertex_mesh(*args)

    def coord(*args):
        """
        coord(self) -> Point
        coord(self) -> Point
        """
        return _dolfin.Vertex_coord(*args)

    def midpoint(*args):
        """midpoint(self, Vertex n) -> Point"""
        return _dolfin.Vertex_midpoint(*args)

    def dist(*args):
        """
        dist(self, Vertex n) -> real
        dist(self, Point p) -> real
        dist(self, real x, real y=0.0, real z=0.0) -> real
        dist(self, real x, real y=0.0) -> real
        dist(self, real x) -> real
        """
        return _dolfin.Vertex_dist(*args)

    def neighbor(*args):
        """neighbor(self, Vertex n) -> bool"""
        return _dolfin.Vertex_neighbor(*args)

    def __ne__(*args):
        """__ne__(self, Vertex vertex) -> bool"""
        return _dolfin.Vertex___ne__(*args)

    def __eq__(*args):
        """
        __eq__(self, Vertex vertex) -> bool
        __eq__(self, int id) -> bool
        """
        return _dolfin.Vertex___eq__(*args)

    def __lt__(*args):
        """__lt__(self, int id) -> bool"""
        return _dolfin.Vertex___lt__(*args)

    def __le__(*args):
        """__le__(self, int id) -> bool"""
        return _dolfin.Vertex___le__(*args)

    def __gt__(*args):
        """__gt__(self, int id) -> bool"""
        return _dolfin.Vertex___gt__(*args)

    def __ge__(*args):
        """__ge__(self, int id) -> bool"""
        return _dolfin.Vertex___ge__(*args)

    __swig_setmethods__["nbids"] = _dolfin.Vertex_nbids_set
    __swig_getmethods__["nbids"] = _dolfin.Vertex_nbids_get
    if _newclass:nbids = property(_dolfin.Vertex_nbids_get, _dolfin.Vertex_nbids_set)
Vertex_swigregister = _dolfin.Vertex_swigregister
Vertex_swigregister(Vertex)

class Edge(_object):
    """Proxy of C++ Edge class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Edge, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Edge, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Edge
        __init__(self, Vertex n0, Vertex n1) -> Edge
        """
        this = _dolfin.new_Edge(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Edge
    __del__ = lambda self : None;
    def clear(*args):
        """clear(self)"""
        return _dolfin.Edge_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.Edge_id(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> unsigned int"""
        return _dolfin.Edge_numCellNeighbors(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.Edge_vertex(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.Edge_cell(*args)

    def localID(*args):
        """localID(self, int i) -> int"""
        return _dolfin.Edge_localID(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.Edge_mesh(*args)

    def coord(*args):
        """coord(self, int i) -> Point"""
        return _dolfin.Edge_coord(*args)

    def length(*args):
        """length(self) -> real"""
        return _dolfin.Edge_length(*args)

    def midpoint(*args):
        """midpoint(self) -> Point"""
        return _dolfin.Edge_midpoint(*args)

    def equals(*args):
        """equals(self, Vertex n0, Vertex n1) -> bool"""
        return _dolfin.Edge_equals(*args)

    def contains(*args):
        """
        contains(self, Vertex n) -> bool
        contains(self, Point point) -> bool
        """
        return _dolfin.Edge_contains(*args)

    __swig_setmethods__["ebids"] = _dolfin.Edge_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.Edge_ebids_get
    if _newclass:ebids = property(_dolfin.Edge_ebids_get, _dolfin.Edge_ebids_set)
Edge_swigregister = _dolfin.Edge_swigregister
Edge_swigregister(Edge)

class Triangle(_object):
    """Proxy of C++ Triangle class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Triangle, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Triangle, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, Vertex n0, Vertex n1, Vertex n2) -> Triangle"""
        this = _dolfin.new_Triangle(*args)
        try: self.this.append(this)
        except: self.this = this
    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.Triangle_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.Triangle_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.Triangle_numFaces(*args)

    def numBoundaries(*args):
        """numBoundaries(self) -> int"""
        return _dolfin.Triangle_numBoundaries(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.Triangle_type(*args)

    def orientation(*args):
        """orientation(self) -> int"""
        return _dolfin.Triangle_orientation(*args)

    def volume(*args):
        """volume(self) -> real"""
        return _dolfin.Triangle_volume(*args)

    def diameter(*args):
        """diameter(self) -> real"""
        return _dolfin.Triangle_diameter(*args)

    def edgeAlignment(*args):
        """edgeAlignment(self, uint i) -> uint"""
        return _dolfin.Triangle_edgeAlignment(*args)

    def faceAlignment(*args):
        """faceAlignment(self, uint i) -> uint"""
        return _dolfin.Triangle_faceAlignment(*args)

Triangle_swigregister = _dolfin.Triangle_swigregister
Triangle_swigregister(Triangle)

class Tetrahedron(_object):
    """Proxy of C++ Tetrahedron class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Tetrahedron, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Tetrahedron, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, Vertex n0, Vertex n1, Vertex n2, Vertex n3) -> Tetrahedron"""
        this = _dolfin.new_Tetrahedron(*args)
        try: self.this.append(this)
        except: self.this = this
    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.Tetrahedron_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.Tetrahedron_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.Tetrahedron_numFaces(*args)

    def numBoundaries(*args):
        """numBoundaries(self) -> int"""
        return _dolfin.Tetrahedron_numBoundaries(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.Tetrahedron_type(*args)

    def orientation(*args):
        """orientation(self) -> int"""
        return _dolfin.Tetrahedron_orientation(*args)

    def volume(*args):
        """volume(self) -> real"""
        return _dolfin.Tetrahedron_volume(*args)

    def diameter(*args):
        """diameter(self) -> real"""
        return _dolfin.Tetrahedron_diameter(*args)

    def edgeAlignment(*args):
        """edgeAlignment(self, uint i) -> uint"""
        return _dolfin.Tetrahedron_edgeAlignment(*args)

    def faceAlignment(*args):
        """faceAlignment(self, uint i) -> uint"""
        return _dolfin.Tetrahedron_faceAlignment(*args)

Tetrahedron_swigregister = _dolfin.Tetrahedron_swigregister
Tetrahedron_swigregister(Tetrahedron)

class Cell(_object):
    """Proxy of C++ Cell class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Cell, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Cell, name)
    __repr__ = _swig_repr
    triangle = _dolfin.Cell_triangle
    tetrahedron = _dolfin.Cell_tetrahedron
    none = _dolfin.Cell_none
    left = _dolfin.Cell_left
    right = _dolfin.Cell_right
    def __init__(self, *args): 
        """
        __init__(self) -> Cell
        __init__(self, Vertex n0, Vertex n1, Vertex n2) -> Cell
        __init__(self, Vertex n0, Vertex n1, Vertex n2, Vertex n3) -> Cell
        """
        this = _dolfin.new_Cell(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Cell
    __del__ = lambda self : None;
    def clear(*args):
        """clear(self)"""
        return _dolfin.Cell_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.Cell_id(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.Cell_type(*args)

    def orientation(*args):
        """orientation(self) -> int"""
        return _dolfin.Cell_orientation(*args)

    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.Cell_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.Cell_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.Cell_numFaces(*args)

    def numBoundaries(*args):
        """numBoundaries(self) -> int"""
        return _dolfin.Cell_numBoundaries(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> int"""
        return _dolfin.Cell_numCellNeighbors(*args)

    def numVertexNeighbors(*args):
        """numVertexNeighbors(self) -> int"""
        return _dolfin.Cell_numVertexNeighbors(*args)

    def numChildren(*args):
        """numChildren(self) -> int"""
        return _dolfin.Cell_numChildren(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.Cell_vertex(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.Cell_edge(*args)

    def face(*args):
        """face(self, int i) -> Face"""
        return _dolfin.Cell_face(*args)

    def neighbor(*args):
        """neighbor(self, int i) -> Cell"""
        return _dolfin.Cell_neighbor(*args)

    def parent(*args):
        """parent(self) -> Cell"""
        return _dolfin.Cell_parent(*args)

    def child(*args):
        """child(self, int i) -> Cell"""
        return _dolfin.Cell_child(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.Cell_mesh(*args)

    def coord(*args):
        """coord(self, int i) -> Point"""
        return _dolfin.Cell_coord(*args)

    def midpoint(*args):
        """midpoint(self) -> Point"""
        return _dolfin.Cell_midpoint(*args)

    def vertexID(*args):
        """vertexID(self, int i) -> int"""
        return _dolfin.Cell_vertexID(*args)

    def edgeID(*args):
        """edgeID(self, int i) -> int"""
        return _dolfin.Cell_edgeID(*args)

    def faceID(*args):
        """faceID(self, int i) -> int"""
        return _dolfin.Cell_faceID(*args)

    def volume(*args):
        """volume(self) -> real"""
        return _dolfin.Cell_volume(*args)

    def diameter(*args):
        """diameter(self) -> real"""
        return _dolfin.Cell_diameter(*args)

    def edgeAlignment(*args):
        """edgeAlignment(self, uint i) -> uint"""
        return _dolfin.Cell_edgeAlignment(*args)

    def faceAlignment(*args):
        """faceAlignment(self, uint i) -> uint"""
        return _dolfin.Cell_faceAlignment(*args)

    def __eq__(*args):
        """__eq__(self, Cell cell) -> bool"""
        return _dolfin.Cell___eq__(*args)

    def __ne__(*args):
        """__ne__(self, Cell cell) -> bool"""
        return _dolfin.Cell___ne__(*args)

    def mark(*args):
        """
        mark(self, bool refine=True)
        mark(self)
        """
        return _dolfin.Cell_mark(*args)

Cell_swigregister = _dolfin.Cell_swigregister
Cell_swigregister(Cell)

class Face(_object):
    """Proxy of C++ Face class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Face, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Face, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> Face"""
        this = _dolfin.new_Face(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Face
    __del__ = lambda self : None;
    def clear(*args):
        """clear(self)"""
        return _dolfin.Face_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.Face_id(*args)

    def numEdges(*args):
        """numEdges(self) -> unsigned int"""
        return _dolfin.Face_numEdges(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> unsigned int"""
        return _dolfin.Face_numCellNeighbors(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.Face_edge(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.Face_cell(*args)

    def localID(*args):
        """localID(self, int i) -> int"""
        return _dolfin.Face_localID(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.Face_mesh(*args)

    def area(*args):
        """area(self) -> real"""
        return _dolfin.Face_area(*args)

    def equals(*args):
        """
        equals(self, Edge e0, Edge e1, Edge e2) -> bool
        equals(self, Edge e0, Edge e1) -> bool
        """
        return _dolfin.Face_equals(*args)

    def contains(*args):
        """
        contains(self, Vertex n) -> bool
        contains(self, Point point) -> bool
        """
        return _dolfin.Face_contains(*args)

    __swig_setmethods__["fbids"] = _dolfin.Face_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.Face_fbids_get
    if _newclass:fbids = property(_dolfin.Face_fbids_get, _dolfin.Face_fbids_set)
Face_swigregister = _dolfin.Face_swigregister
Face_swigregister(Face)

class VertexIterator(_object):
    """Proxy of C++ VertexIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VertexIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VertexIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, Mesh mesh) -> VertexIterator
        __init__(self, Mesh mesh) -> VertexIterator
        __init__(self, Boundary boundary) -> VertexIterator
        __init__(self, Vertex vertex) -> VertexIterator
        __init__(self, VertexIterator vertexIterator) -> VertexIterator
        __init__(self, Cell cell) -> VertexIterator
        __init__(self, CellIterator cellIterator) -> VertexIterator
        """
        this = _dolfin.new_VertexIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_VertexIterator
    __del__ = lambda self : None;
    def increment(*args):
        """increment(self) -> VertexIterator"""
        return _dolfin.VertexIterator_increment(*args)

    def end(*args):
        """end(self) -> bool"""
        return _dolfin.VertexIterator_end(*args)

    def last(*args):
        """last(self) -> bool"""
        return _dolfin.VertexIterator_last(*args)

    def index(*args):
        """index(self) -> int"""
        return _dolfin.VertexIterator_index(*args)

    def __ref__(*args):
        """__ref__(self) -> Vertex"""
        return _dolfin.VertexIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> Vertex"""
        return _dolfin.VertexIterator___deref__(*args)

    def __eq__(*args):
        """
        __eq__(self, VertexIterator n) -> bool
        __eq__(self, Vertex n) -> bool
        """
        return _dolfin.VertexIterator___eq__(*args)

    def __ne__(*args):
        """
        __ne__(self, VertexIterator n) -> bool
        __ne__(self, Vertex n) -> bool
        """
        return _dolfin.VertexIterator___ne__(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.VertexIterator_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.VertexIterator_id(*args)

    def numVertexNeighbors(*args):
        """numVertexNeighbors(self) -> int"""
        return _dolfin.VertexIterator_numVertexNeighbors(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> int"""
        return _dolfin.VertexIterator_numCellNeighbors(*args)

    def numEdgeNeighbors(*args):
        """numEdgeNeighbors(self) -> int"""
        return _dolfin.VertexIterator_numEdgeNeighbors(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.VertexIterator_vertex(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.VertexIterator_cell(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.VertexIterator_edge(*args)

    def parent(*args):
        """parent(self) -> Vertex"""
        return _dolfin.VertexIterator_parent(*args)

    def child(*args):
        """child(self) -> Vertex"""
        return _dolfin.VertexIterator_child(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.VertexIterator_mesh(*args)

    def coord(*args):
        """
        coord(self) -> Point
        coord(self) -> Point
        """
        return _dolfin.VertexIterator_coord(*args)

    def midpoint(*args):
        """midpoint(self, Vertex n) -> Point"""
        return _dolfin.VertexIterator_midpoint(*args)

    def dist(*args):
        """
        dist(self, Vertex n) -> real
        dist(self, Point p) -> real
        dist(self, real x, real y=0.0, real z=0.0) -> real
        dist(self, real x, real y=0.0) -> real
        dist(self, real x) -> real
        """
        return _dolfin.VertexIterator_dist(*args)

    def neighbor(*args):
        """neighbor(self, Vertex n) -> bool"""
        return _dolfin.VertexIterator_neighbor(*args)

    def __lt__(*args):
        """__lt__(self, int id) -> bool"""
        return _dolfin.VertexIterator___lt__(*args)

    def __le__(*args):
        """__le__(self, int id) -> bool"""
        return _dolfin.VertexIterator___le__(*args)

    def __gt__(*args):
        """__gt__(self, int id) -> bool"""
        return _dolfin.VertexIterator___gt__(*args)

    def __ge__(*args):
        """__ge__(self, int id) -> bool"""
        return _dolfin.VertexIterator___ge__(*args)

    __swig_setmethods__["nbids"] = _dolfin.VertexIterator_nbids_set
    __swig_getmethods__["nbids"] = _dolfin.VertexIterator_nbids_get
    if _newclass:nbids = property(_dolfin.VertexIterator_nbids_get, _dolfin.VertexIterator_nbids_set)
VertexIterator_swigregister = _dolfin.VertexIterator_swigregister
VertexIterator_swigregister(VertexIterator)

class CellIterator(_object):
    """Proxy of C++ CellIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CellIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CellIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, Mesh mesh) -> CellIterator
        __init__(self, Mesh mesh) -> CellIterator
        __init__(self, Vertex vertex) -> CellIterator
        __init__(self, VertexIterator vertexIterator) -> CellIterator
        __init__(self, Cell cell) -> CellIterator
        __init__(self, CellIterator cellIterator) -> CellIterator
        __init__(self, Edge edge) -> CellIterator
        __init__(self, EdgeIterator edgeIterator) -> CellIterator
        __init__(self, Face face) -> CellIterator
        __init__(self, FaceIterator faceIterator) -> CellIterator
        """
        this = _dolfin.new_CellIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_CellIterator
    __del__ = lambda self : None;
    def increment(*args):
        """increment(self) -> CellIterator"""
        return _dolfin.CellIterator_increment(*args)

    def end(*args):
        """end(self) -> bool"""
        return _dolfin.CellIterator_end(*args)

    def last(*args):
        """last(self) -> bool"""
        return _dolfin.CellIterator_last(*args)

    def index(*args):
        """index(self) -> int"""
        return _dolfin.CellIterator_index(*args)

    def __ref__(*args):
        """__ref__(self) -> Cell"""
        return _dolfin.CellIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> Cell"""
        return _dolfin.CellIterator___deref__(*args)

    def __eq__(*args):
        """
        __eq__(self, CellIterator c) -> bool
        __eq__(self, Cell c) -> bool
        """
        return _dolfin.CellIterator___eq__(*args)

    def __ne__(*args):
        """
        __ne__(self, CellIterator c) -> bool
        __ne__(self, Cell c) -> bool
        """
        return _dolfin.CellIterator___ne__(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.CellIterator_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.CellIterator_id(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.CellIterator_type(*args)

    def orientation(*args):
        """orientation(self) -> int"""
        return _dolfin.CellIterator_orientation(*args)

    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.CellIterator_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.CellIterator_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.CellIterator_numFaces(*args)

    def numBoundaries(*args):
        """numBoundaries(self) -> int"""
        return _dolfin.CellIterator_numBoundaries(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> int"""
        return _dolfin.CellIterator_numCellNeighbors(*args)

    def numVertexNeighbors(*args):
        """numVertexNeighbors(self) -> int"""
        return _dolfin.CellIterator_numVertexNeighbors(*args)

    def numChildren(*args):
        """numChildren(self) -> int"""
        return _dolfin.CellIterator_numChildren(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.CellIterator_vertex(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.CellIterator_edge(*args)

    def face(*args):
        """face(self, int i) -> Face"""
        return _dolfin.CellIterator_face(*args)

    def neighbor(*args):
        """neighbor(self, int i) -> Cell"""
        return _dolfin.CellIterator_neighbor(*args)

    def parent(*args):
        """parent(self) -> Cell"""
        return _dolfin.CellIterator_parent(*args)

    def child(*args):
        """child(self, int i) -> Cell"""
        return _dolfin.CellIterator_child(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.CellIterator_mesh(*args)

    def coord(*args):
        """coord(self, int i) -> Point"""
        return _dolfin.CellIterator_coord(*args)

    def midpoint(*args):
        """midpoint(self) -> Point"""
        return _dolfin.CellIterator_midpoint(*args)

    def vertexID(*args):
        """vertexID(self, int i) -> int"""
        return _dolfin.CellIterator_vertexID(*args)

    def edgeID(*args):
        """edgeID(self, int i) -> int"""
        return _dolfin.CellIterator_edgeID(*args)

    def faceID(*args):
        """faceID(self, int i) -> int"""
        return _dolfin.CellIterator_faceID(*args)

    def volume(*args):
        """volume(self) -> real"""
        return _dolfin.CellIterator_volume(*args)

    def diameter(*args):
        """diameter(self) -> real"""
        return _dolfin.CellIterator_diameter(*args)

    def edgeAlignment(*args):
        """edgeAlignment(self, uint i) -> uint"""
        return _dolfin.CellIterator_edgeAlignment(*args)

    def faceAlignment(*args):
        """faceAlignment(self, uint i) -> uint"""
        return _dolfin.CellIterator_faceAlignment(*args)

    def mark(*args):
        """
        mark(self, bool refine=True)
        mark(self)
        """
        return _dolfin.CellIterator_mark(*args)

CellIterator_swigregister = _dolfin.CellIterator_swigregister
CellIterator_swigregister(CellIterator)

class EdgeIterator(_object):
    """Proxy of C++ EdgeIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EdgeIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EdgeIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, Mesh mesh) -> EdgeIterator
        __init__(self, Mesh mesh) -> EdgeIterator
        __init__(self, Boundary boundary) -> EdgeIterator
        __init__(self, Boundary boundary) -> EdgeIterator
        __init__(self, Vertex vertex) -> EdgeIterator
        __init__(self, VertexIterator vertexIterator) -> EdgeIterator
        __init__(self, Cell cell) -> EdgeIterator
        __init__(self, CellIterator cellIterator) -> EdgeIterator
        __init__(self, Face face) -> EdgeIterator
        __init__(self, FaceIterator faceIterator) -> EdgeIterator
        """
        this = _dolfin.new_EdgeIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_EdgeIterator
    __del__ = lambda self : None;
    def increment(*args):
        """increment(self) -> EdgeIterator"""
        return _dolfin.EdgeIterator_increment(*args)

    def end(*args):
        """end(self) -> bool"""
        return _dolfin.EdgeIterator_end(*args)

    def last(*args):
        """last(self) -> bool"""
        return _dolfin.EdgeIterator_last(*args)

    def index(*args):
        """index(self) -> int"""
        return _dolfin.EdgeIterator_index(*args)

    def __ref__(*args):
        """__ref__(self) -> Edge"""
        return _dolfin.EdgeIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> Edge"""
        return _dolfin.EdgeIterator___deref__(*args)

    def __eq__(*args):
        """__eq__(self, EdgeIterator n) -> bool"""
        return _dolfin.EdgeIterator___eq__(*args)

    def __ne__(*args):
        """__ne__(self, EdgeIterator n) -> bool"""
        return _dolfin.EdgeIterator___ne__(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.EdgeIterator_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.EdgeIterator_id(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> unsigned int"""
        return _dolfin.EdgeIterator_numCellNeighbors(*args)

    def vertex(*args):
        """vertex(self, int i) -> Vertex"""
        return _dolfin.EdgeIterator_vertex(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.EdgeIterator_cell(*args)

    def localID(*args):
        """localID(self, int i) -> int"""
        return _dolfin.EdgeIterator_localID(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.EdgeIterator_mesh(*args)

    def coord(*args):
        """coord(self, int i) -> Point"""
        return _dolfin.EdgeIterator_coord(*args)

    def length(*args):
        """length(self) -> real"""
        return _dolfin.EdgeIterator_length(*args)

    def midpoint(*args):
        """midpoint(self) -> Point"""
        return _dolfin.EdgeIterator_midpoint(*args)

    def equals(*args):
        """equals(self, Vertex n0, Vertex n1) -> bool"""
        return _dolfin.EdgeIterator_equals(*args)

    def contains(*args):
        """
        contains(self, Vertex n) -> bool
        contains(self, Point point) -> bool
        """
        return _dolfin.EdgeIterator_contains(*args)

    __swig_setmethods__["ebids"] = _dolfin.EdgeIterator_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.EdgeIterator_ebids_get
    if _newclass:ebids = property(_dolfin.EdgeIterator_ebids_get, _dolfin.EdgeIterator_ebids_set)
EdgeIterator_swigregister = _dolfin.EdgeIterator_swigregister
EdgeIterator_swigregister(EdgeIterator)

class FaceIterator(_object):
    """Proxy of C++ FaceIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FaceIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FaceIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, Mesh mesh) -> FaceIterator
        __init__(self, Mesh mesh) -> FaceIterator
        __init__(self, Boundary boundary) -> FaceIterator
        __init__(self, Boundary boundary) -> FaceIterator
        __init__(self, Cell cell) -> FaceIterator
        __init__(self, CellIterator cellIterator) -> FaceIterator
        """
        this = _dolfin.new_FaceIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_FaceIterator
    __del__ = lambda self : None;
    def end(*args):
        """end(self) -> bool"""
        return _dolfin.FaceIterator_end(*args)

    def last(*args):
        """last(self) -> bool"""
        return _dolfin.FaceIterator_last(*args)

    def index(*args):
        """index(self) -> int"""
        return _dolfin.FaceIterator_index(*args)

    def __ref__(*args):
        """__ref__(self) -> Face"""
        return _dolfin.FaceIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> Face"""
        return _dolfin.FaceIterator___deref__(*args)

    def __eq__(*args):
        """
        __eq__(self, FaceIterator f) -> bool
        __eq__(self, Face f) -> bool
        """
        return _dolfin.FaceIterator___eq__(*args)

    def __ne__(*args):
        """
        __ne__(self, FaceIterator f) -> bool
        __ne__(self, Face f) -> bool
        """
        return _dolfin.FaceIterator___ne__(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.FaceIterator_clear(*args)

    def id(*args):
        """id(self) -> int"""
        return _dolfin.FaceIterator_id(*args)

    def numEdges(*args):
        """numEdges(self) -> unsigned int"""
        return _dolfin.FaceIterator_numEdges(*args)

    def numCellNeighbors(*args):
        """numCellNeighbors(self) -> unsigned int"""
        return _dolfin.FaceIterator_numCellNeighbors(*args)

    def edge(*args):
        """edge(self, int i) -> Edge"""
        return _dolfin.FaceIterator_edge(*args)

    def cell(*args):
        """cell(self, int i) -> Cell"""
        return _dolfin.FaceIterator_cell(*args)

    def localID(*args):
        """localID(self, int i) -> int"""
        return _dolfin.FaceIterator_localID(*args)

    def mesh(*args):
        """
        mesh(self) -> Mesh
        mesh(self) -> Mesh
        """
        return _dolfin.FaceIterator_mesh(*args)

    def area(*args):
        """area(self) -> real"""
        return _dolfin.FaceIterator_area(*args)

    def equals(*args):
        """
        equals(self, Edge e0, Edge e1, Edge e2) -> bool
        equals(self, Edge e0, Edge e1) -> bool
        """
        return _dolfin.FaceIterator_equals(*args)

    def contains(*args):
        """
        contains(self, Vertex n) -> bool
        contains(self, Point point) -> bool
        """
        return _dolfin.FaceIterator_contains(*args)

    __swig_setmethods__["fbids"] = _dolfin.FaceIterator_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.FaceIterator_fbids_get
    if _newclass:fbids = property(_dolfin.FaceIterator_fbids_get, _dolfin.FaceIterator_fbids_set)
FaceIterator_swigregister = _dolfin.FaceIterator_swigregister
FaceIterator_swigregister(FaceIterator)

class MeshIterator(_object):
    """Proxy of C++ MeshIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, MeshHierarchy meshs) -> MeshIterator
        __init__(self, MeshHierarchy meshs, Index index) -> MeshIterator
        """
        this = _dolfin.new_MeshIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshIterator
    __del__ = lambda self : None;
    def end(*args):
        """end(self) -> bool"""
        return _dolfin.MeshIterator_end(*args)

    def index(*args):
        """index(self) -> int"""
        return _dolfin.MeshIterator_index(*args)

    def __ref__(*args):
        """__ref__(self) -> Mesh"""
        return _dolfin.MeshIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> Mesh"""
        return _dolfin.MeshIterator___deref__(*args)

    def merge(*args):
        """merge(self, Mesh mesh2)"""
        return _dolfin.MeshIterator_merge(*args)

    def init(*args):
        """init(self)"""
        return _dolfin.MeshIterator_init(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.MeshIterator_clear(*args)

    def numSpaceDim(*args):
        """numSpaceDim(self) -> int"""
        return _dolfin.MeshIterator_numSpaceDim(*args)

    def numVertices(*args):
        """numVertices(self) -> int"""
        return _dolfin.MeshIterator_numVertices(*args)

    def numCells(*args):
        """numCells(self) -> int"""
        return _dolfin.MeshIterator_numCells(*args)

    def numEdges(*args):
        """numEdges(self) -> int"""
        return _dolfin.MeshIterator_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> int"""
        return _dolfin.MeshIterator_numFaces(*args)

    def createVertex(*args):
        """
        createVertex(self, Point p) -> Vertex
        createVertex(self, real x, real y, real z) -> Vertex
        """
        return _dolfin.MeshIterator_createVertex(*args)

    def createCell(*args):
        """
        createCell(self, int n0, int n1, int n2) -> Cell
        createCell(self, int n0, int n1, int n2, int n3) -> Cell
        createCell(self, Vertex n0, Vertex n1, Vertex n2) -> Cell
        createCell(self, Vertex n0, Vertex n1, Vertex n2, Vertex n3) -> Cell
        """
        return _dolfin.MeshIterator_createCell(*args)

    def createEdge(*args):
        """
        createEdge(self, int n0, int n1) -> Edge
        createEdge(self, Vertex n0, Vertex n1) -> Edge
        """
        return _dolfin.MeshIterator_createEdge(*args)

    def createFace(*args):
        """
        createFace(self, int e0, int e1, int e2) -> Face
        createFace(self, Edge e0, Edge e1, Edge e2) -> Face
        """
        return _dolfin.MeshIterator_createFace(*args)

    def remove(*args):
        """
        remove(self, Vertex vertex)
        remove(self, Cell cell)
        remove(self, Edge edge)
        remove(self, Face face)
        """
        return _dolfin.MeshIterator_remove(*args)

    def type(*args):
        """type(self) -> int"""
        return _dolfin.MeshIterator_type(*args)

    def vertex(*args):
        """vertex(self, uint id) -> Vertex"""
        return _dolfin.MeshIterator_vertex(*args)

    def cell(*args):
        """cell(self, uint id) -> Cell"""
        return _dolfin.MeshIterator_cell(*args)

    def edge(*args):
        """edge(self, uint id) -> Edge"""
        return _dolfin.MeshIterator_edge(*args)

    def face(*args):
        """face(self, uint id) -> Face"""
        return _dolfin.MeshIterator_face(*args)

    def boundary(*args):
        """boundary(self) -> Boundary"""
        return _dolfin.MeshIterator_boundary(*args)

    def refine(*args):
        """refine(self)"""
        return _dolfin.MeshIterator_refine(*args)

    def refineUniformly(*args):
        """
        refineUniformly(self)
        refineUniformly(self, int i)
        """
        return _dolfin.MeshIterator_refineUniformly(*args)

    def parent(*args):
        """parent(self) -> Mesh"""
        return _dolfin.MeshIterator_parent(*args)

    def child(*args):
        """child(self) -> Mesh"""
        return _dolfin.MeshIterator_child(*args)

    def __eq__(*args):
        """__eq__(self, Mesh mesh) -> bool"""
        return _dolfin.MeshIterator___eq__(*args)

    def __ne__(*args):
        """__ne__(self, Mesh mesh) -> bool"""
        return _dolfin.MeshIterator___ne__(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MeshIterator_disp(*args)

    def rename(*args):
        """rename(self, string name, string label)"""
        return _dolfin.MeshIterator_rename(*args)

    def name(*args):
        """name(self) -> string"""
        return _dolfin.MeshIterator_name(*args)

    def label(*args):
        """label(self) -> string"""
        return _dolfin.MeshIterator_label(*args)

    def number(*args):
        """number(self) -> int"""
        return _dolfin.MeshIterator_number(*args)

MeshIterator_swigregister = _dolfin.MeshIterator_swigregister
MeshIterator_swigregister(MeshIterator)

class UnitSquare(Mesh):
    """Proxy of C++ UnitSquare class"""
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitSquare, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitSquare, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint nx, uint ny) -> UnitSquare"""
        this = _dolfin.new_UnitSquare(*args)
        try: self.this.append(this)
        except: self.this = this
UnitSquare_swigregister = _dolfin.UnitSquare_swigregister
UnitSquare_swigregister(UnitSquare)

class UnitCube(Mesh):
    """Proxy of C++ UnitCube class"""
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitCube, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitCube, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint nx, uint ny, uint nz) -> UnitCube"""
        this = _dolfin.new_UnitCube(*args)
        try: self.this.append(this)
        except: self.this = this
UnitCube_swigregister = _dolfin.UnitCube_swigregister
UnitCube_swigregister(UnitCube)

class MeshConnectivity(_object):
    """Proxy of C++ MeshConnectivity class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshConnectivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshConnectivity, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> MeshConnectivity
        __init__(self, MeshConnectivity connectivity) -> MeshConnectivity
        """
        this = _dolfin.new_MeshConnectivity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshConnectivity
    __del__ = lambda self : None;
    def size(*args):
        """
        size(self) -> uint
        size(self, uint entity) -> uint
        """
        return _dolfin.MeshConnectivity_size(*args)

    def __call__(*args):
        """__call__(self, uint entity) -> uint"""
        return _dolfin.MeshConnectivity___call__(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.MeshConnectivity_clear(*args)

    def init(*args):
        """
        init(self, uint num_entities, uint num_connections)
        init(self, dolfin::Array<(dolfin::uint)> num_connections)
        """
        return _dolfin.MeshConnectivity_init(*args)

    def set(*args):
        """
        set(self, uint entity, uint connection, uint pos)
        set(self, uint entity, dolfin::Array<(dolfin::uint)> connections)
        set(self, uint entity, uint connections)
        set(self, dolfin::Array<(dolfin::Array<(dolfin::uint)>)> connectivity)
        """
        return _dolfin.MeshConnectivity_set(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MeshConnectivity_disp(*args)

MeshConnectivity_swigregister = _dolfin.MeshConnectivity_swigregister
MeshConnectivity_swigregister(MeshConnectivity)

class MeshEditor(_object):
    """Proxy of C++ MeshEditor class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshEditor, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshEditor, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> MeshEditor"""
        this = _dolfin.new_MeshEditor(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshEditor
    __del__ = lambda self : None;
    def open(*args):
        """open(self, NewMesh mesh, CellType::Type type, uint tdim, uint gdim)"""
        return _dolfin.MeshEditor_open(*args)

    def initVertices(*args):
        """initVertices(self, uint num_vertices)"""
        return _dolfin.MeshEditor_initVertices(*args)

    def initCells(*args):
        """initCells(self, uint num_cells)"""
        return _dolfin.MeshEditor_initCells(*args)

    def addVertex(*args):
        """
        addVertex(self, uint v, NewPoint p)
        addVertex(self, uint v, real x)
        addVertex(self, uint v, real x, real y)
        addVertex(self, uint v, real x, real y, real z)
        """
        return _dolfin.MeshEditor_addVertex(*args)

    def addCell(*args):
        """
        addCell(self, uint c, dolfin::Array<(dolfin::uint)> v)
        addCell(self, uint c, uint v0, uint v1)
        addCell(self, uint c, uint v0, uint v1, uint v2)
        addCell(self, uint c, uint v0, uint v1, uint v2, uint v3)
        """
        return _dolfin.MeshEditor_addCell(*args)

    def close(*args):
        """close(self)"""
        return _dolfin.MeshEditor_close(*args)

MeshEditor_swigregister = _dolfin.MeshEditor_swigregister
MeshEditor_swigregister(MeshEditor)

class MeshEntity(_object):
    """Proxy of C++ MeshEntity class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshEntity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshEntity, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, NewMesh mesh, uint dim, uint index) -> MeshEntity"""
        this = _dolfin.new_MeshEntity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshEntity
    __del__ = lambda self : None;
    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.MeshEntity_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.MeshEntity_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.MeshEntity_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.MeshEntity_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.MeshEntity_connections(*args)

MeshEntity_swigregister = _dolfin.MeshEntity_swigregister
MeshEntity_swigregister(MeshEntity)

class MeshEntityIterator(_object):
    """Proxy of C++ MeshEntityIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshEntityIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshEntityIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh, uint dim) -> MeshEntityIterator
        __init__(self, MeshEntity entity, uint dim) -> MeshEntityIterator
        __init__(self, MeshEntityIterator it, uint dim) -> MeshEntityIterator
        """
        this = _dolfin.new_MeshEntityIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshEntityIterator
    __del__ = lambda self : None;
    def end(*args):
        """end(self) -> bool"""
        return _dolfin.MeshEntityIterator_end(*args)

    def __ref__(*args):
        """__ref__(self) -> MeshEntity"""
        return _dolfin.MeshEntityIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> MeshEntity"""
        return _dolfin.MeshEntityIterator___deref__(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.MeshEntityIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.MeshEntityIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.MeshEntityIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.MeshEntityIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.MeshEntityIterator_connections(*args)

MeshEntityIterator_swigregister = _dolfin.MeshEntityIterator_swigregister
MeshEntityIterator_swigregister(MeshEntityIterator)

class MeshGeometry(_object):
    """Proxy of C++ MeshGeometry class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshGeometry, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshGeometry, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> MeshGeometry
        __init__(self, MeshGeometry geometry) -> MeshGeometry
        """
        this = _dolfin.new_MeshGeometry(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshGeometry
    __del__ = lambda self : None;
    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.MeshGeometry_dim(*args)

    def size(*args):
        """size(self) -> uint"""
        return _dolfin.MeshGeometry_size(*args)

    def x(*args):
        """
        x(self, uint n, uint i) -> real
        x(self, uint n, uint i) -> real
        """
        return _dolfin.MeshGeometry_x(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.MeshGeometry_clear(*args)

    def init(*args):
        """init(self, uint dim, uint size)"""
        return _dolfin.MeshGeometry_init(*args)

    def set(*args):
        """set(self, uint n, uint i, real x)"""
        return _dolfin.MeshGeometry_set(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MeshGeometry_disp(*args)

MeshGeometry_swigregister = _dolfin.MeshGeometry_swigregister
MeshGeometry_swigregister(MeshGeometry)

class MeshTopology(_object):
    """Proxy of C++ MeshTopology class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshTopology, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshTopology, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> MeshTopology
        __init__(self, MeshTopology topology) -> MeshTopology
        """
        this = _dolfin.new_MeshTopology(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshTopology
    __del__ = lambda self : None;
    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.MeshTopology_dim(*args)

    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.MeshTopology_size(*args)

    def clear(*args):
        """clear(self)"""
        return _dolfin.MeshTopology_clear(*args)

    def init(*args):
        """
        init(self, uint dim)
        init(self, uint dim, uint size)
        """
        return _dolfin.MeshTopology_init(*args)

    def __call__(*args):
        """
        __call__(self, uint d0, uint d1) -> MeshConnectivity
        __call__(self, uint d0, uint d1) -> MeshConnectivity
        """
        return _dolfin.MeshTopology___call__(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MeshTopology_disp(*args)

MeshTopology_swigregister = _dolfin.MeshTopology_swigregister
MeshTopology_swigregister(MeshTopology)

class NewMesh(Variable):
    """Proxy of C++ NewMesh class"""
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewMesh, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewMesh, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> NewMesh
        __init__(self, NewMesh mesh) -> NewMesh
        __init__(self, string filename) -> NewMesh
        """
        this = _dolfin.new_NewMesh(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewMesh
    __del__ = lambda self : None;
    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewMesh_dim(*args)

    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.NewMesh_size(*args)

    def numVertices(*args):
        """numVertices(self) -> uint"""
        return _dolfin.NewMesh_numVertices(*args)

    def numEdges(*args):
        """numEdges(self) -> uint"""
        return _dolfin.NewMesh_numEdges(*args)

    def numFaces(*args):
        """numFaces(self) -> uint"""
        return _dolfin.NewMesh_numFaces(*args)

    def numFacets(*args):
        """numFacets(self) -> uint"""
        return _dolfin.NewMesh_numFacets(*args)

    def numCells(*args):
        """numCells(self) -> uint"""
        return _dolfin.NewMesh_numCells(*args)

    def topology(*args):
        """
        topology(self) -> MeshTopology
        topology(self) -> MeshTopology
        """
        return _dolfin.NewMesh_topology(*args)

    def geometry(*args):
        """
        geometry(self) -> MeshGeometry
        geometry(self) -> MeshGeometry
        """
        return _dolfin.NewMesh_geometry(*args)

    def type(*args):
        """
        type(self) -> CellType
        type(self) -> CellType
        """
        return _dolfin.NewMesh_type(*args)

    def init(*args):
        """
        init(self, uint dim) -> uint
        init(self, uint d0, uint d1)
        init(self)
        """
        return _dolfin.NewMesh_init(*args)

    def refine(*args):
        """refine(self)"""
        return _dolfin.NewMesh_refine(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.NewMesh_disp(*args)

NewMesh_swigregister = _dolfin.NewMesh_swigregister
NewMesh_swigregister(NewMesh)

class NewMeshData(_object):
    """Proxy of C++ NewMeshData class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewMeshData, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NewMeshData, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> NewMeshData
        __init__(self, NewMeshData data) -> NewMeshData
        """
        this = _dolfin.new_NewMeshData(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewMeshData
    __del__ = lambda self : None;
    def clear(*args):
        """clear(self)"""
        return _dolfin.NewMeshData_clear(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.NewMeshData_disp(*args)

    __swig_setmethods__["topology"] = _dolfin.NewMeshData_topology_set
    __swig_getmethods__["topology"] = _dolfin.NewMeshData_topology_get
    if _newclass:topology = property(_dolfin.NewMeshData_topology_get, _dolfin.NewMeshData_topology_set)
    __swig_setmethods__["geometry"] = _dolfin.NewMeshData_geometry_set
    __swig_getmethods__["geometry"] = _dolfin.NewMeshData_geometry_get
    if _newclass:geometry = property(_dolfin.NewMeshData_geometry_get, _dolfin.NewMeshData_geometry_set)
    __swig_setmethods__["cell_type"] = _dolfin.NewMeshData_cell_type_set
    __swig_getmethods__["cell_type"] = _dolfin.NewMeshData_cell_type_get
    if _newclass:cell_type = property(_dolfin.NewMeshData_cell_type_get, _dolfin.NewMeshData_cell_type_set)
NewMeshData_swigregister = _dolfin.NewMeshData_swigregister
NewMeshData_swigregister(NewMeshData)

class NewVertex(MeshEntity):
    """Proxy of C++ NewVertex class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntity]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewVertex, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntity]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewVertex, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh, uint index) -> NewVertex
        __init__(self, MeshEntity entity) -> NewVertex
        """
        this = _dolfin.new_NewVertex(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewVertex
    __del__ = lambda self : None;
    def y(*args):
        """y(self) -> real"""
        return _dolfin.NewVertex_y(*args)

    def z(*args):
        """z(self) -> real"""
        return _dolfin.NewVertex_z(*args)

    def x(*args):
        """
        x(self) -> real
        x(self, uint i) -> real
        """
        return _dolfin.NewVertex_x(*args)

    def point(*args):
        """point(self) -> NewPoint"""
        return _dolfin.NewVertex_point(*args)

NewVertex_swigregister = _dolfin.NewVertex_swigregister
NewVertex_swigregister(NewVertex)

class NewVertexIterator(MeshEntityIterator):
    """Proxy of C++ NewVertexIterator class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewVertexIterator, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewVertexIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh) -> NewVertexIterator
        __init__(self, MeshEntity entity) -> NewVertexIterator
        __init__(self, MeshEntityIterator it) -> NewVertexIterator
        """
        this = _dolfin.new_NewVertexIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    def __ref__(*args):
        """__ref__(self) -> NewVertex"""
        return _dolfin.NewVertexIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> NewVertex"""
        return _dolfin.NewVertexIterator___deref__(*args)

    def x(*args):
        """
        x(self) -> real
        x(self, uint i) -> real
        """
        return _dolfin.NewVertexIterator_x(*args)

    def y(*args):
        """y(self) -> real"""
        return _dolfin.NewVertexIterator_y(*args)

    def z(*args):
        """z(self) -> real"""
        return _dolfin.NewVertexIterator_z(*args)

    def point(*args):
        """point(self) -> NewPoint"""
        return _dolfin.NewVertexIterator_point(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.NewVertexIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewVertexIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.NewVertexIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.NewVertexIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.NewVertexIterator_connections(*args)

NewVertexIterator_swigregister = _dolfin.NewVertexIterator_swigregister
NewVertexIterator_swigregister(NewVertexIterator)

class NewEdge(MeshEntity):
    """Proxy of C++ NewEdge class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntity]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewEdge, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntity]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewEdge, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh, uint index) -> NewEdge
        __init__(self, MeshEntity entity) -> NewEdge
        """
        this = _dolfin.new_NewEdge(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewEdge
    __del__ = lambda self : None;
    def midpoint(*args):
        """midpoint(self) -> NewPoint"""
        return _dolfin.NewEdge_midpoint(*args)

NewEdge_swigregister = _dolfin.NewEdge_swigregister
NewEdge_swigregister(NewEdge)

class NewEdgeIterator(MeshEntityIterator):
    """Proxy of C++ NewEdgeIterator class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewEdgeIterator, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewEdgeIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh) -> NewEdgeIterator
        __init__(self, MeshEntity entity) -> NewEdgeIterator
        __init__(self, MeshEntityIterator it) -> NewEdgeIterator
        """
        this = _dolfin.new_NewEdgeIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    def __ref__(*args):
        """__ref__(self) -> NewEdge"""
        return _dolfin.NewEdgeIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> NewEdge"""
        return _dolfin.NewEdgeIterator___deref__(*args)

    def midpoint(*args):
        """midpoint(self) -> NewPoint"""
        return _dolfin.NewEdgeIterator_midpoint(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.NewEdgeIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewEdgeIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.NewEdgeIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.NewEdgeIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.NewEdgeIterator_connections(*args)

NewEdgeIterator_swigregister = _dolfin.NewEdgeIterator_swigregister
NewEdgeIterator_swigregister(NewEdgeIterator)

class NewFace(MeshEntity):
    """Proxy of C++ NewFace class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntity]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewFace, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntity]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewFace, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, NewMesh mesh, uint index) -> NewFace"""
        this = _dolfin.new_NewFace(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewFace
    __del__ = lambda self : None;
NewFace_swigregister = _dolfin.NewFace_swigregister
NewFace_swigregister(NewFace)

class NewFaceIterator(MeshEntityIterator):
    """Proxy of C++ NewFaceIterator class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewFaceIterator, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewFaceIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh) -> NewFaceIterator
        __init__(self, MeshEntity entity) -> NewFaceIterator
        __init__(self, MeshEntityIterator it) -> NewFaceIterator
        """
        this = _dolfin.new_NewFaceIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    def __ref__(*args):
        """__ref__(self) -> NewFace"""
        return _dolfin.NewFaceIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> NewFace"""
        return _dolfin.NewFaceIterator___deref__(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.NewFaceIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewFaceIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.NewFaceIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.NewFaceIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.NewFaceIterator_connections(*args)

NewFaceIterator_swigregister = _dolfin.NewFaceIterator_swigregister
NewFaceIterator_swigregister(NewFaceIterator)

class NewFacet(MeshEntity):
    """Proxy of C++ NewFacet class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntity]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewFacet, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntity]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewFacet, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, NewMesh mesh, uint index) -> NewFacet"""
        this = _dolfin.new_NewFacet(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewFacet
    __del__ = lambda self : None;
NewFacet_swigregister = _dolfin.NewFacet_swigregister
NewFacet_swigregister(NewFacet)

class NewFacetIterator(MeshEntityIterator):
    """Proxy of C++ NewFacetIterator class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewFacetIterator, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewFacetIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh) -> NewFacetIterator
        __init__(self, MeshEntity entity) -> NewFacetIterator
        __init__(self, MeshEntityIterator it) -> NewFacetIterator
        """
        this = _dolfin.new_NewFacetIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    def __ref__(*args):
        """__ref__(self) -> NewFacet"""
        return _dolfin.NewFacetIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> NewFacet"""
        return _dolfin.NewFacetIterator___deref__(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.NewFacetIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewFacetIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.NewFacetIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.NewFacetIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.NewFacetIterator_connections(*args)

NewFacetIterator_swigregister = _dolfin.NewFacetIterator_swigregister
NewFacetIterator_swigregister(NewFacetIterator)

class NewCell(MeshEntity):
    """Proxy of C++ NewCell class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntity]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewCell, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntity]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewCell, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, NewMesh mesh, uint index) -> NewCell"""
        this = _dolfin.new_NewCell(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewCell
    __del__ = lambda self : None;
NewCell_swigregister = _dolfin.NewCell_swigregister
NewCell_swigregister(NewCell)

class NewCellIterator(MeshEntityIterator):
    """Proxy of C++ NewCellIterator class"""
    __swig_setmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewCellIterator, name, value)
    __swig_getmethods__ = {}
    for _s in [MeshEntityIterator]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewCellIterator, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, NewMesh mesh) -> NewCellIterator
        __init__(self, MeshEntity entity) -> NewCellIterator
        __init__(self, MeshEntityIterator it) -> NewCellIterator
        """
        this = _dolfin.new_NewCellIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    def __ref__(*args):
        """__ref__(self) -> NewCell"""
        return _dolfin.NewCellIterator___ref__(*args)

    def __deref__(*args):
        """__deref__(self) -> NewCell"""
        return _dolfin.NewCellIterator___deref__(*args)

    def mesh(*args):
        """
        mesh(self) -> NewMesh
        mesh(self) -> NewMesh
        """
        return _dolfin.NewCellIterator_mesh(*args)

    def dim(*args):
        """dim(self) -> uint"""
        return _dolfin.NewCellIterator_dim(*args)

    def index(*args):
        """index(self) -> uint"""
        return _dolfin.NewCellIterator_index(*args)

    def numConnections(*args):
        """numConnections(self, uint dim) -> uint"""
        return _dolfin.NewCellIterator_numConnections(*args)

    def connections(*args):
        """
        connections(self, uint dim) -> uint
        connections(self, uint dim) -> uint
        """
        return _dolfin.NewCellIterator_connections(*args)

NewCellIterator_swigregister = _dolfin.NewCellIterator_swigregister
NewCellIterator_swigregister(NewCellIterator)

class TopologyComputation(_object):
    """Proxy of C++ TopologyComputation class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TopologyComputation, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TopologyComputation, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def computeEntities(*args):
        """computeEntities(NewMesh mesh, uint dim) -> uint"""
        return _dolfin.TopologyComputation_computeEntities(*args)

    if _newclass:computeEntities = staticmethod(computeEntities)
    __swig_getmethods__["computeEntities"] = lambda x: computeEntities
    def computeConnectivity(*args):
        """computeConnectivity(NewMesh mesh, uint d0, uint d1)"""
        return _dolfin.TopologyComputation_computeConnectivity(*args)

    if _newclass:computeConnectivity = staticmethod(computeConnectivity)
    __swig_getmethods__["computeConnectivity"] = lambda x: computeConnectivity
TopologyComputation_swigregister = _dolfin.TopologyComputation_swigregister
TopologyComputation_swigregister(TopologyComputation)

def TopologyComputation_computeEntities(*args):
  """TopologyComputation_computeEntities(NewMesh mesh, uint dim) -> uint"""
  return _dolfin.TopologyComputation_computeEntities(*args)

def TopologyComputation_computeConnectivity(*args):
  """TopologyComputation_computeConnectivity(NewMesh mesh, uint d0, uint d1)"""
  return _dolfin.TopologyComputation_computeConnectivity(*args)

class CellType(_object):
    """Proxy of C++ CellType class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CellType, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CellType, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    point = _dolfin.CellType_point
    interval = _dolfin.CellType_interval
    triangle = _dolfin.CellType_triangle
    tetrahedron = _dolfin.CellType_tetrahedron
    __swig_destroy__ = _dolfin.delete_CellType
    __del__ = lambda self : None;
    def create(*args):
        """create(Type type) -> CellType"""
        return _dolfin.CellType_create(*args)

    if _newclass:create = staticmethod(create)
    __swig_getmethods__["create"] = lambda x: create
    def type(*args):
        """type(string type) -> int"""
        return _dolfin.CellType_type(*args)

    if _newclass:type = staticmethod(type)
    __swig_getmethods__["type"] = lambda x: type
    def cellType(*args):
        """cellType(self) -> int"""
        return _dolfin.CellType_cellType(*args)

    def facetType(*args):
        """facetType(self) -> int"""
        return _dolfin.CellType_facetType(*args)

    def numEntities(*args):
        """numEntities(self, uint dim) -> uint"""
        return _dolfin.CellType_numEntities(*args)

    def numVertices(*args):
        """numVertices(self, uint dim) -> uint"""
        return _dolfin.CellType_numVertices(*args)

    def createEntities(*args):
        """createEntities(self, uint e, uint dim, uint v)"""
        return _dolfin.CellType_createEntities(*args)

    def refineCell(*args):
        """refineCell(self, NewCell cell, MeshEditor editor, uint current_cell)"""
        return _dolfin.CellType_refineCell(*args)

    def description(*args):
        """description(self) -> string"""
        return _dolfin.CellType_description(*args)

CellType_swigregister = _dolfin.CellType_swigregister
CellType_swigregister(CellType)

def CellType_create(*args):
  """CellType_create(Type type) -> CellType"""
  return _dolfin.CellType_create(*args)

def CellType_type(*args):
  """CellType_type(string type) -> int"""
  return _dolfin.CellType_type(*args)

class Interval(CellType):
    """Proxy of C++ Interval class"""
    __swig_setmethods__ = {}
    for _s in [CellType]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Interval, name, value)
    __swig_getmethods__ = {}
    for _s in [CellType]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Interval, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> Interval"""
        this = _dolfin.new_Interval(*args)
        try: self.this.append(this)
        except: self.this = this
    def numEntities(*args):
        """numEntities(self, uint dim) -> uint"""
        return _dolfin.Interval_numEntities(*args)

    def numVertices(*args):
        """numVertices(self, uint dim) -> uint"""
        return _dolfin.Interval_numVertices(*args)

    def createEntities(*args):
        """createEntities(self, uint e, uint dim, uint v)"""
        return _dolfin.Interval_createEntities(*args)

    def refineCell(*args):
        """refineCell(self, NewCell cell, MeshEditor editor, uint current_cell)"""
        return _dolfin.Interval_refineCell(*args)

    def description(*args):
        """description(self) -> string"""
        return _dolfin.Interval_description(*args)

Interval_swigregister = _dolfin.Interval_swigregister
Interval_swigregister(Interval)

class NewTriangle(CellType):
    """Proxy of C++ NewTriangle class"""
    __swig_setmethods__ = {}
    for _s in [CellType]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewTriangle, name, value)
    __swig_getmethods__ = {}
    for _s in [CellType]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewTriangle, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> NewTriangle"""
        this = _dolfin.new_NewTriangle(*args)
        try: self.this.append(this)
        except: self.this = this
    def numEntities(*args):
        """numEntities(self, uint dim) -> uint"""
        return _dolfin.NewTriangle_numEntities(*args)

    def numVertices(*args):
        """numVertices(self, uint dim) -> uint"""
        return _dolfin.NewTriangle_numVertices(*args)

    def createEntities(*args):
        """createEntities(self, uint e, uint dim, uint v)"""
        return _dolfin.NewTriangle_createEntities(*args)

    def refineCell(*args):
        """refineCell(self, NewCell cell, MeshEditor editor, uint current_cell)"""
        return _dolfin.NewTriangle_refineCell(*args)

    def description(*args):
        """description(self) -> string"""
        return _dolfin.NewTriangle_description(*args)

NewTriangle_swigregister = _dolfin.NewTriangle_swigregister
NewTriangle_swigregister(NewTriangle)

class NewTetrahedron(CellType):
    """Proxy of C++ NewTetrahedron class"""
    __swig_setmethods__ = {}
    for _s in [CellType]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewTetrahedron, name, value)
    __swig_getmethods__ = {}
    for _s in [CellType]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewTetrahedron, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> NewTetrahedron"""
        this = _dolfin.new_NewTetrahedron(*args)
        try: self.this.append(this)
        except: self.this = this
    def numEntities(*args):
        """numEntities(self, uint dim) -> uint"""
        return _dolfin.NewTetrahedron_numEntities(*args)

    def numVertices(*args):
        """numVertices(self, uint dim) -> uint"""
        return _dolfin.NewTetrahedron_numVertices(*args)

    def createEntities(*args):
        """createEntities(self, uint e, uint dim, uint v)"""
        return _dolfin.NewTetrahedron_createEntities(*args)

    def refineCell(*args):
        """refineCell(self, NewCell cell, MeshEditor editor, uint current_cell)"""
        return _dolfin.NewTetrahedron_refineCell(*args)

    def description(*args):
        """description(self) -> string"""
        return _dolfin.NewTetrahedron_description(*args)

NewTetrahedron_swigregister = _dolfin.NewTetrahedron_swigregister
NewTetrahedron_swigregister(NewTetrahedron)

class UniformMeshRefinement(_object):
    """Proxy of C++ UniformMeshRefinement class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UniformMeshRefinement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UniformMeshRefinement, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def refine(*args):
        """refine(NewMesh mesh)"""
        return _dolfin.UniformMeshRefinement_refine(*args)

    if _newclass:refine = staticmethod(refine)
    __swig_getmethods__["refine"] = lambda x: refine
    def refineSimplex(*args):
        """refineSimplex(NewMesh mesh)"""
        return _dolfin.UniformMeshRefinement_refineSimplex(*args)

    if _newclass:refineSimplex = staticmethod(refineSimplex)
    __swig_getmethods__["refineSimplex"] = lambda x: refineSimplex
UniformMeshRefinement_swigregister = _dolfin.UniformMeshRefinement_swigregister
UniformMeshRefinement_swigregister(UniformMeshRefinement)

def UniformMeshRefinement_refine(*args):
  """UniformMeshRefinement_refine(NewMesh mesh)"""
  return _dolfin.UniformMeshRefinement_refine(*args)

def UniformMeshRefinement_refineSimplex(*args):
  """UniformMeshRefinement_refineSimplex(NewMesh mesh)"""
  return _dolfin.UniformMeshRefinement_refineSimplex(*args)

class NewPoint(_object):
    """Proxy of C++ NewPoint class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewPoint, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NewPoint, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> NewPoint
        __init__(self, real x) -> NewPoint
        __init__(self, real x, real y) -> NewPoint
        __init__(self, real x, real y, real z) -> NewPoint
        """
        this = _dolfin.new_NewPoint(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_NewPoint
    __del__ = lambda self : None;
    def x(*args):
        """x(self) -> real"""
        return _dolfin.NewPoint_x(*args)

    def y(*args):
        """y(self) -> real"""
        return _dolfin.NewPoint_y(*args)

    def z(*args):
        """z(self) -> real"""
        return _dolfin.NewPoint_z(*args)

NewPoint_swigregister = _dolfin.NewPoint_swigregister
NewPoint_swigregister(NewPoint)

class BoundaryComputation(_object):
    """Proxy of C++ BoundaryComputation class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryComputation, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryComputation, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def computeBoundary(*args):
        """computeBoundary(NewMesh mesh, BoundaryMesh boundary)"""
        return _dolfin.BoundaryComputation_computeBoundary(*args)

    if _newclass:computeBoundary = staticmethod(computeBoundary)
    __swig_getmethods__["computeBoundary"] = lambda x: computeBoundary
BoundaryComputation_swigregister = _dolfin.BoundaryComputation_swigregister
BoundaryComputation_swigregister(BoundaryComputation)

def BoundaryComputation_computeBoundary(*args):
  """BoundaryComputation_computeBoundary(NewMesh mesh, BoundaryMesh boundary)"""
  return _dolfin.BoundaryComputation_computeBoundary(*args)

class BoundaryMesh(NewMesh):
    """Proxy of C++ BoundaryMesh class"""
    __swig_setmethods__ = {}
    for _s in [NewMesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryMesh, name, value)
    __swig_getmethods__ = {}
    for _s in [NewMesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryMesh, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> BoundaryMesh
        __init__(self, NewMesh mesh) -> BoundaryMesh
        """
        this = _dolfin.new_BoundaryMesh(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_BoundaryMesh
    __del__ = lambda self : None;
    def init(*args):
        """init(self, NewMesh mesh)"""
        return _dolfin.BoundaryMesh_init(*args)

BoundaryMesh_swigregister = _dolfin.BoundaryMesh_swigregister
BoundaryMesh_swigregister(BoundaryMesh)

class NewUnitCube(NewMesh):
    """Proxy of C++ NewUnitCube class"""
    __swig_setmethods__ = {}
    for _s in [NewMesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewUnitCube, name, value)
    __swig_getmethods__ = {}
    for _s in [NewMesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewUnitCube, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint nx, uint ny, uint nz) -> NewUnitCube"""
        this = _dolfin.new_NewUnitCube(*args)
        try: self.this.append(this)
        except: self.this = this
NewUnitCube_swigregister = _dolfin.NewUnitCube_swigregister
NewUnitCube_swigregister(NewUnitCube)

class NewUnitSquare(NewMesh):
    """Proxy of C++ NewUnitSquare class"""
    __swig_setmethods__ = {}
    for _s in [NewMesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NewUnitSquare, name, value)
    __swig_getmethods__ = {}
    for _s in [NewMesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NewUnitSquare, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint nx, uint ny) -> NewUnitSquare"""
        this = _dolfin.new_NewUnitSquare(*args)
        try: self.this.append(this)
        except: self.this = this
NewUnitSquare_swigregister = _dolfin.NewUnitSquare_swigregister
NewUnitSquare_swigregister(NewUnitSquare)

class Dependencies(_object):
    """Proxy of C++ Dependencies class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Dependencies, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Dependencies, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint N) -> Dependencies"""
        this = _dolfin.new_Dependencies(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Dependencies
    __del__ = lambda self : None;
    def setsize(*args):
        """setsize(self, uint i, uint size)"""
        return _dolfin.Dependencies_setsize(*args)

    def set(*args):
        """
        set(self, uint i, uint j, bool checknew=False)
        set(self, uint i, uint j)
        set(self, uBlasSparseMatrix A)
        """
        return _dolfin.Dependencies_set(*args)

    def transp(*args):
        """transp(self, Dependencies dependencies)"""
        return _dolfin.Dependencies_transp(*args)

    def detect(*args):
        """detect(self, ODE ode)"""
        return _dolfin.Dependencies_detect(*args)

    def sparse(*args):
        """sparse(self) -> bool"""
        return _dolfin.Dependencies_sparse(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.Dependencies_disp(*args)

Dependencies_swigregister = _dolfin.Dependencies_swigregister
Dependencies_swigregister(Dependencies)

class Homotopy(_object):
    """Proxy of C++ Homotopy class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Homotopy, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Homotopy, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_Homotopy
    __del__ = lambda self : None;
    def solve(*args):
        """solve(self)"""
        return _dolfin.Homotopy_solve(*args)

    def solutions(*args):
        """solutions(self) -> dolfin::Array<(p.dolfin::complex)>"""
        return _dolfin.Homotopy_solutions(*args)

    def z0(*args):
        """z0(self, complex z)"""
        return _dolfin.Homotopy_z0(*args)

    def F(*args):
        """F(self, complex z, complex y)"""
        return _dolfin.Homotopy_F(*args)

    def JF(*args):
        """JF(self, complex z, complex x, complex y)"""
        return _dolfin.Homotopy_JF(*args)

    def G(*args):
        """G(self, complex z, complex y)"""
        return _dolfin.Homotopy_G(*args)

    def JG(*args):
        """JG(self, complex z, complex x, complex y)"""
        return _dolfin.Homotopy_JG(*args)

    def modify(*args):
        """modify(self, complex z)"""
        return _dolfin.Homotopy_modify(*args)

    def verify(*args):
        """verify(self, complex z) -> bool"""
        return _dolfin.Homotopy_verify(*args)

    def degree(*args):
        """degree(self, uint i) -> uint"""
        return _dolfin.Homotopy_degree(*args)

Homotopy_swigregister = _dolfin.Homotopy_swigregister
Homotopy_swigregister(Homotopy)

class HomotopyJacobian(uBlasKrylovMatrix):
    """Proxy of C++ HomotopyJacobian class"""
    __swig_setmethods__ = {}
    for _s in [uBlasKrylovMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [uBlasKrylovMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyJacobian, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_HomotopyJacobian
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.HomotopyJacobian_size(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.HomotopyJacobian_mult(*args)

HomotopyJacobian_swigregister = _dolfin.HomotopyJacobian_swigregister
HomotopyJacobian_swigregister(HomotopyJacobian)

class HomotopyODE(_object):
    """Proxy of C++ HomotopyODE class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyODE, name)
    __repr__ = _swig_repr
    ode = _dolfin.HomotopyODE_ode
    endgame = _dolfin.HomotopyODE_endgame
    def __init__(self, *args): 
        """__init__(self, Homotopy homotopy, uint n, real T) -> HomotopyODE"""
        this = _dolfin.new_HomotopyODE(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_HomotopyODE
    __del__ = lambda self : None;
    def z0(*args):
        """z0(self, complex z)"""
        return _dolfin.HomotopyODE_z0(*args)

    def f(*args):
        """f(self, complex z, real t, complex y)"""
        return _dolfin.HomotopyODE_f(*args)

    def M(*args):
        """M(self, complex x, complex y, complex z, real t)"""
        return _dolfin.HomotopyODE_M(*args)

    def J(*args):
        """J(self, complex x, complex y, complex u, real t)"""
        return _dolfin.HomotopyODE_J(*args)

    def update(*args):
        """update(self, complex z, real t, bool end) -> bool"""
        return _dolfin.HomotopyODE_update(*args)

    def state(*args):
        """state(self) -> int"""
        return _dolfin.HomotopyODE_state(*args)

HomotopyODE_swigregister = _dolfin.HomotopyODE_swigregister
HomotopyODE_swigregister(HomotopyODE)

class Method(_object):
    """Proxy of C++ Method class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Method, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Method, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    cG = _dolfin.Method_cG
    dG = _dolfin.Method_dG
    none = _dolfin.Method_none
    __swig_destroy__ = _dolfin.delete_Method
    __del__ = lambda self : None;
    def type(*args):
        """type(self) -> int"""
        return _dolfin.Method_type(*args)

    def degree(*args):
        """degree(self) -> unsigned int"""
        return _dolfin.Method_degree(*args)

    def order(*args):
        """order(self) -> unsigned int"""
        return _dolfin.Method_order(*args)

    def nsize(*args):
        """nsize(self) -> unsigned int"""
        return _dolfin.Method_nsize(*args)

    def qsize(*args):
        """qsize(self) -> unsigned int"""
        return _dolfin.Method_qsize(*args)

    def npoint(*args):
        """npoint(self, unsigned int i) -> real"""
        return _dolfin.Method_npoint(*args)

    def qpoint(*args):
        """qpoint(self, unsigned int i) -> real"""
        return _dolfin.Method_qpoint(*args)

    def nweight(*args):
        """nweight(self, unsigned int i, unsigned int j) -> real"""
        return _dolfin.Method_nweight(*args)

    def qweight(*args):
        """qweight(self, unsigned int i) -> real"""
        return _dolfin.Method_qweight(*args)

    def eval(*args):
        """eval(self, unsigned int i, real tau) -> real"""
        return _dolfin.Method_eval(*args)

    def derivative(*args):
        """derivative(self, unsigned int i) -> real"""
        return _dolfin.Method_derivative(*args)

    def update(*args):
        """
        update(self, real x0, real f, real k, real values)
        update(self, real x0, real f, real k, real values, real alpha)
        """
        return _dolfin.Method_update(*args)

    def ueval(*args):
        """
        ueval(self, real x0, real values, real tau) -> real
        ueval(self, real x0, uBlasVector values, uint offset, real tau) -> real
        ueval(self, real x0, real values, uint i) -> real
        """
        return _dolfin.Method_ueval(*args)

    def residual(*args):
        """
        residual(self, real x0, real values, real f, real k) -> real
        residual(self, real x0, uBlasVector values, uint offset, real f, real k) -> real
        """
        return _dolfin.Method_residual(*args)

    def timestep(*args):
        """timestep(self, real r, real tol, real k0, real kmax) -> real"""
        return _dolfin.Method_timestep(*args)

    def error(*args):
        """error(self, real k, real r) -> real"""
        return _dolfin.Method_error(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.Method_disp(*args)

Method_swigregister = _dolfin.Method_swigregister
Method_swigregister(Method)

class MonoAdaptiveFixedPointSolver(_object):
    """Proxy of C++ MonoAdaptiveFixedPointSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveFixedPointSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, MonoAdaptiveTimeSlab timeslab) -> MonoAdaptiveFixedPointSolver"""
        this = _dolfin.new_MonoAdaptiveFixedPointSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveFixedPointSolver
    __del__ = lambda self : None;
MonoAdaptiveFixedPointSolver_swigregister = _dolfin.MonoAdaptiveFixedPointSolver_swigregister
MonoAdaptiveFixedPointSolver_swigregister(MonoAdaptiveFixedPointSolver)

class MonoAdaptiveJacobian(_object):
    """Proxy of C++ MonoAdaptiveJacobian class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveJacobian, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveJacobian, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, MonoAdaptiveTimeSlab timeslab, bool implicit, bool piecewise) -> MonoAdaptiveJacobian"""
        this = _dolfin.new_MonoAdaptiveJacobian(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveJacobian
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.MonoAdaptiveJacobian_size(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.MonoAdaptiveJacobian_mult(*args)

MonoAdaptiveJacobian_swigregister = _dolfin.MonoAdaptiveJacobian_swigregister
MonoAdaptiveJacobian_swigregister(MonoAdaptiveJacobian)

class MonoAdaptiveNewtonSolver(_object):
    """Proxy of C++ MonoAdaptiveNewtonSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveNewtonSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, MonoAdaptiveTimeSlab timeslab, bool implicit=False) -> MonoAdaptiveNewtonSolver
        __init__(self, MonoAdaptiveTimeSlab timeslab) -> MonoAdaptiveNewtonSolver
        """
        this = _dolfin.new_MonoAdaptiveNewtonSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveNewtonSolver
    __del__ = lambda self : None;
MonoAdaptiveNewtonSolver_swigregister = _dolfin.MonoAdaptiveNewtonSolver_swigregister
MonoAdaptiveNewtonSolver_swigregister(MonoAdaptiveNewtonSolver)

class MonoAdaptiveTimeSlab(_object):
    """Proxy of C++ MonoAdaptiveTimeSlab class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveTimeSlab, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ODE ode) -> MonoAdaptiveTimeSlab"""
        this = _dolfin.new_MonoAdaptiveTimeSlab(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveTimeSlab
    __del__ = lambda self : None;
    def build(*args):
        """build(self, real a, real b) -> real"""
        return _dolfin.MonoAdaptiveTimeSlab_build(*args)

    def solve(*args):
        """solve(self) -> bool"""
        return _dolfin.MonoAdaptiveTimeSlab_solve(*args)

    def check(*args):
        """check(self, bool first) -> bool"""
        return _dolfin.MonoAdaptiveTimeSlab_check(*args)

    def shift(*args):
        """shift(self) -> bool"""
        return _dolfin.MonoAdaptiveTimeSlab_shift(*args)

    def sample(*args):
        """sample(self, real t)"""
        return _dolfin.MonoAdaptiveTimeSlab_sample(*args)

    def usample(*args):
        """usample(self, uint i, real t) -> real"""
        return _dolfin.MonoAdaptiveTimeSlab_usample(*args)

    def ksample(*args):
        """ksample(self, uint i, real t) -> real"""
        return _dolfin.MonoAdaptiveTimeSlab_ksample(*args)

    def rsample(*args):
        """rsample(self, uint i, real t) -> real"""
        return _dolfin.MonoAdaptiveTimeSlab_rsample(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MonoAdaptiveTimeSlab_disp(*args)

MonoAdaptiveTimeSlab_swigregister = _dolfin.MonoAdaptiveTimeSlab_swigregister
MonoAdaptiveTimeSlab_swigregister(MonoAdaptiveTimeSlab)

class MonoAdaptivity(_object):
    """Proxy of C++ MonoAdaptivity class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptivity, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ODE ode, Method method) -> MonoAdaptivity"""
        this = _dolfin.new_MonoAdaptivity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptivity
    __del__ = lambda self : None;
    def timestep(*args):
        """timestep(self) -> real"""
        return _dolfin.MonoAdaptivity_timestep(*args)

    def update(*args):
        """update(self, real k0, real r, Method method, real t, bool first)"""
        return _dolfin.MonoAdaptivity_update(*args)

MonoAdaptivity_swigregister = _dolfin.MonoAdaptivity_swigregister
MonoAdaptivity_swigregister(MonoAdaptivity)

class MultiAdaptiveFixedPointSolver(_object):
    """Proxy of C++ MultiAdaptiveFixedPointSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveFixedPointSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, MultiAdaptiveTimeSlab timeslab) -> MultiAdaptiveFixedPointSolver"""
        this = _dolfin.new_MultiAdaptiveFixedPointSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveFixedPointSolver
    __del__ = lambda self : None;
MultiAdaptiveFixedPointSolver_swigregister = _dolfin.MultiAdaptiveFixedPointSolver_swigregister
MultiAdaptiveFixedPointSolver_swigregister(MultiAdaptiveFixedPointSolver)

class MultiAdaptivePreconditioner(uBlasPreconditioner):
    """Proxy of C++ MultiAdaptivePreconditioner class"""
    __swig_setmethods__ = {}
    for _s in [uBlasPreconditioner]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivePreconditioner, name, value)
    __swig_getmethods__ = {}
    for _s in [uBlasPreconditioner]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivePreconditioner, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, MultiAdaptiveTimeSlab timeslab, Method method) -> MultiAdaptivePreconditioner"""
        this = _dolfin.new_MultiAdaptivePreconditioner(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptivePreconditioner
    __del__ = lambda self : None;
    def solve(*args):
        """solve(self, uBlasVector x, uBlasVector b)"""
        return _dolfin.MultiAdaptivePreconditioner_solve(*args)

MultiAdaptivePreconditioner_swigregister = _dolfin.MultiAdaptivePreconditioner_swigregister
MultiAdaptivePreconditioner_swigregister(MultiAdaptivePreconditioner)

class MultiAdaptiveNewtonSolver(_object):
    """Proxy of C++ MultiAdaptiveNewtonSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveNewtonSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, MultiAdaptiveTimeSlab timeslab) -> MultiAdaptiveNewtonSolver"""
        this = _dolfin.new_MultiAdaptiveNewtonSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveNewtonSolver
    __del__ = lambda self : None;
MultiAdaptiveNewtonSolver_swigregister = _dolfin.MultiAdaptiveNewtonSolver_swigregister
MultiAdaptiveNewtonSolver_swigregister(MultiAdaptiveNewtonSolver)

class MultiAdaptiveTimeSlab(_object):
    """Proxy of C++ MultiAdaptiveTimeSlab class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveTimeSlab, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ODE ode) -> MultiAdaptiveTimeSlab"""
        this = _dolfin.new_MultiAdaptiveTimeSlab(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveTimeSlab
    __del__ = lambda self : None;
    def build(*args):
        """build(self, real a, real b) -> real"""
        return _dolfin.MultiAdaptiveTimeSlab_build(*args)

    def solve(*args):
        """solve(self) -> bool"""
        return _dolfin.MultiAdaptiveTimeSlab_solve(*args)

    def check(*args):
        """check(self, bool first) -> bool"""
        return _dolfin.MultiAdaptiveTimeSlab_check(*args)

    def shift(*args):
        """shift(self) -> bool"""
        return _dolfin.MultiAdaptiveTimeSlab_shift(*args)

    def reset(*args):
        """reset(self)"""
        return _dolfin.MultiAdaptiveTimeSlab_reset(*args)

    def sample(*args):
        """sample(self, real t)"""
        return _dolfin.MultiAdaptiveTimeSlab_sample(*args)

    def usample(*args):
        """usample(self, uint i, real t) -> real"""
        return _dolfin.MultiAdaptiveTimeSlab_usample(*args)

    def ksample(*args):
        """ksample(self, uint i, real t) -> real"""
        return _dolfin.MultiAdaptiveTimeSlab_ksample(*args)

    def rsample(*args):
        """rsample(self, uint i, real t) -> real"""
        return _dolfin.MultiAdaptiveTimeSlab_rsample(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.MultiAdaptiveTimeSlab_disp(*args)

MultiAdaptiveTimeSlab_swigregister = _dolfin.MultiAdaptiveTimeSlab_swigregister
MultiAdaptiveTimeSlab_swigregister(MultiAdaptiveTimeSlab)

class MultiAdaptivity(_object):
    """Proxy of C++ MultiAdaptivity class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivity, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ODE ode, Method method) -> MultiAdaptivity"""
        this = _dolfin.new_MultiAdaptivity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptivity
    __del__ = lambda self : None;
    def timestep(*args):
        """timestep(self, uint i) -> real"""
        return _dolfin.MultiAdaptivity_timestep(*args)

    def residual(*args):
        """residual(self, uint i) -> real"""
        return _dolfin.MultiAdaptivity_residual(*args)

    def update(*args):
        """update(self, MultiAdaptiveTimeSlab ts, real t, bool first)"""
        return _dolfin.MultiAdaptivity_update(*args)

MultiAdaptivity_swigregister = _dolfin.MultiAdaptivity_swigregister
MultiAdaptivity_swigregister(MultiAdaptivity)

class ODE(_object):
    """Proxy of C++ ODE class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODE, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint N, real T) -> ODE"""
        if self.__class__ == ODE:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_ODE(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_ODE
    __del__ = lambda self : None;
    def u0(*args):
        """u0(self, uBlasVector u)"""
        return _dolfin.ODE_u0(*args)

    def f(*args):
        """
        f(self, uBlasVector u, real t, uBlasVector y)
        f(self, uBlasVector u, real t, uint i) -> real
        """
        return _dolfin.ODE_f(*args)

    def M(*args):
        """M(self, uBlasVector x, uBlasVector y, uBlasVector u, real t)"""
        return _dolfin.ODE_M(*args)

    def J(*args):
        """J(self, uBlasVector x, uBlasVector y, uBlasVector u, real t)"""
        return _dolfin.ODE_J(*args)

    def dfdu(*args):
        """dfdu(self, uBlasVector u, real t, uint i, uint j) -> real"""
        return _dolfin.ODE_dfdu(*args)

    def timestep(*args):
        """
        timestep(self, real t, real k0) -> real
        timestep(self, real t, uint i, real k0) -> real
        """
        return _dolfin.ODE_timestep(*args)

    def update(*args):
        """update(self, uBlasVector u, real t, bool end) -> bool"""
        return _dolfin.ODE_update(*args)

    def save(*args):
        """save(self, Sample sample)"""
        return _dolfin.ODE_save(*args)

    def sparse(*args):
        """
        sparse(self)
        sparse(self, uBlasSparseMatrix A)
        """
        return _dolfin.ODE_sparse(*args)

    def size(*args):
        """size(self) -> uint"""
        return _dolfin.ODE_size(*args)

    def endtime(*args):
        """endtime(self) -> real"""
        return _dolfin.ODE_endtime(*args)

    def solve(*args):
        """solve(self)"""
        return _dolfin.ODE_solve(*args)

    def __disown__(self):
        self.this.disown()
        _dolfin.disown_ODE(self)
        return weakref_proxy(self)
ODE_swigregister = _dolfin.ODE_swigregister
ODE_swigregister(ODE)

class ODESolver(_object):
    """Proxy of C++ ODESolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODESolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODESolver, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def solve(*args):
        """solve(ODE ode)"""
        return _dolfin.ODESolver_solve(*args)

    if _newclass:solve = staticmethod(solve)
    __swig_getmethods__["solve"] = lambda x: solve
ODESolver_swigregister = _dolfin.ODESolver_swigregister
ODESolver_swigregister(ODESolver)

def ODESolver_solve(*args):
  """ODESolver_solve(ODE ode)"""
  return _dolfin.ODESolver_solve(*args)

class Partition(_object):
    """Proxy of C++ Partition class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Partition, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Partition, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, uint N) -> Partition"""
        this = _dolfin.new_Partition(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Partition
    __del__ = lambda self : None;
    def size(*args):
        """size(self) -> uint"""
        return _dolfin.Partition_size(*args)

    def index(*args):
        """index(self, uint pos) -> uint"""
        return _dolfin.Partition_index(*args)

    def update(*args):
        """
        update(self, uint offset, uint end, MultiAdaptivity adaptivity, 
            real K) -> real
        """
        return _dolfin.Partition_update(*args)

    def debug(*args):
        """debug(self, uint offset, uint end)"""
        return _dolfin.Partition_debug(*args)

Partition_swigregister = _dolfin.Partition_swigregister
Partition_swigregister(Partition)

class Sample(Variable):
    """Proxy of C++ Sample class"""
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Sample, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Sample, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, TimeSlab timeslab, real t, string name, string label) -> Sample"""
        this = _dolfin.new_Sample(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Sample
    __del__ = lambda self : None;
    def size(*args):
        """size(self) -> uint"""
        return _dolfin.Sample_size(*args)

    def t(*args):
        """t(self) -> real"""
        return _dolfin.Sample_t(*args)

    def u(*args):
        """u(self, uint index) -> real"""
        return _dolfin.Sample_u(*args)

    def k(*args):
        """k(self, uint index) -> real"""
        return _dolfin.Sample_k(*args)

    def r(*args):
        """r(self, uint index) -> real"""
        return _dolfin.Sample_r(*args)

Sample_swigregister = _dolfin.Sample_swigregister
Sample_swigregister(Sample)

class TimeSlab(_object):
    """Proxy of C++ TimeSlab class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlab, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_TimeSlab
    __del__ = lambda self : None;
    def build(*args):
        """build(self, real a, real b) -> real"""
        return _dolfin.TimeSlab_build(*args)

    def solve(*args):
        """solve(self) -> bool"""
        return _dolfin.TimeSlab_solve(*args)

    def check(*args):
        """check(self, bool first) -> bool"""
        return _dolfin.TimeSlab_check(*args)

    def shift(*args):
        """shift(self) -> bool"""
        return _dolfin.TimeSlab_shift(*args)

    def sample(*args):
        """sample(self, real t)"""
        return _dolfin.TimeSlab_sample(*args)

    def size(*args):
        """size(self) -> uint"""
        return _dolfin.TimeSlab_size(*args)

    def starttime(*args):
        """starttime(self) -> real"""
        return _dolfin.TimeSlab_starttime(*args)

    def endtime(*args):
        """endtime(self) -> real"""
        return _dolfin.TimeSlab_endtime(*args)

    def length(*args):
        """length(self) -> real"""
        return _dolfin.TimeSlab_length(*args)

    def usample(*args):
        """usample(self, uint i, real t) -> real"""
        return _dolfin.TimeSlab_usample(*args)

    def ksample(*args):
        """ksample(self, uint i, real t) -> real"""
        return _dolfin.TimeSlab_ksample(*args)

    def rsample(*args):
        """rsample(self, uint i, real t) -> real"""
        return _dolfin.TimeSlab_rsample(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.TimeSlab_disp(*args)

TimeSlab_swigregister = _dolfin.TimeSlab_swigregister
TimeSlab_swigregister(TimeSlab)

class TimeSlabJacobian(uBlasKrylovMatrix):
    """Proxy of C++ TimeSlabJacobian class"""
    __swig_setmethods__ = {}
    for _s in [uBlasKrylovMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlabJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [uBlasKrylovMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlabJacobian, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_TimeSlabJacobian
    __del__ = lambda self : None;
    def size(*args):
        """size(self, uint dim) -> uint"""
        return _dolfin.TimeSlabJacobian_size(*args)

    def mult(*args):
        """mult(self, uBlasVector x, uBlasVector y)"""
        return _dolfin.TimeSlabJacobian_mult(*args)

    def init(*args):
        """init(self)"""
        return _dolfin.TimeSlabJacobian_init(*args)

    def update(*args):
        """update(self)"""
        return _dolfin.TimeSlabJacobian_update(*args)

    def matrix(*args):
        """matrix(self) -> uBlasDenseMatrix"""
        return _dolfin.TimeSlabJacobian_matrix(*args)

TimeSlabJacobian_swigregister = _dolfin.TimeSlabJacobian_swigregister
TimeSlabJacobian_swigregister(TimeSlabJacobian)

class TimeStepper(_object):
    """Proxy of C++ TimeStepper class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeStepper, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeStepper, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ODE ode) -> TimeStepper"""
        this = _dolfin.new_TimeStepper(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_TimeStepper
    __del__ = lambda self : None;
    def solve(*args):
        """solve(ODE ode)"""
        return _dolfin.TimeStepper_solve(*args)

    if _newclass:solve = staticmethod(solve)
    __swig_getmethods__["solve"] = lambda x: solve
    def step(*args):
        """step(self) -> real"""
        return _dolfin.TimeStepper_step(*args)

    def finished(*args):
        """finished(self) -> bool"""
        return _dolfin.TimeStepper_finished(*args)

TimeStepper_swigregister = _dolfin.TimeStepper_swigregister
TimeStepper_swigregister(TimeStepper)

def TimeStepper_solve(*args):
  """TimeStepper_solve(ODE ode)"""
  return _dolfin.TimeStepper_solve(*args)

class cGqMethod(Method):
    """Proxy of C++ cGqMethod class"""
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, cGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, cGqMethod, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, unsigned int q) -> cGqMethod"""
        this = _dolfin.new_cGqMethod(*args)
        try: self.this.append(this)
        except: self.this = this
    def ueval(*args):
        """
        ueval(self, real x0, real values, real tau) -> real
        ueval(self, real x0, uBlasVector values, uint offset, real tau) -> real
        ueval(self, real x0, real values, uint i) -> real
        """
        return _dolfin.cGqMethod_ueval(*args)

    def residual(*args):
        """
        residual(self, real x0, real values, real f, real k) -> real
        residual(self, real x0, uBlasVector values, uint offset, real f, real k) -> real
        """
        return _dolfin.cGqMethod_residual(*args)

    def timestep(*args):
        """timestep(self, real r, real tol, real k0, real kmax) -> real"""
        return _dolfin.cGqMethod_timestep(*args)

    def error(*args):
        """error(self, real k, real r) -> real"""
        return _dolfin.cGqMethod_error(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.cGqMethod_disp(*args)

cGqMethod_swigregister = _dolfin.cGqMethod_swigregister
cGqMethod_swigregister(cGqMethod)

class dGqMethod(Method):
    """Proxy of C++ dGqMethod class"""
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, dGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, dGqMethod, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, unsigned int q) -> dGqMethod"""
        this = _dolfin.new_dGqMethod(*args)
        try: self.this.append(this)
        except: self.this = this
    def ueval(*args):
        """
        ueval(self, real x0, real values, real tau) -> real
        ueval(self, real x0, uBlasVector values, uint offset, real tau) -> real
        ueval(self, real x0, real values, uint i) -> real
        """
        return _dolfin.dGqMethod_ueval(*args)

    def residual(*args):
        """
        residual(self, real x0, real values, real f, real k) -> real
        residual(self, real x0, uBlasVector values, uint offset, real f, real k) -> real
        """
        return _dolfin.dGqMethod_residual(*args)

    def timestep(*args):
        """timestep(self, real r, real tol, real k0, real kmax) -> real"""
        return _dolfin.dGqMethod_timestep(*args)

    def error(*args):
        """error(self, real k, real r) -> real"""
        return _dolfin.dGqMethod_error(*args)

    def disp(*args):
        """disp(self)"""
        return _dolfin.dGqMethod_disp(*args)

dGqMethod_swigregister = _dolfin.dGqMethod_swigregister
dGqMethod_swigregister(dGqMethod)

class TimeDependentPDE(_object):
    """Proxy of C++ TimeDependentPDE class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependentPDE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependentPDE, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, BilinearForm a, LinearForm L, Mesh mesh, BoundaryCondition bc, 
            int N, real k, real T) -> TimeDependentPDE
        """
        if self.__class__ == TimeDependentPDE:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_TimeDependentPDE(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_TimeDependentPDE
    __del__ = lambda self : None;
    def solve(*args):
        """solve(self, Function u) -> uint"""
        return _dolfin.TimeDependentPDE_solve(*args)

    def fu(*args):
        """fu(self, Vector x, Vector dotx, real t)"""
        return _dolfin.TimeDependentPDE_fu(*args)

    def init(*args):
        """init(self, Function U)"""
        return _dolfin.TimeDependentPDE_init(*args)

    def save(*args):
        """save(self, Function U, real t)"""
        return _dolfin.TimeDependentPDE_save(*args)

    def preparestep(*args):
        """preparestep(self)"""
        return _dolfin.TimeDependentPDE_preparestep(*args)

    def prepareiteration(*args):
        """prepareiteration(self)"""
        return _dolfin.TimeDependentPDE_prepareiteration(*args)

    def elementdim(*args):
        """elementdim(self) -> uint"""
        return _dolfin.TimeDependentPDE_elementdim(*args)

    def a(*args):
        """a(self) -> BilinearForm"""
        return _dolfin.TimeDependentPDE_a(*args)

    def L(*args):
        """L(self) -> LinearForm"""
        return _dolfin.TimeDependentPDE_L(*args)

    def mesh(*args):
        """mesh(self) -> Mesh"""
        return _dolfin.TimeDependentPDE_mesh(*args)

    def bc(*args):
        """bc(self) -> BoundaryCondition"""
        return _dolfin.TimeDependentPDE_bc(*args)

    __swig_setmethods__["x"] = _dolfin.TimeDependentPDE_x_set
    __swig_getmethods__["x"] = _dolfin.TimeDependentPDE_x_get
    if _newclass:x = property(_dolfin.TimeDependentPDE_x_get, _dolfin.TimeDependentPDE_x_set)
    __swig_setmethods__["dotx"] = _dolfin.TimeDependentPDE_dotx_set
    __swig_getmethods__["dotx"] = _dolfin.TimeDependentPDE_dotx_get
    if _newclass:dotx = property(_dolfin.TimeDependentPDE_dotx_get, _dolfin.TimeDependentPDE_dotx_set)
    __swig_setmethods__["k"] = _dolfin.TimeDependentPDE_k_set
    __swig_getmethods__["k"] = _dolfin.TimeDependentPDE_k_get
    if _newclass:k = property(_dolfin.TimeDependentPDE_k_get, _dolfin.TimeDependentPDE_k_set)
    def __disown__(self):
        self.this.disown()
        _dolfin.disown_TimeDependentPDE(self)
        return weakref_proxy(self)
TimeDependentPDE_swigregister = _dolfin.TimeDependentPDE_swigregister
TimeDependentPDE_swigregister(TimeDependentPDE)

class TimeDependentODE(ODE):
    """Proxy of C++ TimeDependentODE class"""
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependentODE, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependentODE, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, TimeDependentPDE pde, int N, real T) -> TimeDependentODE"""
        this = _dolfin.new_TimeDependentODE(*args)
        try: self.this.append(this)
        except: self.this = this
    def u0(*args):
        """u0(self, uBlasVector u)"""
        return _dolfin.TimeDependentODE_u0(*args)

    def timestep(*args):
        """timestep(self, real t, real k0) -> real"""
        return _dolfin.TimeDependentODE_timestep(*args)

    def f(*args):
        """
        f(self, uBlasVector u, real t, uBlasVector y)
        f(self, uBlasVector u, real t, uint i) -> real
        f(self, uBlasVector u, real t, uBlasVector y)
        """
        return _dolfin.TimeDependentODE_f(*args)

    def update(*args):
        """update(self, uBlasVector u, real t, bool end) -> bool"""
        return _dolfin.TimeDependentODE_update(*args)

TimeDependentODE_swigregister = _dolfin.TimeDependentODE_swigregister
TimeDependentODE_swigregister(TimeDependentODE)

class FiniteElement(_object):
    """Proxy of C++ FiniteElement class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FiniteElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FiniteElement, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfin.delete_FiniteElement
    __del__ = lambda self : None;
    def spacedim(*args):
        """spacedim(self) -> unsigned int"""
        return _dolfin.FiniteElement_spacedim(*args)

    def shapedim(*args):
        """shapedim(self) -> unsigned int"""
        return _dolfin.FiniteElement_shapedim(*args)

    def tensordim(*args):
        """tensordim(self, unsigned int i) -> unsigned int"""
        return _dolfin.FiniteElement_tensordim(*args)

    def elementdim(*args):
        """elementdim(self) -> unsigned int"""
        return _dolfin.FiniteElement_elementdim(*args)

    def rank(*args):
        """rank(self) -> unsigned int"""
        return _dolfin.FiniteElement_rank(*args)

    def nodemap(*args):
        """nodemap(self, int nodes, Cell cell, Mesh mesh)"""
        return _dolfin.FiniteElement_nodemap(*args)

    def pointmap(*args):
        """pointmap(self, Point points, unsigned int components, AffineMap map)"""
        return _dolfin.FiniteElement_pointmap(*args)

    def vertexeval(*args):
        """vertexeval(self, uint vertex_nodes, unsigned int vertex, Mesh mesh)"""
        return _dolfin.FiniteElement_vertexeval(*args)

    def spec(*args):
        """spec(self) -> FiniteElementSpec"""
        return _dolfin.FiniteElement_spec(*args)

    def makeElement(*args):
        """
        makeElement(FiniteElementSpec spec) -> FiniteElement
        makeElement(string type, string shape, uint degree, uint vectordim=0) -> FiniteElement
        makeElement(string type, string shape, uint degree) -> FiniteElement
        """
        return _dolfin.FiniteElement_makeElement(*args)

    if _newclass:makeElement = staticmethod(makeElement)
    __swig_getmethods__["makeElement"] = lambda x: makeElement
    def disp(*args):
        """disp(self)"""
        return _dolfin.FiniteElement_disp(*args)

FiniteElement_swigregister = _dolfin.FiniteElement_swigregister
FiniteElement_swigregister(FiniteElement)

def FiniteElement_makeElement(*args):
  """
    makeElement(FiniteElementSpec spec) -> FiniteElement
    makeElement(string type, string shape, uint degree, uint vectordim=0) -> FiniteElement
    FiniteElement_makeElement(string type, string shape, uint degree) -> FiniteElement
    """
  return _dolfin.FiniteElement_makeElement(*args)

class AffineMap(_object):
    """Proxy of C++ AffineMap class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, AffineMap, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, AffineMap, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> AffineMap"""
        this = _dolfin.new_AffineMap(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_AffineMap
    __del__ = lambda self : None;
    def update(*args):
        """
        update(self, Cell cell)
        update(self, Cell cell, uint facet)
        """
        return _dolfin.AffineMap_update(*args)

    def __call__(*args):
        """
        __call__(self, real X, real Y) -> Point
        __call__(self, real X, real Y, real Z) -> Point
        """
        return _dolfin.AffineMap___call__(*args)

    def cell(*args):
        """cell(self) -> Cell"""
        return _dolfin.AffineMap_cell(*args)

    __swig_setmethods__["det"] = _dolfin.AffineMap_det_set
    __swig_getmethods__["det"] = _dolfin.AffineMap_det_get
    if _newclass:det = property(_dolfin.AffineMap_det_get, _dolfin.AffineMap_det_set)
    __swig_setmethods__["scaling"] = _dolfin.AffineMap_scaling_set
    __swig_getmethods__["scaling"] = _dolfin.AffineMap_scaling_get
    if _newclass:scaling = property(_dolfin.AffineMap_scaling_get, _dolfin.AffineMap_scaling_set)
    __swig_setmethods__["f00"] = _dolfin.AffineMap_f00_set
    __swig_getmethods__["f00"] = _dolfin.AffineMap_f00_get
    if _newclass:f00 = property(_dolfin.AffineMap_f00_get, _dolfin.AffineMap_f00_set)
    __swig_setmethods__["f01"] = _dolfin.AffineMap_f01_set
    __swig_getmethods__["f01"] = _dolfin.AffineMap_f01_get
    if _newclass:f01 = property(_dolfin.AffineMap_f01_get, _dolfin.AffineMap_f01_set)
    __swig_setmethods__["f02"] = _dolfin.AffineMap_f02_set
    __swig_getmethods__["f02"] = _dolfin.AffineMap_f02_get
    if _newclass:f02 = property(_dolfin.AffineMap_f02_get, _dolfin.AffineMap_f02_set)
    __swig_setmethods__["f10"] = _dolfin.AffineMap_f10_set
    __swig_getmethods__["f10"] = _dolfin.AffineMap_f10_get
    if _newclass:f10 = property(_dolfin.AffineMap_f10_get, _dolfin.AffineMap_f10_set)
    __swig_setmethods__["f11"] = _dolfin.AffineMap_f11_set
    __swig_getmethods__["f11"] = _dolfin.AffineMap_f11_get
    if _newclass:f11 = property(_dolfin.AffineMap_f11_get, _dolfin.AffineMap_f11_set)
    __swig_setmethods__["f12"] = _dolfin.AffineMap_f12_set
    __swig_getmethods__["f12"] = _dolfin.AffineMap_f12_get
    if _newclass:f12 = property(_dolfin.AffineMap_f12_get, _dolfin.AffineMap_f12_set)
    __swig_setmethods__["f20"] = _dolfin.AffineMap_f20_set
    __swig_getmethods__["f20"] = _dolfin.AffineMap_f20_get
    if _newclass:f20 = property(_dolfin.AffineMap_f20_get, _dolfin.AffineMap_f20_set)
    __swig_setmethods__["f21"] = _dolfin.AffineMap_f21_set
    __swig_getmethods__["f21"] = _dolfin.AffineMap_f21_get
    if _newclass:f21 = property(_dolfin.AffineMap_f21_get, _dolfin.AffineMap_f21_set)
    __swig_setmethods__["f22"] = _dolfin.AffineMap_f22_set
    __swig_getmethods__["f22"] = _dolfin.AffineMap_f22_get
    if _newclass:f22 = property(_dolfin.AffineMap_f22_get, _dolfin.AffineMap_f22_set)
    __swig_setmethods__["g00"] = _dolfin.AffineMap_g00_set
    __swig_getmethods__["g00"] = _dolfin.AffineMap_g00_get
    if _newclass:g00 = property(_dolfin.AffineMap_g00_get, _dolfin.AffineMap_g00_set)
    __swig_setmethods__["g01"] = _dolfin.AffineMap_g01_set
    __swig_getmethods__["g01"] = _dolfin.AffineMap_g01_get
    if _newclass:g01 = property(_dolfin.AffineMap_g01_get, _dolfin.AffineMap_g01_set)
    __swig_setmethods__["g02"] = _dolfin.AffineMap_g02_set
    __swig_getmethods__["g02"] = _dolfin.AffineMap_g02_get
    if _newclass:g02 = property(_dolfin.AffineMap_g02_get, _dolfin.AffineMap_g02_set)
    __swig_setmethods__["g10"] = _dolfin.AffineMap_g10_set
    __swig_getmethods__["g10"] = _dolfin.AffineMap_g10_get
    if _newclass:g10 = property(_dolfin.AffineMap_g10_get, _dolfin.AffineMap_g10_set)
    __swig_setmethods__["g11"] = _dolfin.AffineMap_g11_set
    __swig_getmethods__["g11"] = _dolfin.AffineMap_g11_get
    if _newclass:g11 = property(_dolfin.AffineMap_g11_get, _dolfin.AffineMap_g11_set)
    __swig_setmethods__["g12"] = _dolfin.AffineMap_g12_set
    __swig_getmethods__["g12"] = _dolfin.AffineMap_g12_get
    if _newclass:g12 = property(_dolfin.AffineMap_g12_get, _dolfin.AffineMap_g12_set)
    __swig_setmethods__["g20"] = _dolfin.AffineMap_g20_set
    __swig_getmethods__["g20"] = _dolfin.AffineMap_g20_get
    if _newclass:g20 = property(_dolfin.AffineMap_g20_get, _dolfin.AffineMap_g20_set)
    __swig_setmethods__["g21"] = _dolfin.AffineMap_g21_set
    __swig_getmethods__["g21"] = _dolfin.AffineMap_g21_get
    if _newclass:g21 = property(_dolfin.AffineMap_g21_get, _dolfin.AffineMap_g21_set)
    __swig_setmethods__["g22"] = _dolfin.AffineMap_g22_set
    __swig_getmethods__["g22"] = _dolfin.AffineMap_g22_get
    if _newclass:g22 = property(_dolfin.AffineMap_g22_get, _dolfin.AffineMap_g22_set)
AffineMap_swigregister = _dolfin.AffineMap_swigregister
AffineMap_swigregister(AffineMap)

class BoundaryValue(_object):
    """Proxy of C++ BoundaryValue class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryValue, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryValue, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> BoundaryValue"""
        this = _dolfin.new_BoundaryValue(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_BoundaryValue
    __del__ = lambda self : None;
    def set(*args):
        """set(self, real value)"""
        return _dolfin.BoundaryValue_set(*args)

    def reset(*args):
        """reset(self)"""
        return _dolfin.BoundaryValue_reset(*args)

BoundaryValue_swigregister = _dolfin.BoundaryValue_swigregister
BoundaryValue_swigregister(BoundaryValue)

class BoundaryCondition(TimeDependent):
    """Proxy of C++ BoundaryCondition class"""
    __swig_setmethods__ = {}
    for _s in [TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryCondition, name, value)
    __swig_getmethods__ = {}
    for _s in [TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryCondition, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> BoundaryCondition"""
        if self.__class__ == BoundaryCondition:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_BoundaryCondition(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_BoundaryCondition
    __del__ = lambda self : None;
    def eval(*args):
        """eval(self, BoundaryValue value, Point p, uint i)"""
        return _dolfin.BoundaryCondition_eval(*args)

    def __disown__(self):
        self.this.disown()
        _dolfin.disown_BoundaryCondition(self)
        return weakref_proxy(self)
BoundaryCondition_swigregister = _dolfin.BoundaryCondition_swigregister
BoundaryCondition_swigregister(BoundaryCondition)

class FEM(_object):
    """Proxy of C++ FEM class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FEM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FEM, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    def assemble(*args):
        """
        assemble(BilinearForm a, LinearForm L, GenericMatrix A, GenericVector b, 
            Mesh mesh)
        assemble(BilinearForm a, LinearForm L, GenericMatrix A, GenericVector b, 
            Mesh mesh, BoundaryCondition bc)
        assemble(BilinearForm a, GenericMatrix A, Mesh mesh)
        assemble(LinearForm L, GenericVector b, Mesh mesh)
        """
        return _dolfin.FEM_assemble(*args)

    if _newclass:assemble = staticmethod(assemble)
    __swig_getmethods__["assemble"] = lambda x: assemble
    def applyBC(*args):
        """
        applyBC(GenericMatrix A, GenericVector b, Mesh mesh, FiniteElement element, 
            BoundaryCondition bc)
        applyBC(GenericMatrix A, Mesh mesh, FiniteElement element, 
            BoundaryCondition bc)
        applyBC(GenericVector b, Mesh mesh, FiniteElement element, 
            BoundaryCondition bc)
        """
        return _dolfin.FEM_applyBC(*args)

    if _newclass:applyBC = staticmethod(applyBC)
    __swig_getmethods__["applyBC"] = lambda x: applyBC
    def assembleResidualBC(*args):
        """
        assembleResidualBC(GenericMatrix A, GenericVector b, GenericVector x, 
            Mesh mesh, FiniteElement element, BoundaryCondition bc)
        assembleResidualBC(GenericVector b, GenericVector x, Mesh mesh, FiniteElement element, 
            BoundaryCondition bc)
        """
        return _dolfin.FEM_assembleResidualBC(*args)

    if _newclass:assembleResidualBC = staticmethod(assembleResidualBC)
    __swig_getmethods__["assembleResidualBC"] = lambda x: assembleResidualBC
    def size(*args):
        """size(Mesh mesh, FiniteElement element) -> uint"""
        return _dolfin.FEM_size(*args)

    if _newclass:size = staticmethod(size)
    __swig_getmethods__["size"] = lambda x: size
    def disp(*args):
        """disp(Mesh mesh, FiniteElement element)"""
        return _dolfin.FEM_disp(*args)

    if _newclass:disp = staticmethod(disp)
    __swig_getmethods__["disp"] = lambda x: disp
    def lump(*args):
        """lump(uBlasSparseMatrix M, uBlasVector m)"""
        return _dolfin.FEM_lump(*args)

    if _newclass:lump = staticmethod(lump)
    __swig_getmethods__["lump"] = lambda x: lump
FEM_swigregister = _dolfin.FEM_swigregister
FEM_swigregister(FEM)

def FEM_assemble(*args):
  """
    assemble(BilinearForm a, LinearForm L, GenericMatrix A, GenericVector b, 
        Mesh mesh)
    assemble(BilinearForm a, LinearForm L, GenericMatrix A, GenericVector b, 
        Mesh mesh, BoundaryCondition bc)
    assemble(BilinearForm a, GenericMatrix A, Mesh mesh)
    FEM_assemble(LinearForm L, GenericVector b, Mesh mesh)
    """
  return _dolfin.FEM_assemble(*args)

def FEM_applyBC(*args):
  """
    applyBC(GenericMatrix A, GenericVector b, Mesh mesh, FiniteElement element, 
        BoundaryCondition bc)
    applyBC(GenericMatrix A, Mesh mesh, FiniteElement element, 
        BoundaryCondition bc)
    FEM_applyBC(GenericVector b, Mesh mesh, FiniteElement element, 
        BoundaryCondition bc)
    """
  return _dolfin.FEM_applyBC(*args)

def FEM_assembleResidualBC(*args):
  """
    assembleResidualBC(GenericMatrix A, GenericVector b, GenericVector x, 
        Mesh mesh, FiniteElement element, BoundaryCondition bc)
    FEM_assembleResidualBC(GenericVector b, GenericVector x, Mesh mesh, FiniteElement element, 
        BoundaryCondition bc)
    """
  return _dolfin.FEM_assembleResidualBC(*args)

def FEM_size(*args):
  """FEM_size(Mesh mesh, FiniteElement element) -> uint"""
  return _dolfin.FEM_size(*args)

def FEM_disp(*args):
  """FEM_disp(Mesh mesh, FiniteElement element)"""
  return _dolfin.FEM_disp(*args)

def FEM_lump(*args):
  """FEM_lump(uBlasSparseMatrix M, uBlasVector m)"""
  return _dolfin.FEM_lump(*args)


def get(*args):
  """get(string name) -> Parameter"""
  return _dolfin.get(*args)

def load_parameters(*args):
  """load_parameters(string filename)"""
  return _dolfin.load_parameters(*args)


def set(*args):
  """
    set(string name, real val)
    set(string name, int val)
    set(string name, bool val)
    set(string name, string val)
    """
  return _dolfin.set(*args)

