# This file was created automatically by SWIG 1.3.27.
# Don't modify this file, modify the SWIG interface instead.

import _dolfin

# This file is compatible with both classic and new-style classes.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name) or (name == "thisown"):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

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



new_realArray = _dolfin.new_realArray

delete_realArray = _dolfin.delete_realArray

realArray_getitem = _dolfin.realArray_getitem

realArray_setitem = _dolfin.realArray_setitem

new_intArray = _dolfin.new_intArray

delete_intArray = _dolfin.delete_intArray

intArray_getitem = _dolfin.intArray_getitem

intArray_setitem = _dolfin.intArray_setitem

dolfin_init = _dolfin.dolfin_init

sqr = _dolfin.sqr

rand = _dolfin.rand

seed = _dolfin.seed
class TimeDependent(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependent, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeDependent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, TimeDependent, 'this', _dolfin.new_TimeDependent(*args))
        _swig_setattr(self, TimeDependent, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_TimeDependent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def sync(*args): return _dolfin.TimeDependent_sync(*args)
    def time(*args): return _dolfin.TimeDependent_time(*args)

class TimeDependentPtr(TimeDependent):
    def __init__(self, this):
        _swig_setattr(self, TimeDependent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeDependent, 'thisown', 0)
        self.__class__ = TimeDependent
_dolfin.TimeDependent_swigregister(TimeDependentPtr)

class Variable(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Variable, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Variable, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Variable instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Variable, 'this', _dolfin.new_Variable(*args))
        _swig_setattr(self, Variable, 'thisown', 1)
    def rename(*args): return _dolfin.Variable_rename(*args)
    def name(*args): return _dolfin.Variable_name(*args)
    def label(*args): return _dolfin.Variable_label(*args)
    def number(*args): return _dolfin.Variable_number(*args)

class VariablePtr(Variable):
    def __init__(self, this):
        _swig_setattr(self, Variable, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Variable, 'thisown', 0)
        self.__class__ = Variable
_dolfin.Variable_swigregister(VariablePtr)


suffix = _dolfin.suffix

remove_newline = _dolfin.remove_newline

length = _dolfin.length

date = _dolfin.date

delay = _dolfin.delay

tic = _dolfin.tic

toc = _dolfin.toc

tocd = _dolfin.tocd

dolfin_update = _dolfin.dolfin_update

dolfin_quit = _dolfin.dolfin_quit

dolfin_finished = _dolfin.dolfin_finished

dolfin_segfault = _dolfin.dolfin_segfault

dolfin_output = _dolfin.dolfin_output

dolfin_log = _dolfin.dolfin_log
class Parameter(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Parameter, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Parameter, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Parameter instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    type_real = _dolfin.Parameter_type_real
    type_int = _dolfin.Parameter_type_int
    type_bool = _dolfin.Parameter_type_bool
    type_string = _dolfin.Parameter_type_string
    def __init__(self, *args):
        _swig_setattr(self, Parameter, 'this', _dolfin.new_Parameter(*args))
        _swig_setattr(self, Parameter, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Parameter):
        try:
            if self.thisown: destroy(self)
        except: pass

    def type(*args): return _dolfin.Parameter_type(*args)

class ParameterPtr(Parameter):
    def __init__(self, this):
        _swig_setattr(self, Parameter, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Parameter, 'thisown', 0)
        self.__class__ = Parameter
_dolfin.Parameter_swigregister(ParameterPtr)

dolfin_begin = _dolfin.dolfin_begin

dolfin_end = _dolfin.dolfin_end

class File(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, File, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, File, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::File instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
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
        _swig_setattr(self, File, 'this', _dolfin.new_File(*args))
        _swig_setattr(self, File, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_File):
        try:
            if self.thisown: destroy(self)
        except: pass

    def __rshift__(*args): return _dolfin.File___rshift__(*args)
    def __lshift__(*args): return _dolfin.File___lshift__(*args)

class FilePtr(File):
    def __init__(self, this):
        _swig_setattr(self, File, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, File, 'thisown', 0)
        self.__class__ = File
_dolfin.File_swigregister(FilePtr)

class Vector(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScVector instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Vector, 'this', _dolfin.new_Vector(*args))
        _swig_setattr(self, Vector, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Vector):
        try:
            if self.thisown: destroy(self)
        except: pass

    def init(*args): return _dolfin.Vector_init(*args)
    def clear(*args): return _dolfin.Vector_clear(*args)
    def size(*args): return _dolfin.Vector_size(*args)
    def vec(*args): return _dolfin.Vector_vec(*args)
    def array(*args): return _dolfin.Vector_array(*args)
    def restore(*args): return _dolfin.Vector_restore(*args)
    def axpy(*args): return _dolfin.Vector_axpy(*args)
    def div(*args): return _dolfin.Vector_div(*args)
    def mult(*args): return _dolfin.Vector_mult(*args)
    def set(*args): return _dolfin.Vector_set(*args)
    def add(*args): return _dolfin.Vector_add(*args)
    def get(*args): return _dolfin.Vector_get(*args)
    def apply(*args): return _dolfin.Vector_apply(*args)
    def zero(*args): return _dolfin.Vector_zero(*args)
    def __call__(*args): return _dolfin.Vector___call__(*args)
    def __iadd__(*args): return _dolfin.Vector___iadd__(*args)
    def __isub__(*args): return _dolfin.Vector___isub__(*args)
    def __imul__(*args): return _dolfin.Vector___imul__(*args)
    def __idiv__(*args): return _dolfin.Vector___idiv__(*args)
    def __mul__(*args): return _dolfin.Vector___mul__(*args)
    l1 = _dolfin.Vector_l1
    l2 = _dolfin.Vector_l2
    linf = _dolfin.Vector_linf
    def norm(*args): return _dolfin.Vector_norm(*args)
    def sum(*args): return _dolfin.Vector_sum(*args)
    def max(*args): return _dolfin.Vector_max(*args)
    def min(*args): return _dolfin.Vector_min(*args)
    def disp(*args): return _dolfin.Vector_disp(*args)
    def getval(*args): return _dolfin.Vector_getval(*args)
    def setval(*args): return _dolfin.Vector_setval(*args)
    def addval(*args): return _dolfin.Vector_addval(*args)
    __swig_getmethods__["createScatterer"] = lambda x: _dolfin.Vector_createScatterer
    if _newclass:createScatterer = staticmethod(_dolfin.Vector_createScatterer)
    __swig_getmethods__["gather"] = lambda x: _dolfin.Vector_gather
    if _newclass:gather = staticmethod(_dolfin.Vector_gather)
    __swig_getmethods__["scatter"] = lambda x: _dolfin.Vector_scatter
    if _newclass:scatter = staticmethod(_dolfin.Vector_scatter)
    __swig_getmethods__["fromArray"] = lambda x: _dolfin.Vector_fromArray
    if _newclass:fromArray = staticmethod(_dolfin.Vector_fromArray)
    __swig_getmethods__["toArray"] = lambda x: _dolfin.Vector_toArray
    if _newclass:toArray = staticmethod(_dolfin.Vector_toArray)

class VectorPtr(Vector):
    def __init__(self, this):
        _swig_setattr(self, Vector, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Vector, 'thisown', 0)
        self.__class__ = Vector
_dolfin.Vector_swigregister(VectorPtr)

Vector_createScatterer = _dolfin.Vector_createScatterer

Vector_gather = _dolfin.Vector_gather

Vector_scatter = _dolfin.Vector_scatter

Vector_fromArray = _dolfin.Vector_fromArray

Vector_toArray = _dolfin.Vector_toArray

class PETScVectorElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PETScVectorElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PETScVectorElement, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScVectorElement instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, PETScVectorElement, 'this', _dolfin.new_PETScVectorElement(*args))
        _swig_setattr(self, PETScVectorElement, 'thisown', 1)
    def __iadd__(*args): return _dolfin.PETScVectorElement___iadd__(*args)
    def __isub__(*args): return _dolfin.PETScVectorElement___isub__(*args)
    def __imul__(*args): return _dolfin.PETScVectorElement___imul__(*args)

class PETScVectorElementPtr(PETScVectorElement):
    def __init__(self, this):
        _swig_setattr(self, PETScVectorElement, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, PETScVectorElement, 'thisown', 0)
        self.__class__ = PETScVectorElement
_dolfin.PETScVectorElement_swigregister(PETScVectorElementPtr)

class Matrix(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Matrix, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Matrix, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScSparseMatrix instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    default_matrix = _dolfin.Matrix_default_matrix
    spooles = _dolfin.Matrix_spooles
    superlu = _dolfin.Matrix_superlu
    umfpack = _dolfin.Matrix_umfpack
    def __init__(self, *args):
        _swig_setattr(self, Matrix, 'this', _dolfin.new_Matrix(*args))
        _swig_setattr(self, Matrix, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Matrix):
        try:
            if self.thisown: destroy(self)
        except: pass

    def init(*args): return _dolfin.Matrix_init(*args)
    def size(*args): return _dolfin.Matrix_size(*args)
    def nz(*args): return _dolfin.Matrix_nz(*args)
    def nzsum(*args): return _dolfin.Matrix_nzsum(*args)
    def nzmax(*args): return _dolfin.Matrix_nzmax(*args)
    def set(*args): return _dolfin.Matrix_set(*args)
    def add(*args): return _dolfin.Matrix_add(*args)
    def getRow(*args): return _dolfin.Matrix_getRow(*args)
    def ident(*args): return _dolfin.Matrix_ident(*args)
    def mult(*args): return _dolfin.Matrix_mult(*args)
    def lump(*args): return _dolfin.Matrix_lump(*args)
    l1 = _dolfin.Matrix_l1
    linf = _dolfin.Matrix_linf
    frobenius = _dolfin.Matrix_frobenius
    def norm(*args): return _dolfin.Matrix_norm(*args)
    def apply(*args): return _dolfin.Matrix_apply(*args)
    def zero(*args): return _dolfin.Matrix_zero(*args)
    def type(*args): return _dolfin.Matrix_type(*args)
    def mat(*args): return _dolfin.Matrix_mat(*args)
    def disp(*args): return _dolfin.Matrix_disp(*args)
    def __call__(*args): return _dolfin.Matrix___call__(*args)
    def getval(*args): return _dolfin.Matrix_getval(*args)
    def setval(*args): return _dolfin.Matrix_setval(*args)
    def addval(*args): return _dolfin.Matrix_addval(*args)

class MatrixPtr(Matrix):
    def __init__(self, this):
        _swig_setattr(self, Matrix, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Matrix, 'thisown', 0)
        self.__class__ = Matrix
_dolfin.Matrix_swigregister(MatrixPtr)

class PETScSparseMatrixElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PETScSparseMatrixElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PETScSparseMatrixElement, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScSparseMatrixElement instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, PETScSparseMatrixElement, 'this', _dolfin.new_PETScSparseMatrixElement(*args))
        _swig_setattr(self, PETScSparseMatrixElement, 'thisown', 1)
    def __iadd__(*args): return _dolfin.PETScSparseMatrixElement___iadd__(*args)
    def __isub__(*args): return _dolfin.PETScSparseMatrixElement___isub__(*args)
    def __imul__(*args): return _dolfin.PETScSparseMatrixElement___imul__(*args)

class PETScSparseMatrixElementPtr(PETScSparseMatrixElement):
    def __init__(self, this):
        _swig_setattr(self, PETScSparseMatrixElement, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, PETScSparseMatrixElement, 'thisown', 0)
        self.__class__ = PETScSparseMatrixElement
_dolfin.PETScSparseMatrixElement_swigregister(PETScSparseMatrixElementPtr)

class VirtualMatrix(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VirtualMatrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VirtualMatrix, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::VirtualMatrix instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_VirtualMatrix):
        try:
            if self.thisown: destroy(self)
        except: pass

    def init(*args): return _dolfin.VirtualMatrix_init(*args)
    def size(*args): return _dolfin.VirtualMatrix_size(*args)
    def mat(*args): return _dolfin.VirtualMatrix_mat(*args)
    def mult(*args): return _dolfin.VirtualMatrix_mult(*args)
    def disp(*args): return _dolfin.VirtualMatrix_disp(*args)

class VirtualMatrixPtr(VirtualMatrix):
    def __init__(self, this):
        _swig_setattr(self, VirtualMatrix, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, VirtualMatrix, 'thisown', 0)
        self.__class__ = VirtualMatrix
_dolfin.VirtualMatrix_swigregister(VirtualMatrixPtr)

class GMRES(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GMRES, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GMRES, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::GMRES instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, GMRES, 'this', _dolfin.new_GMRES(*args))
        _swig_setattr(self, GMRES, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_GMRES):
        try:
            if self.thisown: destroy(self)
        except: pass


class GMRESPtr(GMRES):
    def __init__(self, this):
        _swig_setattr(self, GMRES, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, GMRES, 'thisown', 0)
        self.__class__ = GMRES
_dolfin.GMRES_swigregister(GMRESPtr)

class KrylovSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, KrylovSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, KrylovSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScKrylovSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    bicgstab = _dolfin.KrylovSolver_bicgstab
    cg = _dolfin.KrylovSolver_cg
    default_solver = _dolfin.KrylovSolver_default_solver
    gmres = _dolfin.KrylovSolver_gmres
    def __init__(self, *args):
        _swig_setattr(self, KrylovSolver, 'this', _dolfin.new_KrylovSolver(*args))
        _swig_setattr(self, KrylovSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_KrylovSolver):
        try:
            if self.thisown: destroy(self)
        except: pass

    def solve(*args): return _dolfin.KrylovSolver_solve(*args)
    def disp(*args): return _dolfin.KrylovSolver_disp(*args)

class KrylovSolverPtr(KrylovSolver):
    def __init__(self, this):
        _swig_setattr(self, KrylovSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, KrylovSolver, 'thisown', 0)
        self.__class__ = KrylovSolver
_dolfin.KrylovSolver_swigregister(KrylovSolverPtr)

class LinearSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, LinearSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::LinearSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, LinearSolver, 'this', _dolfin.new_LinearSolver(*args))
        _swig_setattr(self, LinearSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_LinearSolver):
        try:
            if self.thisown: destroy(self)
        except: pass

    def solve(*args): return _dolfin.LinearSolver_solve(*args)

class LinearSolverPtr(LinearSolver):
    def __init__(self, this):
        _swig_setattr(self, LinearSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, LinearSolver, 'thisown', 0)
        self.__class__ = LinearSolver
_dolfin.LinearSolver_swigregister(LinearSolverPtr)

class Preconditioner(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Preconditioner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Preconditioner, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Preconditioner instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    default_pc = _dolfin.Preconditioner_default_pc
    hypre_amg = _dolfin.Preconditioner_hypre_amg
    icc = _dolfin.Preconditioner_icc
    ilu = _dolfin.Preconditioner_ilu
    jacobi = _dolfin.Preconditioner_jacobi
    sor = _dolfin.Preconditioner_sor
    none = _dolfin.Preconditioner_none
    def __del__(self, destroy=_dolfin.delete_Preconditioner):
        try:
            if self.thisown: destroy(self)
        except: pass

    __swig_getmethods__["setup"] = lambda x: _dolfin.Preconditioner_setup
    if _newclass:setup = staticmethod(_dolfin.Preconditioner_setup)
    def solve(*args): return _dolfin.Preconditioner_solve(*args)

class PreconditionerPtr(Preconditioner):
    def __init__(self, this):
        _swig_setattr(self, Preconditioner, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Preconditioner, 'thisown', 0)
        self.__class__ = Preconditioner
_dolfin.Preconditioner_swigregister(PreconditionerPtr)

Preconditioner_setup = _dolfin.Preconditioner_setup

class PETScManager(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PETScManager, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PETScManager, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::PETScManager instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_getmethods__["init"] = lambda x: _dolfin.PETScManager_init
    if _newclass:init = staticmethod(_dolfin.PETScManager_init)

class PETScManagerPtr(PETScManager):
    def __init__(self, this):
        _swig_setattr(self, PETScManager, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, PETScManager, 'thisown', 0)
        self.__class__ = PETScManager
_dolfin.PETScManager_swigregister(PETScManagerPtr)

PETScManager_init = _dolfin.PETScManager_init

class Function(Variable,TimeDependent):
    __swig_setmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Function, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Function, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Function instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        if self.__class__ == Function:
            args = (None,) + args
        else:
            args = (self,) + args
        _swig_setattr(self, Function, 'this', _dolfin.new_Function(*args))
        _swig_setattr(self, Function, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Function):
        try:
            if self.thisown: destroy(self)
        except: pass

    def eval(*args): return _dolfin.Function_eval(*args)
    def __call__(*args): return _dolfin.Function___call__(*args)
    def __getitem__(*args): return _dolfin.Function___getitem__(*args)
    def interpolate(*args): return _dolfin.Function_interpolate(*args)
    def vectordim(*args): return _dolfin.Function_vectordim(*args)
    def vector(*args): return _dolfin.Function_vector(*args)
    def mesh(*args): return _dolfin.Function_mesh(*args)
    def element(*args): return _dolfin.Function_element(*args)
    def attach(*args): return _dolfin.Function_attach(*args)
    def init(*args): return _dolfin.Function_init(*args)
    constant = _dolfin.Function_constant
    user = _dolfin.Function_user
    functionpointer = _dolfin.Function_functionpointer
    discrete = _dolfin.Function_discrete
    def type(*args): return _dolfin.Function_type(*args)
    def __disown__(self):
        self.thisown = 0
        _dolfin.disown_Function(self)
        return weakref_proxy(self)

class FunctionPtr(Function):
    def __init__(self, this):
        _swig_setattr(self, Function, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Function, 'thisown', 0)
        self.__class__ = Function
_dolfin.Function_swigregister(FunctionPtr)

class FEM(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FEM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FEM, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::FEM instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_getmethods__["assemble"] = lambda x: _dolfin.FEM_assemble
    if _newclass:assemble = staticmethod(_dolfin.FEM_assemble)
    __swig_getmethods__["applyBC"] = lambda x: _dolfin.FEM_applyBC
    if _newclass:applyBC = staticmethod(_dolfin.FEM_applyBC)
    __swig_getmethods__["assembleResidualBC"] = lambda x: _dolfin.FEM_assembleResidualBC
    if _newclass:assembleResidualBC = staticmethod(_dolfin.FEM_assembleResidualBC)
    __swig_getmethods__["size"] = lambda x: _dolfin.FEM_size
    if _newclass:size = staticmethod(_dolfin.FEM_size)
    __swig_getmethods__["disp"] = lambda x: _dolfin.FEM_disp
    if _newclass:disp = staticmethod(_dolfin.FEM_disp)

class FEMPtr(FEM):
    def __init__(self, this):
        _swig_setattr(self, FEM, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, FEM, 'thisown', 0)
        self.__class__ = FEM
_dolfin.FEM_swigregister(FEMPtr)

FEM_assemble = _dolfin.FEM_assemble

FEM_applyBC = _dolfin.FEM_applyBC

FEM_assembleResidualBC = _dolfin.FEM_assembleResidualBC

FEM_size = _dolfin.FEM_size

FEM_disp = _dolfin.FEM_disp

class FiniteElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FiniteElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FiniteElement, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::FiniteElement instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_FiniteElement):
        try:
            if self.thisown: destroy(self)
        except: pass

    def spacedim(*args): return _dolfin.FiniteElement_spacedim(*args)
    def shapedim(*args): return _dolfin.FiniteElement_shapedim(*args)
    def tensordim(*args): return _dolfin.FiniteElement_tensordim(*args)
    def elementdim(*args): return _dolfin.FiniteElement_elementdim(*args)
    def rank(*args): return _dolfin.FiniteElement_rank(*args)
    def nodemap(*args): return _dolfin.FiniteElement_nodemap(*args)
    def pointmap(*args): return _dolfin.FiniteElement_pointmap(*args)
    def vertexeval(*args): return _dolfin.FiniteElement_vertexeval(*args)
    def spec(*args): return _dolfin.FiniteElement_spec(*args)
    __swig_getmethods__["makeElement"] = lambda x: _dolfin.FiniteElement_makeElement
    if _newclass:makeElement = staticmethod(_dolfin.FiniteElement_makeElement)
    def disp(*args): return _dolfin.FiniteElement_disp(*args)

class FiniteElementPtr(FiniteElement):
    def __init__(self, this):
        _swig_setattr(self, FiniteElement, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, FiniteElement, 'thisown', 0)
        self.__class__ = FiniteElement
_dolfin.FiniteElement_swigregister(FiniteElementPtr)

FiniteElement_makeElement = _dolfin.FiniteElement_makeElement

class AffineMap(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, AffineMap, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, AffineMap, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::AffineMap instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, AffineMap, 'this', _dolfin.new_AffineMap(*args))
        _swig_setattr(self, AffineMap, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_AffineMap):
        try:
            if self.thisown: destroy(self)
        except: pass

    def update(*args): return _dolfin.AffineMap_update(*args)
    def __call__(*args): return _dolfin.AffineMap___call__(*args)
    def cell(*args): return _dolfin.AffineMap_cell(*args)
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

class AffineMapPtr(AffineMap):
    def __init__(self, this):
        _swig_setattr(self, AffineMap, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, AffineMap, 'thisown', 0)
        self.__class__ = AffineMap
_dolfin.AffineMap_swigregister(AffineMapPtr)

class BoundaryValue(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryValue, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryValue, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::BoundaryValue instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, BoundaryValue, 'this', _dolfin.new_BoundaryValue(*args))
        _swig_setattr(self, BoundaryValue, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_BoundaryValue):
        try:
            if self.thisown: destroy(self)
        except: pass

    def set(*args): return _dolfin.BoundaryValue_set(*args)
    def reset(*args): return _dolfin.BoundaryValue_reset(*args)

class BoundaryValuePtr(BoundaryValue):
    def __init__(self, this):
        _swig_setattr(self, BoundaryValue, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, BoundaryValue, 'thisown', 0)
        self.__class__ = BoundaryValue
_dolfin.BoundaryValue_swigregister(BoundaryValuePtr)

class BoundaryCondition(TimeDependent):
    __swig_setmethods__ = {}
    for _s in [TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryCondition, name, value)
    __swig_getmethods__ = {}
    for _s in [TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryCondition, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::BoundaryCondition instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        if self.__class__ == BoundaryCondition:
            args = (None,) + args
        else:
            args = (self,) + args
        _swig_setattr(self, BoundaryCondition, 'this', _dolfin.new_BoundaryCondition(*args))
        _swig_setattr(self, BoundaryCondition, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_BoundaryCondition):
        try:
            if self.thisown: destroy(self)
        except: pass

    def eval(*args): return _dolfin.BoundaryCondition_eval(*args)
    def __disown__(self):
        self.thisown = 0
        _dolfin.disown_BoundaryCondition(self)
        return weakref_proxy(self)

class BoundaryConditionPtr(BoundaryCondition):
    def __init__(self, this):
        _swig_setattr(self, BoundaryCondition, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, BoundaryCondition, 'thisown', 0)
        self.__class__ = BoundaryCondition
_dolfin.BoundaryCondition_swigregister(BoundaryConditionPtr)

class Form(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Form, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Form, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Form instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Form, 'this', _dolfin.new_Form(*args))
        _swig_setattr(self, Form, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Form):
        try:
            if self.thisown: destroy(self)
        except: pass

    def update(*args): return _dolfin.Form_update(*args)
    def function(*args): return _dolfin.Form_function(*args)
    def element(*args): return _dolfin.Form_element(*args)
    __swig_setmethods__["num_functions"] = _dolfin.Form_num_functions_set
    __swig_getmethods__["num_functions"] = _dolfin.Form_num_functions_get
    if _newclass:num_functions = property(_dolfin.Form_num_functions_get, _dolfin.Form_num_functions_set)

class FormPtr(Form):
    def __init__(self, this):
        _swig_setattr(self, Form, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Form, 'thisown', 0)
        self.__class__ = Form
_dolfin.Form_swigregister(FormPtr)

class BilinearForm(Form):
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BilinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BilinearForm, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::BilinearForm instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_BilinearForm):
        try:
            if self.thisown: destroy(self)
        except: pass

    def eval(*args): return _dolfin.BilinearForm_eval(*args)
    def test(*args): return _dolfin.BilinearForm_test(*args)
    def trial(*args): return _dolfin.BilinearForm_trial(*args)

class BilinearFormPtr(BilinearForm):
    def __init__(self, this):
        _swig_setattr(self, BilinearForm, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, BilinearForm, 'thisown', 0)
        self.__class__ = BilinearForm
_dolfin.BilinearForm_swigregister(BilinearFormPtr)

class LinearForm(Form):
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, LinearForm, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::LinearForm instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_LinearForm):
        try:
            if self.thisown: destroy(self)
        except: pass

    def eval(*args): return _dolfin.LinearForm_eval(*args)
    def test(*args): return _dolfin.LinearForm_test(*args)

class LinearFormPtr(LinearForm):
    def __init__(self, this):
        _swig_setattr(self, LinearForm, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, LinearForm, 'thisown', 0)
        self.__class__ = LinearForm
_dolfin.LinearForm_swigregister(LinearFormPtr)

class Mesh(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Mesh, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Mesh, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Mesh instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    triangles = _dolfin.Mesh_triangles
    tetrahedra = _dolfin.Mesh_tetrahedra
    def __init__(self, *args):
        _swig_setattr(self, Mesh, 'this', _dolfin.new_Mesh(*args))
        _swig_setattr(self, Mesh, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Mesh):
        try:
            if self.thisown: destroy(self)
        except: pass

    def merge(*args): return _dolfin.Mesh_merge(*args)
    def init(*args): return _dolfin.Mesh_init(*args)
    def clear(*args): return _dolfin.Mesh_clear(*args)
    def numSpaceDim(*args): return _dolfin.Mesh_numSpaceDim(*args)
    def numVertices(*args): return _dolfin.Mesh_numVertices(*args)
    def numCells(*args): return _dolfin.Mesh_numCells(*args)
    def numEdges(*args): return _dolfin.Mesh_numEdges(*args)
    def numFaces(*args): return _dolfin.Mesh_numFaces(*args)
    def createVertex(*args): return _dolfin.Mesh_createVertex(*args)
    def createCell(*args): return _dolfin.Mesh_createCell(*args)
    def createEdge(*args): return _dolfin.Mesh_createEdge(*args)
    def createFace(*args): return _dolfin.Mesh_createFace(*args)
    def remove(*args): return _dolfin.Mesh_remove(*args)
    def type(*args): return _dolfin.Mesh_type(*args)
    def vertex(*args): return _dolfin.Mesh_vertex(*args)
    def cell(*args): return _dolfin.Mesh_cell(*args)
    def edge(*args): return _dolfin.Mesh_edge(*args)
    def face(*args): return _dolfin.Mesh_face(*args)
    def boundary(*args): return _dolfin.Mesh_boundary(*args)
    def refine(*args): return _dolfin.Mesh_refine(*args)
    def refineUniformly(*args): return _dolfin.Mesh_refineUniformly(*args)
    def parent(*args): return _dolfin.Mesh_parent(*args)
    def child(*args): return _dolfin.Mesh_child(*args)
    def __eq__(*args): return _dolfin.Mesh___eq__(*args)
    def __ne__(*args): return _dolfin.Mesh___ne__(*args)
    def disp(*args): return _dolfin.Mesh_disp(*args)

class MeshPtr(Mesh):
    def __init__(self, this):
        _swig_setattr(self, Mesh, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Mesh, 'thisown', 0)
        self.__class__ = Mesh
_dolfin.Mesh_swigregister(MeshPtr)

class Boundary(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Boundary, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Boundary, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Boundary instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Boundary, 'this', _dolfin.new_Boundary(*args))
        _swig_setattr(self, Boundary, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Boundary):
        try:
            if self.thisown: destroy(self)
        except: pass

    def numVertices(*args): return _dolfin.Boundary_numVertices(*args)
    def numEdges(*args): return _dolfin.Boundary_numEdges(*args)
    def numFaces(*args): return _dolfin.Boundary_numFaces(*args)
    def numFacets(*args): return _dolfin.Boundary_numFacets(*args)

class BoundaryPtr(Boundary):
    def __init__(self, this):
        _swig_setattr(self, Boundary, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Boundary, 'thisown', 0)
        self.__class__ = Boundary
_dolfin.Boundary_swigregister(BoundaryPtr)

class Point(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Point, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Point, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Point instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Point, 'this', _dolfin.new_Point(*args))
        _swig_setattr(self, Point, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Point):
        try:
            if self.thisown: destroy(self)
        except: pass

    def dist(*args): return _dolfin.Point_dist(*args)
    def norm(*args): return _dolfin.Point_norm(*args)
    def midpoint(*args): return _dolfin.Point_midpoint(*args)
    def __add__(*args): return _dolfin.Point___add__(*args)
    def __sub__(*args): return _dolfin.Point___sub__(*args)
    def __mul__(*args): return _dolfin.Point___mul__(*args)
    def __iadd__(*args): return _dolfin.Point___iadd__(*args)
    def __isub__(*args): return _dolfin.Point___isub__(*args)
    def __imul__(*args): return _dolfin.Point___imul__(*args)
    def __idiv__(*args): return _dolfin.Point___idiv__(*args)
    def cross(*args): return _dolfin.Point_cross(*args)
    __swig_setmethods__["x"] = _dolfin.Point_x_set
    __swig_getmethods__["x"] = _dolfin.Point_x_get
    if _newclass:x = property(_dolfin.Point_x_get, _dolfin.Point_x_set)
    __swig_setmethods__["y"] = _dolfin.Point_y_set
    __swig_getmethods__["y"] = _dolfin.Point_y_get
    if _newclass:y = property(_dolfin.Point_y_get, _dolfin.Point_y_set)
    __swig_setmethods__["z"] = _dolfin.Point_z_set
    __swig_getmethods__["z"] = _dolfin.Point_z_get
    if _newclass:z = property(_dolfin.Point_z_get, _dolfin.Point_z_set)

class PointPtr(Point):
    def __init__(self, this):
        _swig_setattr(self, Point, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Point, 'thisown', 0)
        self.__class__ = Point
_dolfin.Point_swigregister(PointPtr)

class Vertex(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vertex, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vertex, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Vertex instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Vertex, 'this', _dolfin.new_Vertex(*args))
        _swig_setattr(self, Vertex, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Vertex):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clear(*args): return _dolfin.Vertex_clear(*args)
    def id(*args): return _dolfin.Vertex_id(*args)
    def numVertexNeighbors(*args): return _dolfin.Vertex_numVertexNeighbors(*args)
    def numCellNeighbors(*args): return _dolfin.Vertex_numCellNeighbors(*args)
    def numEdgeNeighbors(*args): return _dolfin.Vertex_numEdgeNeighbors(*args)
    def vertex(*args): return _dolfin.Vertex_vertex(*args)
    def cell(*args): return _dolfin.Vertex_cell(*args)
    def edge(*args): return _dolfin.Vertex_edge(*args)
    def parent(*args): return _dolfin.Vertex_parent(*args)
    def child(*args): return _dolfin.Vertex_child(*args)
    def mesh(*args): return _dolfin.Vertex_mesh(*args)
    def coord(*args): return _dolfin.Vertex_coord(*args)
    def midpoint(*args): return _dolfin.Vertex_midpoint(*args)
    def dist(*args): return _dolfin.Vertex_dist(*args)
    def neighbor(*args): return _dolfin.Vertex_neighbor(*args)
    def __ne__(*args): return _dolfin.Vertex___ne__(*args)
    def __eq__(*args): return _dolfin.Vertex___eq__(*args)
    def __lt__(*args): return _dolfin.Vertex___lt__(*args)
    def __le__(*args): return _dolfin.Vertex___le__(*args)
    def __gt__(*args): return _dolfin.Vertex___gt__(*args)
    def __ge__(*args): return _dolfin.Vertex___ge__(*args)
    __swig_setmethods__["nbids"] = _dolfin.Vertex_nbids_set
    __swig_getmethods__["nbids"] = _dolfin.Vertex_nbids_get
    if _newclass:nbids = property(_dolfin.Vertex_nbids_get, _dolfin.Vertex_nbids_set)

class VertexPtr(Vertex):
    def __init__(self, this):
        _swig_setattr(self, Vertex, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Vertex, 'thisown', 0)
        self.__class__ = Vertex
_dolfin.Vertex_swigregister(VertexPtr)

class Edge(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Edge, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Edge, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Edge instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Edge, 'this', _dolfin.new_Edge(*args))
        _swig_setattr(self, Edge, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Edge):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clear(*args): return _dolfin.Edge_clear(*args)
    def id(*args): return _dolfin.Edge_id(*args)
    def numCellNeighbors(*args): return _dolfin.Edge_numCellNeighbors(*args)
    def vertex(*args): return _dolfin.Edge_vertex(*args)
    def cell(*args): return _dolfin.Edge_cell(*args)
    def localID(*args): return _dolfin.Edge_localID(*args)
    def mesh(*args): return _dolfin.Edge_mesh(*args)
    def coord(*args): return _dolfin.Edge_coord(*args)
    def length(*args): return _dolfin.Edge_length(*args)
    def midpoint(*args): return _dolfin.Edge_midpoint(*args)
    def equals(*args): return _dolfin.Edge_equals(*args)
    def contains(*args): return _dolfin.Edge_contains(*args)
    __swig_setmethods__["ebids"] = _dolfin.Edge_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.Edge_ebids_get
    if _newclass:ebids = property(_dolfin.Edge_ebids_get, _dolfin.Edge_ebids_set)

class EdgePtr(Edge):
    def __init__(self, this):
        _swig_setattr(self, Edge, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Edge, 'thisown', 0)
        self.__class__ = Edge
_dolfin.Edge_swigregister(EdgePtr)

class Triangle(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Triangle, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Triangle, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Triangle instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Triangle, 'this', _dolfin.new_Triangle(*args))
        _swig_setattr(self, Triangle, 'thisown', 1)
    def numVertices(*args): return _dolfin.Triangle_numVertices(*args)
    def numEdges(*args): return _dolfin.Triangle_numEdges(*args)
    def numFaces(*args): return _dolfin.Triangle_numFaces(*args)
    def numBoundaries(*args): return _dolfin.Triangle_numBoundaries(*args)
    def type(*args): return _dolfin.Triangle_type(*args)
    def orientation(*args): return _dolfin.Triangle_orientation(*args)
    def volume(*args): return _dolfin.Triangle_volume(*args)
    def diameter(*args): return _dolfin.Triangle_diameter(*args)
    def edgeAlignment(*args): return _dolfin.Triangle_edgeAlignment(*args)
    def faceAlignment(*args): return _dolfin.Triangle_faceAlignment(*args)

class TrianglePtr(Triangle):
    def __init__(self, this):
        _swig_setattr(self, Triangle, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Triangle, 'thisown', 0)
        self.__class__ = Triangle
_dolfin.Triangle_swigregister(TrianglePtr)

class Tetrahedron(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Tetrahedron, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Tetrahedron, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Tetrahedron instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Tetrahedron, 'this', _dolfin.new_Tetrahedron(*args))
        _swig_setattr(self, Tetrahedron, 'thisown', 1)
    def numVertices(*args): return _dolfin.Tetrahedron_numVertices(*args)
    def numEdges(*args): return _dolfin.Tetrahedron_numEdges(*args)
    def numFaces(*args): return _dolfin.Tetrahedron_numFaces(*args)
    def numBoundaries(*args): return _dolfin.Tetrahedron_numBoundaries(*args)
    def type(*args): return _dolfin.Tetrahedron_type(*args)
    def orientation(*args): return _dolfin.Tetrahedron_orientation(*args)
    def volume(*args): return _dolfin.Tetrahedron_volume(*args)
    def diameter(*args): return _dolfin.Tetrahedron_diameter(*args)
    def edgeAlignment(*args): return _dolfin.Tetrahedron_edgeAlignment(*args)
    def faceAlignment(*args): return _dolfin.Tetrahedron_faceAlignment(*args)

class TetrahedronPtr(Tetrahedron):
    def __init__(self, this):
        _swig_setattr(self, Tetrahedron, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Tetrahedron, 'thisown', 0)
        self.__class__ = Tetrahedron
_dolfin.Tetrahedron_swigregister(TetrahedronPtr)

class Cell(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Cell, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Cell, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Cell instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    triangle = _dolfin.Cell_triangle
    tetrahedron = _dolfin.Cell_tetrahedron
    none = _dolfin.Cell_none
    left = _dolfin.Cell_left
    right = _dolfin.Cell_right
    def __init__(self, *args):
        _swig_setattr(self, Cell, 'this', _dolfin.new_Cell(*args))
        _swig_setattr(self, Cell, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Cell):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clear(*args): return _dolfin.Cell_clear(*args)
    def id(*args): return _dolfin.Cell_id(*args)
    def type(*args): return _dolfin.Cell_type(*args)
    def orientation(*args): return _dolfin.Cell_orientation(*args)
    def numVertices(*args): return _dolfin.Cell_numVertices(*args)
    def numEdges(*args): return _dolfin.Cell_numEdges(*args)
    def numFaces(*args): return _dolfin.Cell_numFaces(*args)
    def numBoundaries(*args): return _dolfin.Cell_numBoundaries(*args)
    def numCellNeighbors(*args): return _dolfin.Cell_numCellNeighbors(*args)
    def numVertexNeighbors(*args): return _dolfin.Cell_numVertexNeighbors(*args)
    def numChildren(*args): return _dolfin.Cell_numChildren(*args)
    def vertex(*args): return _dolfin.Cell_vertex(*args)
    def edge(*args): return _dolfin.Cell_edge(*args)
    def face(*args): return _dolfin.Cell_face(*args)
    def neighbor(*args): return _dolfin.Cell_neighbor(*args)
    def parent(*args): return _dolfin.Cell_parent(*args)
    def child(*args): return _dolfin.Cell_child(*args)
    def mesh(*args): return _dolfin.Cell_mesh(*args)
    def coord(*args): return _dolfin.Cell_coord(*args)
    def midpoint(*args): return _dolfin.Cell_midpoint(*args)
    def vertexID(*args): return _dolfin.Cell_vertexID(*args)
    def edgeID(*args): return _dolfin.Cell_edgeID(*args)
    def faceID(*args): return _dolfin.Cell_faceID(*args)
    def volume(*args): return _dolfin.Cell_volume(*args)
    def diameter(*args): return _dolfin.Cell_diameter(*args)
    def edgeAlignment(*args): return _dolfin.Cell_edgeAlignment(*args)
    def faceAlignment(*args): return _dolfin.Cell_faceAlignment(*args)
    def __eq__(*args): return _dolfin.Cell___eq__(*args)
    def __ne__(*args): return _dolfin.Cell___ne__(*args)
    def mark(*args): return _dolfin.Cell_mark(*args)

class CellPtr(Cell):
    def __init__(self, this):
        _swig_setattr(self, Cell, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Cell, 'thisown', 0)
        self.__class__ = Cell
_dolfin.Cell_swigregister(CellPtr)

class Face(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Face, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Face, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Face instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Face, 'this', _dolfin.new_Face(*args))
        _swig_setattr(self, Face, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Face):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clear(*args): return _dolfin.Face_clear(*args)
    def id(*args): return _dolfin.Face_id(*args)
    def numEdges(*args): return _dolfin.Face_numEdges(*args)
    def numCellNeighbors(*args): return _dolfin.Face_numCellNeighbors(*args)
    def edge(*args): return _dolfin.Face_edge(*args)
    def cell(*args): return _dolfin.Face_cell(*args)
    def localID(*args): return _dolfin.Face_localID(*args)
    def mesh(*args): return _dolfin.Face_mesh(*args)
    def area(*args): return _dolfin.Face_area(*args)
    def equals(*args): return _dolfin.Face_equals(*args)
    def contains(*args): return _dolfin.Face_contains(*args)
    __swig_setmethods__["fbids"] = _dolfin.Face_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.Face_fbids_get
    if _newclass:fbids = property(_dolfin.Face_fbids_get, _dolfin.Face_fbids_set)

class FacePtr(Face):
    def __init__(self, this):
        _swig_setattr(self, Face, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Face, 'thisown', 0)
        self.__class__ = Face
_dolfin.Face_swigregister(FacePtr)

class VertexIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VertexIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VertexIterator, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::VertexIterator instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, VertexIterator, 'this', _dolfin.new_VertexIterator(*args))
        _swig_setattr(self, VertexIterator, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_VertexIterator):
        try:
            if self.thisown: destroy(self)
        except: pass

    def increment(*args): return _dolfin.VertexIterator_increment(*args)
    def end(*args): return _dolfin.VertexIterator_end(*args)
    def last(*args): return _dolfin.VertexIterator_last(*args)
    def index(*args): return _dolfin.VertexIterator_index(*args)
    def __ref__(*args): return _dolfin.VertexIterator___ref__(*args)
    def __deref__(*args): return _dolfin.VertexIterator___deref__(*args)
    def __eq__(*args): return _dolfin.VertexIterator___eq__(*args)
    def __ne__(*args): return _dolfin.VertexIterator___ne__(*args)
    def clear(*args): return _dolfin.VertexIterator_clear(*args)
    def id(*args): return _dolfin.VertexIterator_id(*args)
    def numVertexNeighbors(*args): return _dolfin.VertexIterator_numVertexNeighbors(*args)
    def numCellNeighbors(*args): return _dolfin.VertexIterator_numCellNeighbors(*args)
    def numEdgeNeighbors(*args): return _dolfin.VertexIterator_numEdgeNeighbors(*args)
    def vertex(*args): return _dolfin.VertexIterator_vertex(*args)
    def cell(*args): return _dolfin.VertexIterator_cell(*args)
    def edge(*args): return _dolfin.VertexIterator_edge(*args)
    def parent(*args): return _dolfin.VertexIterator_parent(*args)
    def child(*args): return _dolfin.VertexIterator_child(*args)
    def mesh(*args): return _dolfin.VertexIterator_mesh(*args)
    def coord(*args): return _dolfin.VertexIterator_coord(*args)
    def midpoint(*args): return _dolfin.VertexIterator_midpoint(*args)
    def dist(*args): return _dolfin.VertexIterator_dist(*args)
    def neighbor(*args): return _dolfin.VertexIterator_neighbor(*args)
    def __lt__(*args): return _dolfin.VertexIterator___lt__(*args)
    def __le__(*args): return _dolfin.VertexIterator___le__(*args)
    def __gt__(*args): return _dolfin.VertexIterator___gt__(*args)
    def __ge__(*args): return _dolfin.VertexIterator___ge__(*args)
    __swig_setmethods__["nbids"] = _dolfin.VertexIterator_nbids_set
    __swig_getmethods__["nbids"] = _dolfin.VertexIterator_nbids_get
    if _newclass:nbids = property(_dolfin.VertexIterator_nbids_get, _dolfin.VertexIterator_nbids_set)

class VertexIteratorPtr(VertexIterator):
    def __init__(self, this):
        _swig_setattr(self, VertexIterator, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, VertexIterator, 'thisown', 0)
        self.__class__ = VertexIterator
_dolfin.VertexIterator_swigregister(VertexIteratorPtr)

class CellIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CellIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CellIterator, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::CellIterator instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, CellIterator, 'this', _dolfin.new_CellIterator(*args))
        _swig_setattr(self, CellIterator, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_CellIterator):
        try:
            if self.thisown: destroy(self)
        except: pass

    def increment(*args): return _dolfin.CellIterator_increment(*args)
    def end(*args): return _dolfin.CellIterator_end(*args)
    def last(*args): return _dolfin.CellIterator_last(*args)
    def index(*args): return _dolfin.CellIterator_index(*args)
    def __ref__(*args): return _dolfin.CellIterator___ref__(*args)
    def __deref__(*args): return _dolfin.CellIterator___deref__(*args)
    def __eq__(*args): return _dolfin.CellIterator___eq__(*args)
    def __ne__(*args): return _dolfin.CellIterator___ne__(*args)
    def clear(*args): return _dolfin.CellIterator_clear(*args)
    def id(*args): return _dolfin.CellIterator_id(*args)
    def type(*args): return _dolfin.CellIterator_type(*args)
    def orientation(*args): return _dolfin.CellIterator_orientation(*args)
    def numVertices(*args): return _dolfin.CellIterator_numVertices(*args)
    def numEdges(*args): return _dolfin.CellIterator_numEdges(*args)
    def numFaces(*args): return _dolfin.CellIterator_numFaces(*args)
    def numBoundaries(*args): return _dolfin.CellIterator_numBoundaries(*args)
    def numCellNeighbors(*args): return _dolfin.CellIterator_numCellNeighbors(*args)
    def numVertexNeighbors(*args): return _dolfin.CellIterator_numVertexNeighbors(*args)
    def numChildren(*args): return _dolfin.CellIterator_numChildren(*args)
    def vertex(*args): return _dolfin.CellIterator_vertex(*args)
    def edge(*args): return _dolfin.CellIterator_edge(*args)
    def face(*args): return _dolfin.CellIterator_face(*args)
    def neighbor(*args): return _dolfin.CellIterator_neighbor(*args)
    def parent(*args): return _dolfin.CellIterator_parent(*args)
    def child(*args): return _dolfin.CellIterator_child(*args)
    def mesh(*args): return _dolfin.CellIterator_mesh(*args)
    def coord(*args): return _dolfin.CellIterator_coord(*args)
    def midpoint(*args): return _dolfin.CellIterator_midpoint(*args)
    def vertexID(*args): return _dolfin.CellIterator_vertexID(*args)
    def edgeID(*args): return _dolfin.CellIterator_edgeID(*args)
    def faceID(*args): return _dolfin.CellIterator_faceID(*args)
    def volume(*args): return _dolfin.CellIterator_volume(*args)
    def diameter(*args): return _dolfin.CellIterator_diameter(*args)
    def edgeAlignment(*args): return _dolfin.CellIterator_edgeAlignment(*args)
    def faceAlignment(*args): return _dolfin.CellIterator_faceAlignment(*args)
    def mark(*args): return _dolfin.CellIterator_mark(*args)

class CellIteratorPtr(CellIterator):
    def __init__(self, this):
        _swig_setattr(self, CellIterator, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, CellIterator, 'thisown', 0)
        self.__class__ = CellIterator
_dolfin.CellIterator_swigregister(CellIteratorPtr)

class EdgeIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EdgeIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EdgeIterator, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::EdgeIterator instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, EdgeIterator, 'this', _dolfin.new_EdgeIterator(*args))
        _swig_setattr(self, EdgeIterator, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_EdgeIterator):
        try:
            if self.thisown: destroy(self)
        except: pass

    def increment(*args): return _dolfin.EdgeIterator_increment(*args)
    def end(*args): return _dolfin.EdgeIterator_end(*args)
    def last(*args): return _dolfin.EdgeIterator_last(*args)
    def index(*args): return _dolfin.EdgeIterator_index(*args)
    def __ref__(*args): return _dolfin.EdgeIterator___ref__(*args)
    def __deref__(*args): return _dolfin.EdgeIterator___deref__(*args)
    def __eq__(*args): return _dolfin.EdgeIterator___eq__(*args)
    def __ne__(*args): return _dolfin.EdgeIterator___ne__(*args)
    def clear(*args): return _dolfin.EdgeIterator_clear(*args)
    def id(*args): return _dolfin.EdgeIterator_id(*args)
    def numCellNeighbors(*args): return _dolfin.EdgeIterator_numCellNeighbors(*args)
    def vertex(*args): return _dolfin.EdgeIterator_vertex(*args)
    def cell(*args): return _dolfin.EdgeIterator_cell(*args)
    def localID(*args): return _dolfin.EdgeIterator_localID(*args)
    def mesh(*args): return _dolfin.EdgeIterator_mesh(*args)
    def coord(*args): return _dolfin.EdgeIterator_coord(*args)
    def length(*args): return _dolfin.EdgeIterator_length(*args)
    def midpoint(*args): return _dolfin.EdgeIterator_midpoint(*args)
    def equals(*args): return _dolfin.EdgeIterator_equals(*args)
    def contains(*args): return _dolfin.EdgeIterator_contains(*args)
    __swig_setmethods__["ebids"] = _dolfin.EdgeIterator_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.EdgeIterator_ebids_get
    if _newclass:ebids = property(_dolfin.EdgeIterator_ebids_get, _dolfin.EdgeIterator_ebids_set)

class EdgeIteratorPtr(EdgeIterator):
    def __init__(self, this):
        _swig_setattr(self, EdgeIterator, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, EdgeIterator, 'thisown', 0)
        self.__class__ = EdgeIterator
_dolfin.EdgeIterator_swigregister(EdgeIteratorPtr)

class FaceIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FaceIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FaceIterator, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::FaceIterator instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, FaceIterator, 'this', _dolfin.new_FaceIterator(*args))
        _swig_setattr(self, FaceIterator, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_FaceIterator):
        try:
            if self.thisown: destroy(self)
        except: pass

    def end(*args): return _dolfin.FaceIterator_end(*args)
    def last(*args): return _dolfin.FaceIterator_last(*args)
    def index(*args): return _dolfin.FaceIterator_index(*args)
    def __ref__(*args): return _dolfin.FaceIterator___ref__(*args)
    def __deref__(*args): return _dolfin.FaceIterator___deref__(*args)
    def __eq__(*args): return _dolfin.FaceIterator___eq__(*args)
    def __ne__(*args): return _dolfin.FaceIterator___ne__(*args)
    def clear(*args): return _dolfin.FaceIterator_clear(*args)
    def id(*args): return _dolfin.FaceIterator_id(*args)
    def numEdges(*args): return _dolfin.FaceIterator_numEdges(*args)
    def numCellNeighbors(*args): return _dolfin.FaceIterator_numCellNeighbors(*args)
    def edge(*args): return _dolfin.FaceIterator_edge(*args)
    def cell(*args): return _dolfin.FaceIterator_cell(*args)
    def localID(*args): return _dolfin.FaceIterator_localID(*args)
    def mesh(*args): return _dolfin.FaceIterator_mesh(*args)
    def area(*args): return _dolfin.FaceIterator_area(*args)
    def equals(*args): return _dolfin.FaceIterator_equals(*args)
    def contains(*args): return _dolfin.FaceIterator_contains(*args)
    __swig_setmethods__["fbids"] = _dolfin.FaceIterator_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.FaceIterator_fbids_get
    if _newclass:fbids = property(_dolfin.FaceIterator_fbids_get, _dolfin.FaceIterator_fbids_set)

class FaceIteratorPtr(FaceIterator):
    def __init__(self, this):
        _swig_setattr(self, FaceIterator, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, FaceIterator, 'thisown', 0)
        self.__class__ = FaceIterator
_dolfin.FaceIterator_swigregister(FaceIteratorPtr)

class MeshIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshIterator, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MeshIterator instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MeshIterator, 'this', _dolfin.new_MeshIterator(*args))
        _swig_setattr(self, MeshIterator, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MeshIterator):
        try:
            if self.thisown: destroy(self)
        except: pass

    def end(*args): return _dolfin.MeshIterator_end(*args)
    def index(*args): return _dolfin.MeshIterator_index(*args)
    def __ref__(*args): return _dolfin.MeshIterator___ref__(*args)
    def __deref__(*args): return _dolfin.MeshIterator___deref__(*args)
    def merge(*args): return _dolfin.MeshIterator_merge(*args)
    def init(*args): return _dolfin.MeshIterator_init(*args)
    def clear(*args): return _dolfin.MeshIterator_clear(*args)
    def numSpaceDim(*args): return _dolfin.MeshIterator_numSpaceDim(*args)
    def numVertices(*args): return _dolfin.MeshIterator_numVertices(*args)
    def numCells(*args): return _dolfin.MeshIterator_numCells(*args)
    def numEdges(*args): return _dolfin.MeshIterator_numEdges(*args)
    def numFaces(*args): return _dolfin.MeshIterator_numFaces(*args)
    def createVertex(*args): return _dolfin.MeshIterator_createVertex(*args)
    def createCell(*args): return _dolfin.MeshIterator_createCell(*args)
    def createEdge(*args): return _dolfin.MeshIterator_createEdge(*args)
    def createFace(*args): return _dolfin.MeshIterator_createFace(*args)
    def remove(*args): return _dolfin.MeshIterator_remove(*args)
    def type(*args): return _dolfin.MeshIterator_type(*args)
    def vertex(*args): return _dolfin.MeshIterator_vertex(*args)
    def cell(*args): return _dolfin.MeshIterator_cell(*args)
    def edge(*args): return _dolfin.MeshIterator_edge(*args)
    def face(*args): return _dolfin.MeshIterator_face(*args)
    def boundary(*args): return _dolfin.MeshIterator_boundary(*args)
    def refine(*args): return _dolfin.MeshIterator_refine(*args)
    def refineUniformly(*args): return _dolfin.MeshIterator_refineUniformly(*args)
    def parent(*args): return _dolfin.MeshIterator_parent(*args)
    def child(*args): return _dolfin.MeshIterator_child(*args)
    def __eq__(*args): return _dolfin.MeshIterator___eq__(*args)
    def __ne__(*args): return _dolfin.MeshIterator___ne__(*args)
    def disp(*args): return _dolfin.MeshIterator_disp(*args)
    def rename(*args): return _dolfin.MeshIterator_rename(*args)
    def name(*args): return _dolfin.MeshIterator_name(*args)
    def label(*args): return _dolfin.MeshIterator_label(*args)
    def number(*args): return _dolfin.MeshIterator_number(*args)

class MeshIteratorPtr(MeshIterator):
    def __init__(self, this):
        _swig_setattr(self, MeshIterator, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MeshIterator, 'thisown', 0)
        self.__class__ = MeshIterator
_dolfin.MeshIterator_swigregister(MeshIteratorPtr)

class UnitSquare(Mesh):
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitSquare, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitSquare, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::UnitSquare instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, UnitSquare, 'this', _dolfin.new_UnitSquare(*args))
        _swig_setattr(self, UnitSquare, 'thisown', 1)

class UnitSquarePtr(UnitSquare):
    def __init__(self, this):
        _swig_setattr(self, UnitSquare, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, UnitSquare, 'thisown', 0)
        self.__class__ = UnitSquare
_dolfin.UnitSquare_swigregister(UnitSquarePtr)

class UnitCube(Mesh):
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitCube, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitCube, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::UnitCube instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, UnitCube, 'this', _dolfin.new_UnitCube(*args))
        _swig_setattr(self, UnitCube, 'thisown', 1)

class UnitCubePtr(UnitCube):
    def __init__(self, this):
        _swig_setattr(self, UnitCube, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, UnitCube, 'thisown', 0)
        self.__class__ = UnitCube
_dolfin.UnitCube_swigregister(UnitCubePtr)

class Dependencies(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Dependencies, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Dependencies, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Dependencies instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Dependencies, 'this', _dolfin.new_Dependencies(*args))
        _swig_setattr(self, Dependencies, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Dependencies):
        try:
            if self.thisown: destroy(self)
        except: pass

    def setsize(*args): return _dolfin.Dependencies_setsize(*args)
    def set(*args): return _dolfin.Dependencies_set(*args)
    def transp(*args): return _dolfin.Dependencies_transp(*args)
    def detect(*args): return _dolfin.Dependencies_detect(*args)
    def sparse(*args): return _dolfin.Dependencies_sparse(*args)
    def disp(*args): return _dolfin.Dependencies_disp(*args)

class DependenciesPtr(Dependencies):
    def __init__(self, this):
        _swig_setattr(self, Dependencies, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Dependencies, 'thisown', 0)
        self.__class__ = Dependencies
_dolfin.Dependencies_swigregister(DependenciesPtr)

class Homotopy(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Homotopy, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Homotopy, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Homotopy instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_Homotopy):
        try:
            if self.thisown: destroy(self)
        except: pass

    def solve(*args): return _dolfin.Homotopy_solve(*args)
    def solutions(*args): return _dolfin.Homotopy_solutions(*args)
    def z0(*args): return _dolfin.Homotopy_z0(*args)
    def F(*args): return _dolfin.Homotopy_F(*args)
    def JF(*args): return _dolfin.Homotopy_JF(*args)
    def G(*args): return _dolfin.Homotopy_G(*args)
    def JG(*args): return _dolfin.Homotopy_JG(*args)
    def modify(*args): return _dolfin.Homotopy_modify(*args)
    def verify(*args): return _dolfin.Homotopy_verify(*args)
    def degree(*args): return _dolfin.Homotopy_degree(*args)

class HomotopyPtr(Homotopy):
    def __init__(self, this):
        _swig_setattr(self, Homotopy, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Homotopy, 'thisown', 0)
        self.__class__ = Homotopy
_dolfin.Homotopy_swigregister(HomotopyPtr)

class HomotopyJacobian(VirtualMatrix):
    __swig_setmethods__ = {}
    for _s in [VirtualMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [VirtualMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyJacobian, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::HomotopyJacobian instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, HomotopyJacobian, 'this', _dolfin.new_HomotopyJacobian(*args))
        _swig_setattr(self, HomotopyJacobian, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_HomotopyJacobian):
        try:
            if self.thisown: destroy(self)
        except: pass

    def mult(*args): return _dolfin.HomotopyJacobian_mult(*args)

class HomotopyJacobianPtr(HomotopyJacobian):
    def __init__(self, this):
        _swig_setattr(self, HomotopyJacobian, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, HomotopyJacobian, 'thisown', 0)
        self.__class__ = HomotopyJacobian
_dolfin.HomotopyJacobian_swigregister(HomotopyJacobianPtr)

class HomotopyODE(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyODE, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::HomotopyODE instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    ode = _dolfin.HomotopyODE_ode
    endgame = _dolfin.HomotopyODE_endgame
    def __init__(self, *args):
        _swig_setattr(self, HomotopyODE, 'this', _dolfin.new_HomotopyODE(*args))
        _swig_setattr(self, HomotopyODE, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_HomotopyODE):
        try:
            if self.thisown: destroy(self)
        except: pass

    def z0(*args): return _dolfin.HomotopyODE_z0(*args)
    def f(*args): return _dolfin.HomotopyODE_f(*args)
    def M(*args): return _dolfin.HomotopyODE_M(*args)
    def J(*args): return _dolfin.HomotopyODE_J(*args)
    def update(*args): return _dolfin.HomotopyODE_update(*args)
    def state(*args): return _dolfin.HomotopyODE_state(*args)

class HomotopyODEPtr(HomotopyODE):
    def __init__(self, this):
        _swig_setattr(self, HomotopyODE, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, HomotopyODE, 'thisown', 0)
        self.__class__ = HomotopyODE
_dolfin.HomotopyODE_swigregister(HomotopyODEPtr)

class Method(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Method, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Method, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Method instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    cG = _dolfin.Method_cG
    dG = _dolfin.Method_dG
    none = _dolfin.Method_none
    def __del__(self, destroy=_dolfin.delete_Method):
        try:
            if self.thisown: destroy(self)
        except: pass

    def type(*args): return _dolfin.Method_type(*args)
    def degree(*args): return _dolfin.Method_degree(*args)
    def order(*args): return _dolfin.Method_order(*args)
    def nsize(*args): return _dolfin.Method_nsize(*args)
    def qsize(*args): return _dolfin.Method_qsize(*args)
    def npoint(*args): return _dolfin.Method_npoint(*args)
    def qpoint(*args): return _dolfin.Method_qpoint(*args)
    def nweight(*args): return _dolfin.Method_nweight(*args)
    def qweight(*args): return _dolfin.Method_qweight(*args)
    def eval(*args): return _dolfin.Method_eval(*args)
    def derivative(*args): return _dolfin.Method_derivative(*args)
    def update(*args): return _dolfin.Method_update(*args)
    def ueval(*args): return _dolfin.Method_ueval(*args)
    def residual(*args): return _dolfin.Method_residual(*args)
    def timestep(*args): return _dolfin.Method_timestep(*args)
    def error(*args): return _dolfin.Method_error(*args)
    def disp(*args): return _dolfin.Method_disp(*args)

class MethodPtr(Method):
    def __init__(self, this):
        _swig_setattr(self, Method, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Method, 'thisown', 0)
        self.__class__ = Method
_dolfin.Method_swigregister(MethodPtr)

class MonoAdaptiveFixedPointSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveFixedPointSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveFixedPointSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MonoAdaptiveFixedPointSolver, 'this', _dolfin.new_MonoAdaptiveFixedPointSolver(*args))
        _swig_setattr(self, MonoAdaptiveFixedPointSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MonoAdaptiveFixedPointSolver):
        try:
            if self.thisown: destroy(self)
        except: pass


class MonoAdaptiveFixedPointSolverPtr(MonoAdaptiveFixedPointSolver):
    def __init__(self, this):
        _swig_setattr(self, MonoAdaptiveFixedPointSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MonoAdaptiveFixedPointSolver, 'thisown', 0)
        self.__class__ = MonoAdaptiveFixedPointSolver
_dolfin.MonoAdaptiveFixedPointSolver_swigregister(MonoAdaptiveFixedPointSolverPtr)

class MonoAdaptiveJacobian(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveJacobian, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveJacobian, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveJacobian instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MonoAdaptiveJacobian, 'this', _dolfin.new_MonoAdaptiveJacobian(*args))
        _swig_setattr(self, MonoAdaptiveJacobian, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MonoAdaptiveJacobian):
        try:
            if self.thisown: destroy(self)
        except: pass

    def mult(*args): return _dolfin.MonoAdaptiveJacobian_mult(*args)

class MonoAdaptiveJacobianPtr(MonoAdaptiveJacobian):
    def __init__(self, this):
        _swig_setattr(self, MonoAdaptiveJacobian, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MonoAdaptiveJacobian, 'thisown', 0)
        self.__class__ = MonoAdaptiveJacobian
_dolfin.MonoAdaptiveJacobian_swigregister(MonoAdaptiveJacobianPtr)

class MonoAdaptiveNewtonSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveNewtonSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveNewtonSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MonoAdaptiveNewtonSolver, 'this', _dolfin.new_MonoAdaptiveNewtonSolver(*args))
        _swig_setattr(self, MonoAdaptiveNewtonSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MonoAdaptiveNewtonSolver):
        try:
            if self.thisown: destroy(self)
        except: pass


class MonoAdaptiveNewtonSolverPtr(MonoAdaptiveNewtonSolver):
    def __init__(self, this):
        _swig_setattr(self, MonoAdaptiveNewtonSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MonoAdaptiveNewtonSolver, 'thisown', 0)
        self.__class__ = MonoAdaptiveNewtonSolver
_dolfin.MonoAdaptiveNewtonSolver_swigregister(MonoAdaptiveNewtonSolverPtr)

class MonoAdaptiveTimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveTimeSlab, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveTimeSlab instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MonoAdaptiveTimeSlab, 'this', _dolfin.new_MonoAdaptiveTimeSlab(*args))
        _swig_setattr(self, MonoAdaptiveTimeSlab, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MonoAdaptiveTimeSlab):
        try:
            if self.thisown: destroy(self)
        except: pass

    def build(*args): return _dolfin.MonoAdaptiveTimeSlab_build(*args)
    def solve(*args): return _dolfin.MonoAdaptiveTimeSlab_solve(*args)
    def check(*args): return _dolfin.MonoAdaptiveTimeSlab_check(*args)
    def shift(*args): return _dolfin.MonoAdaptiveTimeSlab_shift(*args)
    def sample(*args): return _dolfin.MonoAdaptiveTimeSlab_sample(*args)
    def usample(*args): return _dolfin.MonoAdaptiveTimeSlab_usample(*args)
    def ksample(*args): return _dolfin.MonoAdaptiveTimeSlab_ksample(*args)
    def rsample(*args): return _dolfin.MonoAdaptiveTimeSlab_rsample(*args)
    def disp(*args): return _dolfin.MonoAdaptiveTimeSlab_disp(*args)

class MonoAdaptiveTimeSlabPtr(MonoAdaptiveTimeSlab):
    def __init__(self, this):
        _swig_setattr(self, MonoAdaptiveTimeSlab, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MonoAdaptiveTimeSlab, 'thisown', 0)
        self.__class__ = MonoAdaptiveTimeSlab
_dolfin.MonoAdaptiveTimeSlab_swigregister(MonoAdaptiveTimeSlabPtr)

class MonoAdaptivity(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptivity, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptivity instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MonoAdaptivity, 'this', _dolfin.new_MonoAdaptivity(*args))
        _swig_setattr(self, MonoAdaptivity, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MonoAdaptivity):
        try:
            if self.thisown: destroy(self)
        except: pass

    def timestep(*args): return _dolfin.MonoAdaptivity_timestep(*args)
    def update(*args): return _dolfin.MonoAdaptivity_update(*args)

class MonoAdaptivityPtr(MonoAdaptivity):
    def __init__(self, this):
        _swig_setattr(self, MonoAdaptivity, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MonoAdaptivity, 'thisown', 0)
        self.__class__ = MonoAdaptivity
_dolfin.MonoAdaptivity_swigregister(MonoAdaptivityPtr)

class MultiAdaptiveFixedPointSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveFixedPointSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveFixedPointSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MultiAdaptiveFixedPointSolver, 'this', _dolfin.new_MultiAdaptiveFixedPointSolver(*args))
        _swig_setattr(self, MultiAdaptiveFixedPointSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MultiAdaptiveFixedPointSolver):
        try:
            if self.thisown: destroy(self)
        except: pass


class MultiAdaptiveFixedPointSolverPtr(MultiAdaptiveFixedPointSolver):
    def __init__(self, this):
        _swig_setattr(self, MultiAdaptiveFixedPointSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MultiAdaptiveFixedPointSolver, 'thisown', 0)
        self.__class__ = MultiAdaptiveFixedPointSolver
_dolfin.MultiAdaptiveFixedPointSolver_swigregister(MultiAdaptiveFixedPointSolverPtr)

class MultiAdaptivePreconditioner(Preconditioner):
    __swig_setmethods__ = {}
    for _s in [Preconditioner]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivePreconditioner, name, value)
    __swig_getmethods__ = {}
    for _s in [Preconditioner]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivePreconditioner, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptivePreconditioner instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MultiAdaptivePreconditioner, 'this', _dolfin.new_MultiAdaptivePreconditioner(*args))
        _swig_setattr(self, MultiAdaptivePreconditioner, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MultiAdaptivePreconditioner):
        try:
            if self.thisown: destroy(self)
        except: pass

    def solve(*args): return _dolfin.MultiAdaptivePreconditioner_solve(*args)

class MultiAdaptivePreconditionerPtr(MultiAdaptivePreconditioner):
    def __init__(self, this):
        _swig_setattr(self, MultiAdaptivePreconditioner, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MultiAdaptivePreconditioner, 'thisown', 0)
        self.__class__ = MultiAdaptivePreconditioner
_dolfin.MultiAdaptivePreconditioner_swigregister(MultiAdaptivePreconditionerPtr)

class MultiAdaptiveNewtonSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveNewtonSolver, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveNewtonSolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MultiAdaptiveNewtonSolver, 'this', _dolfin.new_MultiAdaptiveNewtonSolver(*args))
        _swig_setattr(self, MultiAdaptiveNewtonSolver, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MultiAdaptiveNewtonSolver):
        try:
            if self.thisown: destroy(self)
        except: pass


class MultiAdaptiveNewtonSolverPtr(MultiAdaptiveNewtonSolver):
    def __init__(self, this):
        _swig_setattr(self, MultiAdaptiveNewtonSolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MultiAdaptiveNewtonSolver, 'thisown', 0)
        self.__class__ = MultiAdaptiveNewtonSolver
_dolfin.MultiAdaptiveNewtonSolver_swigregister(MultiAdaptiveNewtonSolverPtr)

class MultiAdaptiveTimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveTimeSlab, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveTimeSlab instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MultiAdaptiveTimeSlab, 'this', _dolfin.new_MultiAdaptiveTimeSlab(*args))
        _swig_setattr(self, MultiAdaptiveTimeSlab, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MultiAdaptiveTimeSlab):
        try:
            if self.thisown: destroy(self)
        except: pass

    def build(*args): return _dolfin.MultiAdaptiveTimeSlab_build(*args)
    def solve(*args): return _dolfin.MultiAdaptiveTimeSlab_solve(*args)
    def check(*args): return _dolfin.MultiAdaptiveTimeSlab_check(*args)
    def shift(*args): return _dolfin.MultiAdaptiveTimeSlab_shift(*args)
    def reset(*args): return _dolfin.MultiAdaptiveTimeSlab_reset(*args)
    def sample(*args): return _dolfin.MultiAdaptiveTimeSlab_sample(*args)
    def usample(*args): return _dolfin.MultiAdaptiveTimeSlab_usample(*args)
    def ksample(*args): return _dolfin.MultiAdaptiveTimeSlab_ksample(*args)
    def rsample(*args): return _dolfin.MultiAdaptiveTimeSlab_rsample(*args)
    def disp(*args): return _dolfin.MultiAdaptiveTimeSlab_disp(*args)

class MultiAdaptiveTimeSlabPtr(MultiAdaptiveTimeSlab):
    def __init__(self, this):
        _swig_setattr(self, MultiAdaptiveTimeSlab, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MultiAdaptiveTimeSlab, 'thisown', 0)
        self.__class__ = MultiAdaptiveTimeSlab
_dolfin.MultiAdaptiveTimeSlab_swigregister(MultiAdaptiveTimeSlabPtr)

class MultiAdaptivity(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivity, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptivity instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, MultiAdaptivity, 'this', _dolfin.new_MultiAdaptivity(*args))
        _swig_setattr(self, MultiAdaptivity, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_MultiAdaptivity):
        try:
            if self.thisown: destroy(self)
        except: pass

    def timestep(*args): return _dolfin.MultiAdaptivity_timestep(*args)
    def residual(*args): return _dolfin.MultiAdaptivity_residual(*args)
    def update(*args): return _dolfin.MultiAdaptivity_update(*args)

class MultiAdaptivityPtr(MultiAdaptivity):
    def __init__(self, this):
        _swig_setattr(self, MultiAdaptivity, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, MultiAdaptivity, 'thisown', 0)
        self.__class__ = MultiAdaptivity
_dolfin.MultiAdaptivity_swigregister(MultiAdaptivityPtr)

class ODE(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODE, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::ODE instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        if self.__class__ == ODE:
            args = (None,) + args
        else:
            args = (self,) + args
        _swig_setattr(self, ODE, 'this', _dolfin.new_ODE(*args))
        _swig_setattr(self, ODE, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_ODE):
        try:
            if self.thisown: destroy(self)
        except: pass

    def u0(*args): return _dolfin.ODE_u0(*args)
    def fmulti(*args): return _dolfin.ODE_fmulti(*args)
    def fmono(*args): return _dolfin.ODE_fmono(*args)
    def M(*args): return _dolfin.ODE_M(*args)
    def J(*args): return _dolfin.ODE_J(*args)
    def dfdu(*args): return _dolfin.ODE_dfdu(*args)
    def timestep(*args): return _dolfin.ODE_timestep(*args)
    def update(*args): return _dolfin.ODE_update(*args)
    def save(*args): return _dolfin.ODE_save(*args)
    def sparse(*args): return _dolfin.ODE_sparse(*args)
    def size(*args): return _dolfin.ODE_size(*args)
    def endtime(*args): return _dolfin.ODE_endtime(*args)
    def solve(*args): return _dolfin.ODE_solve(*args)
    def __disown__(self):
        self.thisown = 0
        _dolfin.disown_ODE(self)
        return weakref_proxy(self)

class ODEPtr(ODE):
    def __init__(self, this):
        _swig_setattr(self, ODE, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, ODE, 'thisown', 0)
        self.__class__ = ODE
_dolfin.ODE_swigregister(ODEPtr)

class ODESolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODESolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODESolver, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::ODESolver instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_getmethods__["solve"] = lambda x: _dolfin.ODESolver_solve
    if _newclass:solve = staticmethod(_dolfin.ODESolver_solve)

class ODESolverPtr(ODESolver):
    def __init__(self, this):
        _swig_setattr(self, ODESolver, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, ODESolver, 'thisown', 0)
        self.__class__ = ODESolver
_dolfin.ODESolver_swigregister(ODESolverPtr)

ODESolver_solve = _dolfin.ODESolver_solve

class ParticleSystem(ODE):
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, ParticleSystem, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, ParticleSystem, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::ParticleSystem instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, ParticleSystem, 'this', _dolfin.new_ParticleSystem(*args))
        _swig_setattr(self, ParticleSystem, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_ParticleSystem):
        try:
            if self.thisown: destroy(self)
        except: pass

    def x0(*args): return _dolfin.ParticleSystem_x0(*args)
    def y0(*args): return _dolfin.ParticleSystem_y0(*args)
    def z0(*args): return _dolfin.ParticleSystem_z0(*args)
    def vx0(*args): return _dolfin.ParticleSystem_vx0(*args)
    def vy0(*args): return _dolfin.ParticleSystem_vy0(*args)
    def vz0(*args): return _dolfin.ParticleSystem_vz0(*args)
    def Fx(*args): return _dolfin.ParticleSystem_Fx(*args)
    def Fy(*args): return _dolfin.ParticleSystem_Fy(*args)
    def Fz(*args): return _dolfin.ParticleSystem_Fz(*args)
    def mass(*args): return _dolfin.ParticleSystem_mass(*args)
    def k(*args): return _dolfin.ParticleSystem_k(*args)
    def u0(*args): return _dolfin.ParticleSystem_u0(*args)
    def f(*args): return _dolfin.ParticleSystem_f(*args)
    def timestep(*args): return _dolfin.ParticleSystem_timestep(*args)

class ParticleSystemPtr(ParticleSystem):
    def __init__(self, this):
        _swig_setattr(self, ParticleSystem, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, ParticleSystem, 'thisown', 0)
        self.__class__ = ParticleSystem
_dolfin.ParticleSystem_swigregister(ParticleSystemPtr)

class Partition(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Partition, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Partition, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Partition instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Partition, 'this', _dolfin.new_Partition(*args))
        _swig_setattr(self, Partition, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Partition):
        try:
            if self.thisown: destroy(self)
        except: pass

    def size(*args): return _dolfin.Partition_size(*args)
    def index(*args): return _dolfin.Partition_index(*args)
    def update(*args): return _dolfin.Partition_update(*args)
    def debug(*args): return _dolfin.Partition_debug(*args)

class PartitionPtr(Partition):
    def __init__(self, this):
        _swig_setattr(self, Partition, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Partition, 'thisown', 0)
        self.__class__ = Partition
_dolfin.Partition_swigregister(PartitionPtr)

class Sample(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Sample, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Sample, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::Sample instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Sample, 'this', _dolfin.new_Sample(*args))
        _swig_setattr(self, Sample, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_Sample):
        try:
            if self.thisown: destroy(self)
        except: pass

    def size(*args): return _dolfin.Sample_size(*args)
    def t(*args): return _dolfin.Sample_t(*args)
    def u(*args): return _dolfin.Sample_u(*args)
    def k(*args): return _dolfin.Sample_k(*args)
    def r(*args): return _dolfin.Sample_r(*args)

class SamplePtr(Sample):
    def __init__(self, this):
        _swig_setattr(self, Sample, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Sample, 'thisown', 0)
        self.__class__ = Sample
_dolfin.Sample_swigregister(SamplePtr)

class TimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlab, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeSlab instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_TimeSlab):
        try:
            if self.thisown: destroy(self)
        except: pass

    def build(*args): return _dolfin.TimeSlab_build(*args)
    def solve(*args): return _dolfin.TimeSlab_solve(*args)
    def check(*args): return _dolfin.TimeSlab_check(*args)
    def shift(*args): return _dolfin.TimeSlab_shift(*args)
    def sample(*args): return _dolfin.TimeSlab_sample(*args)
    def size(*args): return _dolfin.TimeSlab_size(*args)
    def starttime(*args): return _dolfin.TimeSlab_starttime(*args)
    def endtime(*args): return _dolfin.TimeSlab_endtime(*args)
    def length(*args): return _dolfin.TimeSlab_length(*args)
    def usample(*args): return _dolfin.TimeSlab_usample(*args)
    def ksample(*args): return _dolfin.TimeSlab_ksample(*args)
    def rsample(*args): return _dolfin.TimeSlab_rsample(*args)
    def disp(*args): return _dolfin.TimeSlab_disp(*args)

class TimeSlabPtr(TimeSlab):
    def __init__(self, this):
        _swig_setattr(self, TimeSlab, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeSlab, 'thisown', 0)
        self.__class__ = TimeSlab
_dolfin.TimeSlab_swigregister(TimeSlabPtr)

class TimeSlabJacobian(VirtualMatrix):
    __swig_setmethods__ = {}
    for _s in [VirtualMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlabJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [VirtualMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlabJacobian, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeSlabJacobian instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_dolfin.delete_TimeSlabJacobian):
        try:
            if self.thisown: destroy(self)
        except: pass

    def mult(*args): return _dolfin.TimeSlabJacobian_mult(*args)
    def update(*args): return _dolfin.TimeSlabJacobian_update(*args)

class TimeSlabJacobianPtr(TimeSlabJacobian):
    def __init__(self, this):
        _swig_setattr(self, TimeSlabJacobian, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeSlabJacobian, 'thisown', 0)
        self.__class__ = TimeSlabJacobian
_dolfin.TimeSlabJacobian_swigregister(TimeSlabJacobianPtr)

class TimeStepper(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeStepper, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeStepper, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeStepper instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, TimeStepper, 'this', _dolfin.new_TimeStepper(*args))
        _swig_setattr(self, TimeStepper, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_TimeStepper):
        try:
            if self.thisown: destroy(self)
        except: pass

    __swig_getmethods__["solve"] = lambda x: _dolfin.TimeStepper_solve
    if _newclass:solve = staticmethod(_dolfin.TimeStepper_solve)
    def step(*args): return _dolfin.TimeStepper_step(*args)
    def finished(*args): return _dolfin.TimeStepper_finished(*args)

class TimeStepperPtr(TimeStepper):
    def __init__(self, this):
        _swig_setattr(self, TimeStepper, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeStepper, 'thisown', 0)
        self.__class__ = TimeStepper
_dolfin.TimeStepper_swigregister(TimeStepperPtr)

TimeStepper_solve = _dolfin.TimeStepper_solve

class cGqMethod(Method):
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, cGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, cGqMethod, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::cGqMethod instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, cGqMethod, 'this', _dolfin.new_cGqMethod(*args))
        _swig_setattr(self, cGqMethod, 'thisown', 1)
    def ueval(*args): return _dolfin.cGqMethod_ueval(*args)
    def residual(*args): return _dolfin.cGqMethod_residual(*args)
    def timestep(*args): return _dolfin.cGqMethod_timestep(*args)
    def error(*args): return _dolfin.cGqMethod_error(*args)
    def disp(*args): return _dolfin.cGqMethod_disp(*args)

class cGqMethodPtr(cGqMethod):
    def __init__(self, this):
        _swig_setattr(self, cGqMethod, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, cGqMethod, 'thisown', 0)
        self.__class__ = cGqMethod
_dolfin.cGqMethod_swigregister(cGqMethodPtr)

class dGqMethod(Method):
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, dGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, dGqMethod, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::dGqMethod instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, dGqMethod, 'this', _dolfin.new_dGqMethod(*args))
        _swig_setattr(self, dGqMethod, 'thisown', 1)
    def ueval(*args): return _dolfin.dGqMethod_ueval(*args)
    def residual(*args): return _dolfin.dGqMethod_residual(*args)
    def timestep(*args): return _dolfin.dGqMethod_timestep(*args)
    def error(*args): return _dolfin.dGqMethod_error(*args)
    def disp(*args): return _dolfin.dGqMethod_disp(*args)

class dGqMethodPtr(dGqMethod):
    def __init__(self, this):
        _swig_setattr(self, dGqMethod, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, dGqMethod, 'thisown', 0)
        self.__class__ = dGqMethod
_dolfin.dGqMethod_swigregister(dGqMethodPtr)

class TimeDependentPDE(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependentPDE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependentPDE, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeDependentPDE instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        if self.__class__ == TimeDependentPDE:
            args = (None,) + args
        else:
            args = (self,) + args
        _swig_setattr(self, TimeDependentPDE, 'this', _dolfin.new_TimeDependentPDE(*args))
        _swig_setattr(self, TimeDependentPDE, 'thisown', 1)
    def __del__(self, destroy=_dolfin.delete_TimeDependentPDE):
        try:
            if self.thisown: destroy(self)
        except: pass

    def solve(*args): return _dolfin.TimeDependentPDE_solve(*args)
    def fu(*args): return _dolfin.TimeDependentPDE_fu(*args)
    def init(*args): return _dolfin.TimeDependentPDE_init(*args)
    def save(*args): return _dolfin.TimeDependentPDE_save(*args)
    def preparestep(*args): return _dolfin.TimeDependentPDE_preparestep(*args)
    def prepareiteration(*args): return _dolfin.TimeDependentPDE_prepareiteration(*args)
    def elementdim(*args): return _dolfin.TimeDependentPDE_elementdim(*args)
    def a(*args): return _dolfin.TimeDependentPDE_a(*args)
    def L(*args): return _dolfin.TimeDependentPDE_L(*args)
    def mesh(*args): return _dolfin.TimeDependentPDE_mesh(*args)
    def bc(*args): return _dolfin.TimeDependentPDE_bc(*args)
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
        self.thisown = 0
        _dolfin.disown_TimeDependentPDE(self)
        return weakref_proxy(self)

class TimeDependentPDEPtr(TimeDependentPDE):
    def __init__(self, this):
        _swig_setattr(self, TimeDependentPDE, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeDependentPDE, 'thisown', 0)
        self.__class__ = TimeDependentPDE
_dolfin.TimeDependentPDE_swigregister(TimeDependentPDEPtr)

class TimeDependentODE(ODE):
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeDependentODE, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TimeDependentODE, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ dolfin::TimeDependentODE instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, TimeDependentODE, 'this', _dolfin.new_TimeDependentODE(*args))
        _swig_setattr(self, TimeDependentODE, 'thisown', 1)
    def u0(*args): return _dolfin.TimeDependentODE_u0(*args)
    def timestep(*args): return _dolfin.TimeDependentODE_timestep(*args)
    def fmono(*args): return _dolfin.TimeDependentODE_fmono(*args)
    def update(*args): return _dolfin.TimeDependentODE_update(*args)

class TimeDependentODEPtr(TimeDependentODE):
    def __init__(self, this):
        _swig_setattr(self, TimeDependentODE, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TimeDependentODE, 'thisown', 0)
        self.__class__ = TimeDependentODE
_dolfin.TimeDependentODE_swigregister(TimeDependentODEPtr)


get = _dolfin.get

load_parameters = _dolfin.load_parameters

assemble = _dolfin.assemble


set = _dolfin.set

