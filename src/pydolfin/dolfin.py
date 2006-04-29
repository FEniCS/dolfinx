# This file was created automatically by SWIG 1.3.28.
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
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::TimeDependent instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_TimeDependent(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_TimeDependent
    __del__ = lambda self : None;
    def sync(*args): return _dolfin.TimeDependent_sync(*args)
    def time(*args): return _dolfin.TimeDependent_time(*args)
_dolfin.TimeDependent_swigregister(TimeDependent)

class Variable(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Variable, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Variable, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Variable instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Variable(*args)
        try: self.this.append(this)
        except: self.this = this
    def rename(*args): return _dolfin.Variable_rename(*args)
    def name(*args): return _dolfin.Variable_name(*args)
    def label(*args): return _dolfin.Variable_label(*args)
    def number(*args): return _dolfin.Variable_number(*args)
_dolfin.Variable_swigregister(Variable)


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
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Parameter instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    type_real = _dolfin.Parameter_type_real
    type_int = _dolfin.Parameter_type_int
    type_bool = _dolfin.Parameter_type_bool
    type_string = _dolfin.Parameter_type_string
    def __init__(self, *args):
        this = _dolfin.new_Parameter(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Parameter
    __del__ = lambda self : None;
    def type(*args): return _dolfin.Parameter_type(*args)
_dolfin.Parameter_swigregister(Parameter)

dolfin_begin = _dolfin.dolfin_begin

dolfin_end = _dolfin.dolfin_end

class File(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, File, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, File, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::File instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
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
        this = _dolfin.new_File(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_File
    __del__ = lambda self : None;
    def __rshift__(*args): return _dolfin.File___rshift__(*args)
    def __lshift__(*args): return _dolfin.File___lshift__(*args)
_dolfin.File_swigregister(File)

class Vector(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Vector instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Vector
    __del__ = lambda self : None;
    def init(*args): return _dolfin.Vector_init(*args)
    def clear(*args): return _dolfin.Vector_clear(*args)
    def size(*args): return _dolfin.Vector_size(*args)
    def vec(*args): return _dolfin.Vector_vec(*args)
    def array(*args): return _dolfin.Vector_array(*args)
    def restore(*args): return _dolfin.Vector_restore(*args)
    def axpy(*args): return _dolfin.Vector_axpy(*args)
    def div(*args): return _dolfin.Vector_div(*args)
    def mult(*args): return _dolfin.Vector_mult(*args)
    def add(*args): return _dolfin.Vector_add(*args)
    def insert(*args): return _dolfin.Vector_insert(*args)
    def get(*args): return _dolfin.Vector_get(*args)
    def apply(*args): return _dolfin.Vector_apply(*args)
    def __call__(*args): return _dolfin.Vector___call__(*args)
    def copy(*args): return _dolfin.Vector_copy(*args)
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
    def __getitem__(*args): return _dolfin.Vector___getitem__(*args)
    def __setitem__(*args): return _dolfin.Vector___setitem__(*args)
    def addval(*args): return _dolfin.Vector_addval(*args)
_dolfin.Vector_swigregister(Vector)

class VectorElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorElement, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::VectorElement instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_VectorElement(*args)
        try: self.this.append(this)
        except: self.this = this
    def __iadd__(*args): return _dolfin.VectorElement___iadd__(*args)
    def __isub__(*args): return _dolfin.VectorElement___isub__(*args)
    def __imul__(*args): return _dolfin.VectorElement___imul__(*args)
_dolfin.VectorElement_swigregister(VectorElement)

class Matrix(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Matrix, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Matrix, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Matrix instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    default_matrix = _dolfin.Matrix_default_matrix
    spooles = _dolfin.Matrix_spooles
    superlu = _dolfin.Matrix_superlu
    umfpack = _dolfin.Matrix_umfpack
    def __init__(self, *args):
        this = _dolfin.new_Matrix(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Matrix
    __del__ = lambda self : None;
    def init(*args): return _dolfin.Matrix_init(*args)
    def size(*args): return _dolfin.Matrix_size(*args)
    def nz(*args): return _dolfin.Matrix_nz(*args)
    def nzsum(*args): return _dolfin.Matrix_nzsum(*args)
    def nzmax(*args): return _dolfin.Matrix_nzmax(*args)
    def add(*args): return _dolfin.Matrix_add(*args)
    def ident(*args): return _dolfin.Matrix_ident(*args)
    def mult(*args): return _dolfin.Matrix_mult(*args)
    l1 = _dolfin.Matrix_l1
    linf = _dolfin.Matrix_linf
    frobenius = _dolfin.Matrix_frobenius
    def norm(*args): return _dolfin.Matrix_norm(*args)
    def apply(*args): return _dolfin.Matrix_apply(*args)
    def getMatrixType(*args): return _dolfin.Matrix_getMatrixType(*args)
    def mat(*args): return _dolfin.Matrix_mat(*args)
    def disp(*args): return _dolfin.Matrix_disp(*args)
    def __call__(*args): return _dolfin.Matrix___call__(*args)
    def getval(*args): return _dolfin.Matrix_getval(*args)
    def setval(*args): return _dolfin.Matrix_setval(*args)
    def addval(*args): return _dolfin.Matrix_addval(*args)
_dolfin.Matrix_swigregister(Matrix)

class MatrixElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MatrixElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MatrixElement, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MatrixElement instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MatrixElement(*args)
        try: self.this.append(this)
        except: self.this = this
    def __iadd__(*args): return _dolfin.MatrixElement___iadd__(*args)
    def __isub__(*args): return _dolfin.MatrixElement___isub__(*args)
    def __imul__(*args): return _dolfin.MatrixElement___imul__(*args)
_dolfin.MatrixElement_swigregister(MatrixElement)

class VirtualMatrix(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VirtualMatrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VirtualMatrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::VirtualMatrix instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_VirtualMatrix
    __del__ = lambda self : None;
    def init(*args): return _dolfin.VirtualMatrix_init(*args)
    def size(*args): return _dolfin.VirtualMatrix_size(*args)
    def mat(*args): return _dolfin.VirtualMatrix_mat(*args)
    def mult(*args): return _dolfin.VirtualMatrix_mult(*args)
    def disp(*args): return _dolfin.VirtualMatrix_disp(*args)
_dolfin.VirtualMatrix_swigregister(VirtualMatrix)

class GMRES(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GMRES, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GMRES, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::GMRES instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_GMRES(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_GMRES
    __del__ = lambda self : None;
_dolfin.GMRES_swigregister(GMRES)

class LinearSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, LinearSolver, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::LinearSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_LinearSolver
    __del__ = lambda self : None;
    def solve(*args): return _dolfin.LinearSolver_solve(*args)
_dolfin.LinearSolver_swigregister(LinearSolver)

class LU(LinearSolver):
    __swig_setmethods__ = {}
    for _s in [LinearSolver]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, LU, name, value)
    __swig_getmethods__ = {}
    for _s in [LinearSolver]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, LU, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::LU instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_LU(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_LU
    __del__ = lambda self : None;
    def solve(*args): return _dolfin.LU_solve(*args)
    def disp(*args): return _dolfin.LU_disp(*args)
_dolfin.LU_swigregister(LU)

class KrylovSolver(LinearSolver):
    __swig_setmethods__ = {}
    for _s in [LinearSolver]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, KrylovSolver, name, value)
    __swig_getmethods__ = {}
    for _s in [LinearSolver]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, KrylovSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::KrylovSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    bicgstab = _dolfin.KrylovSolver_bicgstab
    cg = _dolfin.KrylovSolver_cg
    default_solver = _dolfin.KrylovSolver_default_solver
    gmres = _dolfin.KrylovSolver_gmres
    def __init__(self, *args):
        this = _dolfin.new_KrylovSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_KrylovSolver
    __del__ = lambda self : None;
    def solve(*args): return _dolfin.KrylovSolver_solve(*args)
    def disp(*args): return _dolfin.KrylovSolver_disp(*args)
_dolfin.KrylovSolver_swigregister(KrylovSolver)

class EigenvalueSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EigenvalueSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EigenvalueSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::EigenvalueSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_EigenvalueSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_EigenvalueSolver
    __del__ = lambda self : None;
    def eigen(*args): return _dolfin.EigenvalueSolver_eigen(*args)
_dolfin.EigenvalueSolver_swigregister(EigenvalueSolver)

class Preconditioner(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Preconditioner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Preconditioner, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Preconditioner instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    default_pc = _dolfin.Preconditioner_default_pc
    hypre_amg = _dolfin.Preconditioner_hypre_amg
    icc = _dolfin.Preconditioner_icc
    ilu = _dolfin.Preconditioner_ilu
    jacobi = _dolfin.Preconditioner_jacobi
    sor = _dolfin.Preconditioner_sor
    none = _dolfin.Preconditioner_none
    __swig_destroy__ = _dolfin.delete_Preconditioner
    __del__ = lambda self : None;
    __swig_getmethods__["setup"] = lambda x: _dolfin.Preconditioner_setup
    if _newclass:setup = staticmethod(_dolfin.Preconditioner_setup)
    def solve(*args): return _dolfin.Preconditioner_solve(*args)
_dolfin.Preconditioner_swigregister(Preconditioner)

Preconditioner_setup = _dolfin.Preconditioner_setup

class PETScManager(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PETScManager, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PETScManager, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::PETScManager instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_getmethods__["init"] = lambda x: _dolfin.PETScManager_init
    if _newclass:init = staticmethod(_dolfin.PETScManager_init)
_dolfin.PETScManager_swigregister(PETScManager)

PETScManager_init = _dolfin.PETScManager_init

class Function(Variable,TimeDependent):
    __swig_setmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Function, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable,TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Function, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Function instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        if self.__class__ == Function:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_Function(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Function
    __del__ = lambda self : None;
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
        self.this.disown()
        _dolfin.disown_Function(self)
        return weakref_proxy(self)
_dolfin.Function_swigregister(Function)

class FEM(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FEM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FEM, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::FEM instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_getmethods__["size"] = lambda x: _dolfin.FEM_size
    if _newclass:size = staticmethod(_dolfin.FEM_size)
    __swig_getmethods__["lump"] = lambda x: _dolfin.FEM_lump
    if _newclass:lump = staticmethod(_dolfin.FEM_lump)
    __swig_getmethods__["disp"] = lambda x: _dolfin.FEM_disp
    if _newclass:disp = staticmethod(_dolfin.FEM_disp)
_dolfin.FEM_swigregister(FEM)

FEM_size = _dolfin.FEM_size

FEM_lump = _dolfin.FEM_lump

FEM_disp = _dolfin.FEM_disp

class FiniteElement(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FiniteElement, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FiniteElement, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::FiniteElement instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_FiniteElement
    __del__ = lambda self : None;
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
_dolfin.FiniteElement_swigregister(FiniteElement)

FiniteElement_makeElement = _dolfin.FiniteElement_makeElement

class AffineMap(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, AffineMap, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, AffineMap, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::AffineMap instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_AffineMap(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_AffineMap
    __del__ = lambda self : None;
    def update(*args): return _dolfin.AffineMap_update(*args)
    def __call__(*args): return _dolfin.AffineMap___call__(*args)
    def cell(*args): return _dolfin.AffineMap_cell(*args)
    __swig_setmethods__["det"] = _dolfin.AffineMap_det_set
    __swig_getmethods__["det"] = _dolfin.AffineMap_det_get
    if _newclass:det = property(_dolfin.AffineMap_det_get, _dolfin.AffineMap_det_set)
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
_dolfin.AffineMap_swigregister(AffineMap)

class BoundaryValue(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryValue, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryValue, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::BoundaryValue instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_BoundaryValue(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_BoundaryValue
    __del__ = lambda self : None;
    def set(*args): return _dolfin.BoundaryValue_set(*args)
    def reset(*args): return _dolfin.BoundaryValue_reset(*args)
_dolfin.BoundaryValue_swigregister(BoundaryValue)

class BoundaryCondition(TimeDependent):
    __swig_setmethods__ = {}
    for _s in [TimeDependent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoundaryCondition, name, value)
    __swig_getmethods__ = {}
    for _s in [TimeDependent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BoundaryCondition, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::BoundaryCondition instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        if self.__class__ == BoundaryCondition:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_BoundaryCondition(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_BoundaryCondition
    __del__ = lambda self : None;
    def eval(*args): return _dolfin.BoundaryCondition_eval(*args)
    def __disown__(self):
        self.this.disown()
        _dolfin.disown_BoundaryCondition(self)
        return weakref_proxy(self)
_dolfin.BoundaryCondition_swigregister(BoundaryCondition)

class Form(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Form, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Form, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Form instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Form(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Form
    __del__ = lambda self : None;
    def update(*args): return _dolfin.Form_update(*args)
_dolfin.Form_swigregister(Form)

class BilinearForm(Form):
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BilinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BilinearForm, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::BilinearForm instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_BilinearForm
    __del__ = lambda self : None;
    def eval(*args): return _dolfin.BilinearForm_eval(*args)
    def test(*args): return _dolfin.BilinearForm_test(*args)
    def trial(*args): return _dolfin.BilinearForm_trial(*args)
_dolfin.BilinearForm_swigregister(BilinearForm)

class LinearForm(Form):
    __swig_setmethods__ = {}
    for _s in [Form]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearForm, name, value)
    __swig_getmethods__ = {}
    for _s in [Form]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, LinearForm, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::LinearForm instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_LinearForm
    __del__ = lambda self : None;
    def eval(*args): return _dolfin.LinearForm_eval(*args)
    def test(*args): return _dolfin.LinearForm_test(*args)
_dolfin.LinearForm_swigregister(LinearForm)

class Mesh(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Mesh, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Mesh, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Mesh instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    triangles = _dolfin.Mesh_triangles
    tetrahedra = _dolfin.Mesh_tetrahedra
    def __init__(self, *args):
        this = _dolfin.new_Mesh(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Mesh
    __del__ = lambda self : None;
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
_dolfin.Mesh_swigregister(Mesh)

class Boundary(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Boundary, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Boundary, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Boundary instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Boundary(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Boundary
    __del__ = lambda self : None;
    def numVertices(*args): return _dolfin.Boundary_numVertices(*args)
    def numEdges(*args): return _dolfin.Boundary_numEdges(*args)
    def numFaces(*args): return _dolfin.Boundary_numFaces(*args)
_dolfin.Boundary_swigregister(Boundary)

class Point(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Point, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Point, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Point instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Point(*args)
        try: self.this.append(this)
        except: self.this = this
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
_dolfin.Point_swigregister(Point)

class Vertex(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vertex, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vertex, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Vertex instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Vertex(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Vertex
    __del__ = lambda self : None;
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
_dolfin.Vertex_swigregister(Vertex)

class Edge(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Edge, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Edge, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Edge instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Edge(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Edge
    __del__ = lambda self : None;
    def clear(*args): return _dolfin.Edge_clear(*args)
    def id(*args): return _dolfin.Edge_id(*args)
    def numCellNeighbors(*args): return _dolfin.Edge_numCellNeighbors(*args)
    def vertex(*args): return _dolfin.Edge_vertex(*args)
    def cell(*args): return _dolfin.Edge_cell(*args)
    def mesh(*args): return _dolfin.Edge_mesh(*args)
    def coord(*args): return _dolfin.Edge_coord(*args)
    def length(*args): return _dolfin.Edge_length(*args)
    def midpoint(*args): return _dolfin.Edge_midpoint(*args)
    def equals(*args): return _dolfin.Edge_equals(*args)
    def contains(*args): return _dolfin.Edge_contains(*args)
    __swig_setmethods__["ebids"] = _dolfin.Edge_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.Edge_ebids_get
    if _newclass:ebids = property(_dolfin.Edge_ebids_get, _dolfin.Edge_ebids_set)
_dolfin.Edge_swigregister(Edge)

class Triangle(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Triangle, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Triangle, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Triangle instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Triangle(*args)
        try: self.this.append(this)
        except: self.this = this
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
_dolfin.Triangle_swigregister(Triangle)

class Tetrahedron(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Tetrahedron, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Tetrahedron, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Tetrahedron instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Tetrahedron(*args)
        try: self.this.append(this)
        except: self.this = this
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
_dolfin.Tetrahedron_swigregister(Tetrahedron)

class Cell(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Cell, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Cell, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Cell instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    triangle = _dolfin.Cell_triangle
    tetrahedron = _dolfin.Cell_tetrahedron
    none = _dolfin.Cell_none
    left = _dolfin.Cell_left
    right = _dolfin.Cell_right
    def __init__(self, *args):
        this = _dolfin.new_Cell(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Cell
    __del__ = lambda self : None;
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
_dolfin.Cell_swigregister(Cell)

class Face(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Face, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Face, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Face instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Face(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Face
    __del__ = lambda self : None;
    def clear(*args): return _dolfin.Face_clear(*args)
    def id(*args): return _dolfin.Face_id(*args)
    def numEdges(*args): return _dolfin.Face_numEdges(*args)
    def numCellNeighbors(*args): return _dolfin.Face_numCellNeighbors(*args)
    def edge(*args): return _dolfin.Face_edge(*args)
    def cell(*args): return _dolfin.Face_cell(*args)
    def mesh(*args): return _dolfin.Face_mesh(*args)
    def equals(*args): return _dolfin.Face_equals(*args)
    def contains(*args): return _dolfin.Face_contains(*args)
    __swig_setmethods__["fbids"] = _dolfin.Face_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.Face_fbids_get
    if _newclass:fbids = property(_dolfin.Face_fbids_get, _dolfin.Face_fbids_set)
_dolfin.Face_swigregister(Face)

class VertexIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VertexIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VertexIterator, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::VertexIterator instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_VertexIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_VertexIterator
    __del__ = lambda self : None;
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
_dolfin.VertexIterator_swigregister(VertexIterator)

class CellIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CellIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CellIterator, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::CellIterator instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_CellIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_CellIterator
    __del__ = lambda self : None;
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
_dolfin.CellIterator_swigregister(CellIterator)

class EdgeIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EdgeIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EdgeIterator, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::EdgeIterator instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_EdgeIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_EdgeIterator
    __del__ = lambda self : None;
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
    def mesh(*args): return _dolfin.EdgeIterator_mesh(*args)
    def coord(*args): return _dolfin.EdgeIterator_coord(*args)
    def length(*args): return _dolfin.EdgeIterator_length(*args)
    def midpoint(*args): return _dolfin.EdgeIterator_midpoint(*args)
    def equals(*args): return _dolfin.EdgeIterator_equals(*args)
    def contains(*args): return _dolfin.EdgeIterator_contains(*args)
    __swig_setmethods__["ebids"] = _dolfin.EdgeIterator_ebids_set
    __swig_getmethods__["ebids"] = _dolfin.EdgeIterator_ebids_get
    if _newclass:ebids = property(_dolfin.EdgeIterator_ebids_get, _dolfin.EdgeIterator_ebids_set)
_dolfin.EdgeIterator_swigregister(EdgeIterator)

class FaceIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FaceIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FaceIterator, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::FaceIterator instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_FaceIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_FaceIterator
    __del__ = lambda self : None;
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
    def mesh(*args): return _dolfin.FaceIterator_mesh(*args)
    def equals(*args): return _dolfin.FaceIterator_equals(*args)
    def contains(*args): return _dolfin.FaceIterator_contains(*args)
    __swig_setmethods__["fbids"] = _dolfin.FaceIterator_fbids_set
    __swig_getmethods__["fbids"] = _dolfin.FaceIterator_fbids_get
    if _newclass:fbids = property(_dolfin.FaceIterator_fbids_get, _dolfin.FaceIterator_fbids_set)
_dolfin.FaceIterator_swigregister(FaceIterator)

class MeshIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MeshIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MeshIterator, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MeshIterator instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MeshIterator(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MeshIterator
    __del__ = lambda self : None;
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
_dolfin.MeshIterator_swigregister(MeshIterator)

class UnitSquare(Mesh):
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitSquare, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitSquare, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::UnitSquare instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_UnitSquare(*args)
        try: self.this.append(this)
        except: self.this = this
_dolfin.UnitSquare_swigregister(UnitSquare)

class UnitCube(Mesh):
    __swig_setmethods__ = {}
    for _s in [Mesh]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnitCube, name, value)
    __swig_getmethods__ = {}
    for _s in [Mesh]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UnitCube, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::UnitCube instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_UnitCube(*args)
        try: self.this.append(this)
        except: self.this = this
_dolfin.UnitCube_swigregister(UnitCube)

class Dependencies(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Dependencies, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Dependencies, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Dependencies instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Dependencies(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Dependencies
    __del__ = lambda self : None;
    def setsize(*args): return _dolfin.Dependencies_setsize(*args)
    def set(*args): return _dolfin.Dependencies_set(*args)
    def transp(*args): return _dolfin.Dependencies_transp(*args)
    def detect(*args): return _dolfin.Dependencies_detect(*args)
    def sparse(*args): return _dolfin.Dependencies_sparse(*args)
    def disp(*args): return _dolfin.Dependencies_disp(*args)
_dolfin.Dependencies_swigregister(Dependencies)

class Homotopy(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Homotopy, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Homotopy, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Homotopy instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_Homotopy
    __del__ = lambda self : None;
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
_dolfin.Homotopy_swigregister(Homotopy)

class HomotopyJacobian(VirtualMatrix):
    __swig_setmethods__ = {}
    for _s in [VirtualMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [VirtualMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyJacobian, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::HomotopyJacobian instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_HomotopyJacobian(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_HomotopyJacobian
    __del__ = lambda self : None;
    def mult(*args): return _dolfin.HomotopyJacobian_mult(*args)
_dolfin.HomotopyJacobian_swigregister(HomotopyJacobian)

class HomotopyODE(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HomotopyODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HomotopyODE, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::HomotopyODE instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    ode = _dolfin.HomotopyODE_ode
    endgame = _dolfin.HomotopyODE_endgame
    def __init__(self, *args):
        this = _dolfin.new_HomotopyODE(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_HomotopyODE
    __del__ = lambda self : None;
    def z0(*args): return _dolfin.HomotopyODE_z0(*args)
    def f(*args): return _dolfin.HomotopyODE_f(*args)
    def M(*args): return _dolfin.HomotopyODE_M(*args)
    def J(*args): return _dolfin.HomotopyODE_J(*args)
    def update(*args): return _dolfin.HomotopyODE_update(*args)
    def state(*args): return _dolfin.HomotopyODE_state(*args)
_dolfin.HomotopyODE_swigregister(HomotopyODE)

class Method(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Method, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Method, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Method instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    cG = _dolfin.Method_cG
    dG = _dolfin.Method_dG
    none = _dolfin.Method_none
    __swig_destroy__ = _dolfin.delete_Method
    __del__ = lambda self : None;
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
_dolfin.Method_swigregister(Method)

class MonoAdaptiveFixedPointSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveFixedPointSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveFixedPointSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MonoAdaptiveFixedPointSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveFixedPointSolver
    __del__ = lambda self : None;
_dolfin.MonoAdaptiveFixedPointSolver_swigregister(MonoAdaptiveFixedPointSolver)

class MonoAdaptiveJacobian(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveJacobian, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveJacobian, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveJacobian instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MonoAdaptiveJacobian(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveJacobian
    __del__ = lambda self : None;
    def mult(*args): return _dolfin.MonoAdaptiveJacobian_mult(*args)
_dolfin.MonoAdaptiveJacobian_swigregister(MonoAdaptiveJacobian)

class MonoAdaptiveNewtonSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveNewtonSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveNewtonSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MonoAdaptiveNewtonSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveNewtonSolver
    __del__ = lambda self : None;
_dolfin.MonoAdaptiveNewtonSolver_swigregister(MonoAdaptiveNewtonSolver)

class MonoAdaptiveTimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptiveTimeSlab, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptiveTimeSlab instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MonoAdaptiveTimeSlab(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptiveTimeSlab
    __del__ = lambda self : None;
    def build(*args): return _dolfin.MonoAdaptiveTimeSlab_build(*args)
    def solve(*args): return _dolfin.MonoAdaptiveTimeSlab_solve(*args)
    def check(*args): return _dolfin.MonoAdaptiveTimeSlab_check(*args)
    def shift(*args): return _dolfin.MonoAdaptiveTimeSlab_shift(*args)
    def sample(*args): return _dolfin.MonoAdaptiveTimeSlab_sample(*args)
    def usample(*args): return _dolfin.MonoAdaptiveTimeSlab_usample(*args)
    def ksample(*args): return _dolfin.MonoAdaptiveTimeSlab_ksample(*args)
    def rsample(*args): return _dolfin.MonoAdaptiveTimeSlab_rsample(*args)
    def disp(*args): return _dolfin.MonoAdaptiveTimeSlab_disp(*args)
_dolfin.MonoAdaptiveTimeSlab_swigregister(MonoAdaptiveTimeSlab)

class MonoAdaptivity(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MonoAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MonoAdaptivity, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MonoAdaptivity instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MonoAdaptivity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MonoAdaptivity
    __del__ = lambda self : None;
    def timestep(*args): return _dolfin.MonoAdaptivity_timestep(*args)
    def update(*args): return _dolfin.MonoAdaptivity_update(*args)
_dolfin.MonoAdaptivity_swigregister(MonoAdaptivity)

class MultiAdaptiveFixedPointSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveFixedPointSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveFixedPointSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveFixedPointSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MultiAdaptiveFixedPointSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveFixedPointSolver
    __del__ = lambda self : None;
_dolfin.MultiAdaptiveFixedPointSolver_swigregister(MultiAdaptiveFixedPointSolver)

class MultiAdaptivePreconditioner(Preconditioner):
    __swig_setmethods__ = {}
    for _s in [Preconditioner]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivePreconditioner, name, value)
    __swig_getmethods__ = {}
    for _s in [Preconditioner]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivePreconditioner, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptivePreconditioner instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MultiAdaptivePreconditioner(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptivePreconditioner
    __del__ = lambda self : None;
    def solve(*args): return _dolfin.MultiAdaptivePreconditioner_solve(*args)
_dolfin.MultiAdaptivePreconditioner_swigregister(MultiAdaptivePreconditioner)

class MultiAdaptiveNewtonSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveNewtonSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveNewtonSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveNewtonSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MultiAdaptiveNewtonSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveNewtonSolver
    __del__ = lambda self : None;
_dolfin.MultiAdaptiveNewtonSolver_swigregister(MultiAdaptiveNewtonSolver)

class MultiAdaptiveTimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptiveTimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptiveTimeSlab, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptiveTimeSlab instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MultiAdaptiveTimeSlab(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptiveTimeSlab
    __del__ = lambda self : None;
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
_dolfin.MultiAdaptiveTimeSlab_swigregister(MultiAdaptiveTimeSlab)

class MultiAdaptivity(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MultiAdaptivity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MultiAdaptivity, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::MultiAdaptivity instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_MultiAdaptivity(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_MultiAdaptivity
    __del__ = lambda self : None;
    def timestep(*args): return _dolfin.MultiAdaptivity_timestep(*args)
    def residual(*args): return _dolfin.MultiAdaptivity_residual(*args)
    def update(*args): return _dolfin.MultiAdaptivity_update(*args)
_dolfin.MultiAdaptivity_swigregister(MultiAdaptivity)

class ODE(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODE, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ODE instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        if self.__class__ == ODE:
            args = (None,) + args
        else:
            args = (self,) + args
        this = _dolfin.new_ODE(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_ODE
    __del__ = lambda self : None;
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
        self.this.disown()
        _dolfin.disown_ODE(self)
        return weakref_proxy(self)
_dolfin.ODE_swigregister(ODE)

class ODESolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ODESolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ODESolver, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ODESolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_getmethods__["solve"] = lambda x: _dolfin.ODESolver_solve
    if _newclass:solve = staticmethod(_dolfin.ODESolver_solve)
_dolfin.ODESolver_swigregister(ODESolver)

ODESolver_solve = _dolfin.ODESolver_solve

class ParticleSystem(ODE):
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, ParticleSystem, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, ParticleSystem, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ParticleSystem instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_ParticleSystem(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_ParticleSystem
    __del__ = lambda self : None;
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
_dolfin.ParticleSystem_swigregister(ParticleSystem)

class Partition(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Partition, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Partition, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Partition instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Partition(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Partition
    __del__ = lambda self : None;
    def size(*args): return _dolfin.Partition_size(*args)
    def index(*args): return _dolfin.Partition_index(*args)
    def update(*args): return _dolfin.Partition_update(*args)
    def debug(*args): return _dolfin.Partition_debug(*args)
_dolfin.Partition_swigregister(Partition)

class ReducedModel(ODE):
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, ReducedModel, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, ReducedModel, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ReducedModel instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_ReducedModel(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_ReducedModel
    __del__ = lambda self : None;
    def f(*args): return _dolfin.ReducedModel_f(*args)
    def u0(*args): return _dolfin.ReducedModel_u0(*args)
_dolfin.ReducedModel_swigregister(ReducedModel)

class Sample(Variable):
    __swig_setmethods__ = {}
    for _s in [Variable]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Sample, name, value)
    __swig_getmethods__ = {}
    for _s in [Variable]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Sample, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::Sample instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_Sample(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_Sample
    __del__ = lambda self : None;
    def size(*args): return _dolfin.Sample_size(*args)
    def t(*args): return _dolfin.Sample_t(*args)
    def u(*args): return _dolfin.Sample_u(*args)
    def k(*args): return _dolfin.Sample_k(*args)
    def r(*args): return _dolfin.Sample_r(*args)
_dolfin.Sample_swigregister(Sample)

class TimeSlab(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlab, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlab, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::TimeSlab instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_TimeSlab
    __del__ = lambda self : None;
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
_dolfin.TimeSlab_swigregister(TimeSlab)

class TimeSlabJacobian(VirtualMatrix):
    __swig_setmethods__ = {}
    for _s in [VirtualMatrix]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeSlabJacobian, name, value)
    __swig_getmethods__ = {}
    for _s in [VirtualMatrix]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TimeSlabJacobian, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::TimeSlabJacobian instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    __swig_destroy__ = _dolfin.delete_TimeSlabJacobian
    __del__ = lambda self : None;
    def mult(*args): return _dolfin.TimeSlabJacobian_mult(*args)
    def update(*args): return _dolfin.TimeSlabJacobian_update(*args)
_dolfin.TimeSlabJacobian_swigregister(TimeSlabJacobian)

class TimeStepper(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TimeStepper, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TimeStepper, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::TimeStepper instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_TimeStepper(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfin.delete_TimeStepper
    __del__ = lambda self : None;
    __swig_getmethods__["solve"] = lambda x: _dolfin.TimeStepper_solve
    if _newclass:solve = staticmethod(_dolfin.TimeStepper_solve)
    def step(*args): return _dolfin.TimeStepper_step(*args)
    def finished(*args): return _dolfin.TimeStepper_finished(*args)
_dolfin.TimeStepper_swigregister(TimeStepper)

TimeStepper_solve = _dolfin.TimeStepper_solve

class cGqMethod(Method):
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, cGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, cGqMethod, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::cGqMethod instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_cGqMethod(*args)
        try: self.this.append(this)
        except: self.this = this
    def ueval(*args): return _dolfin.cGqMethod_ueval(*args)
    def residual(*args): return _dolfin.cGqMethod_residual(*args)
    def timestep(*args): return _dolfin.cGqMethod_timestep(*args)
    def error(*args): return _dolfin.cGqMethod_error(*args)
    def disp(*args): return _dolfin.cGqMethod_disp(*args)
_dolfin.cGqMethod_swigregister(cGqMethod)

class dGqMethod(Method):
    __swig_setmethods__ = {}
    for _s in [Method]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, dGqMethod, name, value)
    __swig_getmethods__ = {}
    for _s in [Method]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, dGqMethod, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::dGqMethod instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_dGqMethod(*args)
        try: self.this.append(this)
        except: self.this = this
    def ueval(*args): return _dolfin.dGqMethod_ueval(*args)
    def residual(*args): return _dolfin.dGqMethod_residual(*args)
    def timestep(*args): return _dolfin.dGqMethod_timestep(*args)
    def error(*args): return _dolfin.dGqMethod_error(*args)
    def disp(*args): return _dolfin.dGqMethod_disp(*args)
_dolfin.dGqMethod_swigregister(dGqMethod)


get = _dolfin.get
class ElasticityUpdatedSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ElasticityUpdatedSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ElasticityUpdatedSolver, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ElasticityUpdatedSolver instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_ElasticityUpdatedSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    def init(*args): return _dolfin.ElasticityUpdatedSolver_init(*args)
    def step(*args): return _dolfin.ElasticityUpdatedSolver_step(*args)
    def oldstep(*args): return _dolfin.ElasticityUpdatedSolver_oldstep(*args)
    def fu(*args): return _dolfin.ElasticityUpdatedSolver_fu(*args)
    def gather(*args): return _dolfin.ElasticityUpdatedSolver_gather(*args)
    def preparestep(*args): return _dolfin.ElasticityUpdatedSolver_preparestep(*args)
    def prepareiteration(*args): return _dolfin.ElasticityUpdatedSolver_prepareiteration(*args)
    def save(*args): return _dolfin.ElasticityUpdatedSolver_save(*args)
    def condsave(*args): return _dolfin.ElasticityUpdatedSolver_condsave(*args)
    __swig_getmethods__["solve"] = lambda x: _dolfin.ElasticityUpdatedSolver_solve
    if _newclass:solve = staticmethod(_dolfin.ElasticityUpdatedSolver_solve)
    __swig_getmethods__["finterpolate"] = lambda x: _dolfin.ElasticityUpdatedSolver_finterpolate
    if _newclass:finterpolate = staticmethod(_dolfin.ElasticityUpdatedSolver_finterpolate)
    __swig_getmethods__["plasticity"] = lambda x: _dolfin.ElasticityUpdatedSolver_plasticity
    if _newclass:plasticity = staticmethod(_dolfin.ElasticityUpdatedSolver_plasticity)
    __swig_getmethods__["initmsigma"] = lambda x: _dolfin.ElasticityUpdatedSolver_initmsigma
    if _newclass:initmsigma = staticmethod(_dolfin.ElasticityUpdatedSolver_initmsigma)
    __swig_getmethods__["initu0"] = lambda x: _dolfin.ElasticityUpdatedSolver_initu0
    if _newclass:initu0 = staticmethod(_dolfin.ElasticityUpdatedSolver_initu0)
    __swig_getmethods__["initJ0"] = lambda x: _dolfin.ElasticityUpdatedSolver_initJ0
    if _newclass:initJ0 = staticmethod(_dolfin.ElasticityUpdatedSolver_initJ0)
    __swig_getmethods__["computeJ"] = lambda x: _dolfin.ElasticityUpdatedSolver_computeJ
    if _newclass:computeJ = staticmethod(_dolfin.ElasticityUpdatedSolver_computeJ)
    __swig_getmethods__["initF0Green"] = lambda x: _dolfin.ElasticityUpdatedSolver_initF0Green
    if _newclass:initF0Green = staticmethod(_dolfin.ElasticityUpdatedSolver_initF0Green)
    __swig_getmethods__["computeFGreen"] = lambda x: _dolfin.ElasticityUpdatedSolver_computeFGreen
    if _newclass:computeFGreen = staticmethod(_dolfin.ElasticityUpdatedSolver_computeFGreen)
    __swig_getmethods__["initF0Euler"] = lambda x: _dolfin.ElasticityUpdatedSolver_initF0Euler
    if _newclass:initF0Euler = staticmethod(_dolfin.ElasticityUpdatedSolver_initF0Euler)
    __swig_getmethods__["computeFEuler"] = lambda x: _dolfin.ElasticityUpdatedSolver_computeFEuler
    if _newclass:computeFEuler = staticmethod(_dolfin.ElasticityUpdatedSolver_computeFEuler)
    __swig_getmethods__["computeFBEuler"] = lambda x: _dolfin.ElasticityUpdatedSolver_computeFBEuler
    if _newclass:computeFBEuler = staticmethod(_dolfin.ElasticityUpdatedSolver_computeFBEuler)
    __swig_getmethods__["computeBEuler"] = lambda x: _dolfin.ElasticityUpdatedSolver_computeBEuler
    if _newclass:computeBEuler = staticmethod(_dolfin.ElasticityUpdatedSolver_computeBEuler)
    __swig_getmethods__["multF"] = lambda x: _dolfin.ElasticityUpdatedSolver_multF
    if _newclass:multF = staticmethod(_dolfin.ElasticityUpdatedSolver_multF)
    __swig_getmethods__["multB"] = lambda x: _dolfin.ElasticityUpdatedSolver_multB
    if _newclass:multB = staticmethod(_dolfin.ElasticityUpdatedSolver_multB)
    __swig_getmethods__["deform"] = lambda x: _dolfin.ElasticityUpdatedSolver_deform
    if _newclass:deform = staticmethod(_dolfin.ElasticityUpdatedSolver_deform)
    __swig_setmethods__["mesh"] = _dolfin.ElasticityUpdatedSolver_mesh_set
    __swig_getmethods__["mesh"] = _dolfin.ElasticityUpdatedSolver_mesh_get
    if _newclass:mesh = property(_dolfin.ElasticityUpdatedSolver_mesh_get, _dolfin.ElasticityUpdatedSolver_mesh_set)
    __swig_setmethods__["f"] = _dolfin.ElasticityUpdatedSolver_f_set
    __swig_getmethods__["f"] = _dolfin.ElasticityUpdatedSolver_f_get
    if _newclass:f = property(_dolfin.ElasticityUpdatedSolver_f_get, _dolfin.ElasticityUpdatedSolver_f_set)
    __swig_setmethods__["v0"] = _dolfin.ElasticityUpdatedSolver_v0_set
    __swig_getmethods__["v0"] = _dolfin.ElasticityUpdatedSolver_v0_get
    if _newclass:v0 = property(_dolfin.ElasticityUpdatedSolver_v0_get, _dolfin.ElasticityUpdatedSolver_v0_set)
    __swig_setmethods__["rho"] = _dolfin.ElasticityUpdatedSolver_rho_set
    __swig_getmethods__["rho"] = _dolfin.ElasticityUpdatedSolver_rho_get
    if _newclass:rho = property(_dolfin.ElasticityUpdatedSolver_rho_get, _dolfin.ElasticityUpdatedSolver_rho_set)
    __swig_setmethods__["E"] = _dolfin.ElasticityUpdatedSolver_E_set
    __swig_getmethods__["E"] = _dolfin.ElasticityUpdatedSolver_E_get
    if _newclass:E = property(_dolfin.ElasticityUpdatedSolver_E_get, _dolfin.ElasticityUpdatedSolver_E_set)
    __swig_setmethods__["nu"] = _dolfin.ElasticityUpdatedSolver_nu_set
    __swig_getmethods__["nu"] = _dolfin.ElasticityUpdatedSolver_nu_get
    if _newclass:nu = property(_dolfin.ElasticityUpdatedSolver_nu_get, _dolfin.ElasticityUpdatedSolver_nu_set)
    __swig_setmethods__["nuv"] = _dolfin.ElasticityUpdatedSolver_nuv_set
    __swig_getmethods__["nuv"] = _dolfin.ElasticityUpdatedSolver_nuv_get
    if _newclass:nuv = property(_dolfin.ElasticityUpdatedSolver_nuv_get, _dolfin.ElasticityUpdatedSolver_nuv_set)
    __swig_setmethods__["nuplast"] = _dolfin.ElasticityUpdatedSolver_nuplast_set
    __swig_getmethods__["nuplast"] = _dolfin.ElasticityUpdatedSolver_nuplast_get
    if _newclass:nuplast = property(_dolfin.ElasticityUpdatedSolver_nuplast_get, _dolfin.ElasticityUpdatedSolver_nuplast_set)
    __swig_setmethods__["bc"] = _dolfin.ElasticityUpdatedSolver_bc_set
    __swig_getmethods__["bc"] = _dolfin.ElasticityUpdatedSolver_bc_get
    if _newclass:bc = property(_dolfin.ElasticityUpdatedSolver_bc_get, _dolfin.ElasticityUpdatedSolver_bc_set)
    __swig_setmethods__["k"] = _dolfin.ElasticityUpdatedSolver_k_set
    __swig_getmethods__["k"] = _dolfin.ElasticityUpdatedSolver_k_get
    if _newclass:k = property(_dolfin.ElasticityUpdatedSolver_k_get, _dolfin.ElasticityUpdatedSolver_k_set)
    __swig_setmethods__["T"] = _dolfin.ElasticityUpdatedSolver_T_set
    __swig_getmethods__["T"] = _dolfin.ElasticityUpdatedSolver_T_get
    if _newclass:T = property(_dolfin.ElasticityUpdatedSolver_T_get, _dolfin.ElasticityUpdatedSolver_T_set)
    __swig_setmethods__["counter"] = _dolfin.ElasticityUpdatedSolver_counter_set
    __swig_getmethods__["counter"] = _dolfin.ElasticityUpdatedSolver_counter_get
    if _newclass:counter = property(_dolfin.ElasticityUpdatedSolver_counter_get, _dolfin.ElasticityUpdatedSolver_counter_set)
    __swig_setmethods__["lastsample"] = _dolfin.ElasticityUpdatedSolver_lastsample_set
    __swig_getmethods__["lastsample"] = _dolfin.ElasticityUpdatedSolver_lastsample_get
    if _newclass:lastsample = property(_dolfin.ElasticityUpdatedSolver_lastsample_get, _dolfin.ElasticityUpdatedSolver_lastsample_set)
    __swig_setmethods__["_lambda"] = _dolfin.ElasticityUpdatedSolver__lambda_set
    __swig_getmethods__["_lambda"] = _dolfin.ElasticityUpdatedSolver__lambda_get
    if _newclass:_lambda = property(_dolfin.ElasticityUpdatedSolver__lambda_get, _dolfin.ElasticityUpdatedSolver__lambda_set)
    __swig_setmethods__["mu"] = _dolfin.ElasticityUpdatedSolver_mu_set
    __swig_getmethods__["mu"] = _dolfin.ElasticityUpdatedSolver_mu_get
    if _newclass:mu = property(_dolfin.ElasticityUpdatedSolver_mu_get, _dolfin.ElasticityUpdatedSolver_mu_set)
    __swig_setmethods__["t"] = _dolfin.ElasticityUpdatedSolver_t_set
    __swig_getmethods__["t"] = _dolfin.ElasticityUpdatedSolver_t_get
    if _newclass:t = property(_dolfin.ElasticityUpdatedSolver_t_get, _dolfin.ElasticityUpdatedSolver_t_set)
    __swig_setmethods__["rtol"] = _dolfin.ElasticityUpdatedSolver_rtol_set
    __swig_getmethods__["rtol"] = _dolfin.ElasticityUpdatedSolver_rtol_get
    if _newclass:rtol = property(_dolfin.ElasticityUpdatedSolver_rtol_get, _dolfin.ElasticityUpdatedSolver_rtol_set)
    __swig_setmethods__["maxiters"] = _dolfin.ElasticityUpdatedSolver_maxiters_set
    __swig_getmethods__["maxiters"] = _dolfin.ElasticityUpdatedSolver_maxiters_get
    if _newclass:maxiters = property(_dolfin.ElasticityUpdatedSolver_maxiters_get, _dolfin.ElasticityUpdatedSolver_maxiters_set)
    __swig_setmethods__["do_plasticity"] = _dolfin.ElasticityUpdatedSolver_do_plasticity_set
    __swig_getmethods__["do_plasticity"] = _dolfin.ElasticityUpdatedSolver_do_plasticity_get
    if _newclass:do_plasticity = property(_dolfin.ElasticityUpdatedSolver_do_plasticity_get, _dolfin.ElasticityUpdatedSolver_do_plasticity_set)
    __swig_setmethods__["_yield"] = _dolfin.ElasticityUpdatedSolver__yield_set
    __swig_getmethods__["_yield"] = _dolfin.ElasticityUpdatedSolver__yield_get
    if _newclass:_yield = property(_dolfin.ElasticityUpdatedSolver__yield_get, _dolfin.ElasticityUpdatedSolver__yield_set)
    __swig_setmethods__["savesamplefreq"] = _dolfin.ElasticityUpdatedSolver_savesamplefreq_set
    __swig_getmethods__["savesamplefreq"] = _dolfin.ElasticityUpdatedSolver_savesamplefreq_get
    if _newclass:savesamplefreq = property(_dolfin.ElasticityUpdatedSolver_savesamplefreq_get, _dolfin.ElasticityUpdatedSolver_savesamplefreq_set)
    __swig_setmethods__["fevals"] = _dolfin.ElasticityUpdatedSolver_fevals_set
    __swig_getmethods__["fevals"] = _dolfin.ElasticityUpdatedSolver_fevals_get
    if _newclass:fevals = property(_dolfin.ElasticityUpdatedSolver_fevals_get, _dolfin.ElasticityUpdatedSolver_fevals_set)
    __swig_setmethods__["Nv"] = _dolfin.ElasticityUpdatedSolver_Nv_set
    __swig_getmethods__["Nv"] = _dolfin.ElasticityUpdatedSolver_Nv_get
    if _newclass:Nv = property(_dolfin.ElasticityUpdatedSolver_Nv_get, _dolfin.ElasticityUpdatedSolver_Nv_set)
    __swig_setmethods__["Nsigma"] = _dolfin.ElasticityUpdatedSolver_Nsigma_set
    __swig_getmethods__["Nsigma"] = _dolfin.ElasticityUpdatedSolver_Nsigma_get
    if _newclass:Nsigma = property(_dolfin.ElasticityUpdatedSolver_Nsigma_get, _dolfin.ElasticityUpdatedSolver_Nsigma_set)
    __swig_setmethods__["Nsigmanorm"] = _dolfin.ElasticityUpdatedSolver_Nsigmanorm_set
    __swig_getmethods__["Nsigmanorm"] = _dolfin.ElasticityUpdatedSolver_Nsigmanorm_get
    if _newclass:Nsigmanorm = property(_dolfin.ElasticityUpdatedSolver_Nsigmanorm_get, _dolfin.ElasticityUpdatedSolver_Nsigmanorm_set)
    __swig_setmethods__["ode"] = _dolfin.ElasticityUpdatedSolver_ode_set
    __swig_getmethods__["ode"] = _dolfin.ElasticityUpdatedSolver_ode_get
    if _newclass:ode = property(_dolfin.ElasticityUpdatedSolver_ode_get, _dolfin.ElasticityUpdatedSolver_ode_set)
    __swig_setmethods__["ts"] = _dolfin.ElasticityUpdatedSolver_ts_set
    __swig_getmethods__["ts"] = _dolfin.ElasticityUpdatedSolver_ts_get
    if _newclass:ts = property(_dolfin.ElasticityUpdatedSolver_ts_get, _dolfin.ElasticityUpdatedSolver_ts_set)
    __swig_setmethods__["element1"] = _dolfin.ElasticityUpdatedSolver_element1_set
    __swig_getmethods__["element1"] = _dolfin.ElasticityUpdatedSolver_element1_get
    if _newclass:element1 = property(_dolfin.ElasticityUpdatedSolver_element1_get, _dolfin.ElasticityUpdatedSolver_element1_set)
    __swig_setmethods__["element2"] = _dolfin.ElasticityUpdatedSolver_element2_set
    __swig_getmethods__["element2"] = _dolfin.ElasticityUpdatedSolver_element2_get
    if _newclass:element2 = property(_dolfin.ElasticityUpdatedSolver_element2_get, _dolfin.ElasticityUpdatedSolver_element2_set)
    __swig_setmethods__["element3"] = _dolfin.ElasticityUpdatedSolver_element3_set
    __swig_getmethods__["element3"] = _dolfin.ElasticityUpdatedSolver_element3_get
    if _newclass:element3 = property(_dolfin.ElasticityUpdatedSolver_element3_get, _dolfin.ElasticityUpdatedSolver_element3_set)
    __swig_setmethods__["x1_0"] = _dolfin.ElasticityUpdatedSolver_x1_0_set
    __swig_getmethods__["x1_0"] = _dolfin.ElasticityUpdatedSolver_x1_0_get
    if _newclass:x1_0 = property(_dolfin.ElasticityUpdatedSolver_x1_0_get, _dolfin.ElasticityUpdatedSolver_x1_0_set)
    __swig_setmethods__["x1_1"] = _dolfin.ElasticityUpdatedSolver_x1_1_set
    __swig_getmethods__["x1_1"] = _dolfin.ElasticityUpdatedSolver_x1_1_get
    if _newclass:x1_1 = property(_dolfin.ElasticityUpdatedSolver_x1_1_get, _dolfin.ElasticityUpdatedSolver_x1_1_set)
    __swig_setmethods__["x2_0"] = _dolfin.ElasticityUpdatedSolver_x2_0_set
    __swig_getmethods__["x2_0"] = _dolfin.ElasticityUpdatedSolver_x2_0_get
    if _newclass:x2_0 = property(_dolfin.ElasticityUpdatedSolver_x2_0_get, _dolfin.ElasticityUpdatedSolver_x2_0_set)
    __swig_setmethods__["x2_1"] = _dolfin.ElasticityUpdatedSolver_x2_1_set
    __swig_getmethods__["x2_1"] = _dolfin.ElasticityUpdatedSolver_x2_1_get
    if _newclass:x2_1 = property(_dolfin.ElasticityUpdatedSolver_x2_1_get, _dolfin.ElasticityUpdatedSolver_x2_1_set)
    __swig_setmethods__["b"] = _dolfin.ElasticityUpdatedSolver_b_set
    __swig_getmethods__["b"] = _dolfin.ElasticityUpdatedSolver_b_get
    if _newclass:b = property(_dolfin.ElasticityUpdatedSolver_b_get, _dolfin.ElasticityUpdatedSolver_b_set)
    __swig_setmethods__["m"] = _dolfin.ElasticityUpdatedSolver_m_set
    __swig_getmethods__["m"] = _dolfin.ElasticityUpdatedSolver_m_get
    if _newclass:m = property(_dolfin.ElasticityUpdatedSolver_m_get, _dolfin.ElasticityUpdatedSolver_m_set)
    __swig_setmethods__["msigma"] = _dolfin.ElasticityUpdatedSolver_msigma_set
    __swig_getmethods__["msigma"] = _dolfin.ElasticityUpdatedSolver_msigma_get
    if _newclass:msigma = property(_dolfin.ElasticityUpdatedSolver_msigma_get, _dolfin.ElasticityUpdatedSolver_msigma_set)
    __swig_setmethods__["stepresidual"] = _dolfin.ElasticityUpdatedSolver_stepresidual_set
    __swig_getmethods__["stepresidual"] = _dolfin.ElasticityUpdatedSolver_stepresidual_get
    if _newclass:stepresidual = property(_dolfin.ElasticityUpdatedSolver_stepresidual_get, _dolfin.ElasticityUpdatedSolver_stepresidual_set)
    __swig_setmethods__["xsigma0"] = _dolfin.ElasticityUpdatedSolver_xsigma0_set
    __swig_getmethods__["xsigma0"] = _dolfin.ElasticityUpdatedSolver_xsigma0_get
    if _newclass:xsigma0 = property(_dolfin.ElasticityUpdatedSolver_xsigma0_get, _dolfin.ElasticityUpdatedSolver_xsigma0_set)
    __swig_setmethods__["xsigma1"] = _dolfin.ElasticityUpdatedSolver_xsigma1_set
    __swig_getmethods__["xsigma1"] = _dolfin.ElasticityUpdatedSolver_xsigma1_get
    if _newclass:xsigma1 = property(_dolfin.ElasticityUpdatedSolver_xsigma1_get, _dolfin.ElasticityUpdatedSolver_xsigma1_set)
    __swig_setmethods__["xepsilon1"] = _dolfin.ElasticityUpdatedSolver_xepsilon1_set
    __swig_getmethods__["xepsilon1"] = _dolfin.ElasticityUpdatedSolver_xepsilon1_get
    if _newclass:xepsilon1 = property(_dolfin.ElasticityUpdatedSolver_xepsilon1_get, _dolfin.ElasticityUpdatedSolver_xepsilon1_set)
    __swig_setmethods__["xsigmanorm"] = _dolfin.ElasticityUpdatedSolver_xsigmanorm_set
    __swig_getmethods__["xsigmanorm"] = _dolfin.ElasticityUpdatedSolver_xsigmanorm_get
    if _newclass:xsigmanorm = property(_dolfin.ElasticityUpdatedSolver_xsigmanorm_get, _dolfin.ElasticityUpdatedSolver_xsigmanorm_set)
    __swig_setmethods__["xjaumann1"] = _dolfin.ElasticityUpdatedSolver_xjaumann1_set
    __swig_getmethods__["xjaumann1"] = _dolfin.ElasticityUpdatedSolver_xjaumann1_get
    if _newclass:xjaumann1 = property(_dolfin.ElasticityUpdatedSolver_xjaumann1_get, _dolfin.ElasticityUpdatedSolver_xjaumann1_set)
    __swig_setmethods__["xtmp1"] = _dolfin.ElasticityUpdatedSolver_xtmp1_set
    __swig_getmethods__["xtmp1"] = _dolfin.ElasticityUpdatedSolver_xtmp1_get
    if _newclass:xtmp1 = property(_dolfin.ElasticityUpdatedSolver_xtmp1_get, _dolfin.ElasticityUpdatedSolver_xtmp1_set)
    __swig_setmethods__["xtmp2"] = _dolfin.ElasticityUpdatedSolver_xtmp2_set
    __swig_getmethods__["xtmp2"] = _dolfin.ElasticityUpdatedSolver_xtmp2_get
    if _newclass:xtmp2 = property(_dolfin.ElasticityUpdatedSolver_xtmp2_get, _dolfin.ElasticityUpdatedSolver_xtmp2_set)
    __swig_setmethods__["xsigmatmp1"] = _dolfin.ElasticityUpdatedSolver_xsigmatmp1_set
    __swig_getmethods__["xsigmatmp1"] = _dolfin.ElasticityUpdatedSolver_xsigmatmp1_get
    if _newclass:xsigmatmp1 = property(_dolfin.ElasticityUpdatedSolver_xsigmatmp1_get, _dolfin.ElasticityUpdatedSolver_xsigmatmp1_set)
    __swig_setmethods__["xsigmatmp2"] = _dolfin.ElasticityUpdatedSolver_xsigmatmp2_set
    __swig_getmethods__["xsigmatmp2"] = _dolfin.ElasticityUpdatedSolver_xsigmatmp2_get
    if _newclass:xsigmatmp2 = property(_dolfin.ElasticityUpdatedSolver_xsigmatmp2_get, _dolfin.ElasticityUpdatedSolver_xsigmatmp2_set)
    __swig_setmethods__["fcontact"] = _dolfin.ElasticityUpdatedSolver_fcontact_set
    __swig_getmethods__["fcontact"] = _dolfin.ElasticityUpdatedSolver_fcontact_get
    if _newclass:fcontact = property(_dolfin.ElasticityUpdatedSolver_fcontact_get, _dolfin.ElasticityUpdatedSolver_fcontact_set)
    __swig_setmethods__["Dummy"] = _dolfin.ElasticityUpdatedSolver_Dummy_set
    __swig_getmethods__["Dummy"] = _dolfin.ElasticityUpdatedSolver_Dummy_get
    if _newclass:Dummy = property(_dolfin.ElasticityUpdatedSolver_Dummy_get, _dolfin.ElasticityUpdatedSolver_Dummy_set)
    __swig_setmethods__["dotu_x1"] = _dolfin.ElasticityUpdatedSolver_dotu_x1_set
    __swig_getmethods__["dotu_x1"] = _dolfin.ElasticityUpdatedSolver_dotu_x1_get
    if _newclass:dotu_x1 = property(_dolfin.ElasticityUpdatedSolver_dotu_x1_get, _dolfin.ElasticityUpdatedSolver_dotu_x1_set)
    __swig_setmethods__["dotu_x2"] = _dolfin.ElasticityUpdatedSolver_dotu_x2_set
    __swig_getmethods__["dotu_x2"] = _dolfin.ElasticityUpdatedSolver_dotu_x2_get
    if _newclass:dotu_x2 = property(_dolfin.ElasticityUpdatedSolver_dotu_x2_get, _dolfin.ElasticityUpdatedSolver_dotu_x2_set)
    __swig_setmethods__["dotu_xsigma"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigma_set
    __swig_getmethods__["dotu_xsigma"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigma_get
    if _newclass:dotu_xsigma = property(_dolfin.ElasticityUpdatedSolver_dotu_xsigma_get, _dolfin.ElasticityUpdatedSolver_dotu_xsigma_set)
    __swig_setmethods__["dotu"] = _dolfin.ElasticityUpdatedSolver_dotu_set
    __swig_getmethods__["dotu"] = _dolfin.ElasticityUpdatedSolver_dotu_get
    if _newclass:dotu = property(_dolfin.ElasticityUpdatedSolver_dotu_get, _dolfin.ElasticityUpdatedSolver_dotu_set)
    __swig_setmethods__["dotu_x1sc"] = _dolfin.ElasticityUpdatedSolver_dotu_x1sc_set
    __swig_getmethods__["dotu_x1sc"] = _dolfin.ElasticityUpdatedSolver_dotu_x1sc_get
    if _newclass:dotu_x1sc = property(_dolfin.ElasticityUpdatedSolver_dotu_x1sc_get, _dolfin.ElasticityUpdatedSolver_dotu_x1sc_set)
    __swig_setmethods__["dotu_x2sc"] = _dolfin.ElasticityUpdatedSolver_dotu_x2sc_set
    __swig_getmethods__["dotu_x2sc"] = _dolfin.ElasticityUpdatedSolver_dotu_x2sc_get
    if _newclass:dotu_x2sc = property(_dolfin.ElasticityUpdatedSolver_dotu_x2sc_get, _dolfin.ElasticityUpdatedSolver_dotu_x2sc_set)
    __swig_setmethods__["dotu_xsigmasc"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigmasc_set
    __swig_getmethods__["dotu_xsigmasc"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigmasc_get
    if _newclass:dotu_xsigmasc = property(_dolfin.ElasticityUpdatedSolver_dotu_xsigmasc_get, _dolfin.ElasticityUpdatedSolver_dotu_xsigmasc_set)
    __swig_setmethods__["dotu_x1is"] = _dolfin.ElasticityUpdatedSolver_dotu_x1is_set
    __swig_getmethods__["dotu_x1is"] = _dolfin.ElasticityUpdatedSolver_dotu_x1is_get
    if _newclass:dotu_x1is = property(_dolfin.ElasticityUpdatedSolver_dotu_x1is_get, _dolfin.ElasticityUpdatedSolver_dotu_x1is_set)
    __swig_setmethods__["dotu_x2is"] = _dolfin.ElasticityUpdatedSolver_dotu_x2is_set
    __swig_getmethods__["dotu_x2is"] = _dolfin.ElasticityUpdatedSolver_dotu_x2is_get
    if _newclass:dotu_x2is = property(_dolfin.ElasticityUpdatedSolver_dotu_x2is_get, _dolfin.ElasticityUpdatedSolver_dotu_x2is_set)
    __swig_setmethods__["dotu_xsigmais"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigmais_set
    __swig_getmethods__["dotu_xsigmais"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigmais_get
    if _newclass:dotu_xsigmais = property(_dolfin.ElasticityUpdatedSolver_dotu_xsigmais_get, _dolfin.ElasticityUpdatedSolver_dotu_xsigmais_set)
    __swig_setmethods__["dotu_x1_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_x1_indices_set
    __swig_getmethods__["dotu_x1_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_x1_indices_get
    if _newclass:dotu_x1_indices = property(_dolfin.ElasticityUpdatedSolver_dotu_x1_indices_get, _dolfin.ElasticityUpdatedSolver_dotu_x1_indices_set)
    __swig_setmethods__["dotu_x2_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_x2_indices_set
    __swig_getmethods__["dotu_x2_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_x2_indices_get
    if _newclass:dotu_x2_indices = property(_dolfin.ElasticityUpdatedSolver_dotu_x2_indices_get, _dolfin.ElasticityUpdatedSolver_dotu_x2_indices_set)
    __swig_setmethods__["dotu_xsigma_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigma_indices_set
    __swig_getmethods__["dotu_xsigma_indices"] = _dolfin.ElasticityUpdatedSolver_dotu_xsigma_indices_get
    if _newclass:dotu_xsigma_indices = property(_dolfin.ElasticityUpdatedSolver_dotu_xsigma_indices_get, _dolfin.ElasticityUpdatedSolver_dotu_xsigma_indices_set)
    __swig_setmethods__["v1"] = _dolfin.ElasticityUpdatedSolver_v1_set
    __swig_getmethods__["v1"] = _dolfin.ElasticityUpdatedSolver_v1_get
    if _newclass:v1 = property(_dolfin.ElasticityUpdatedSolver_v1_get, _dolfin.ElasticityUpdatedSolver_v1_set)
    __swig_setmethods__["u0"] = _dolfin.ElasticityUpdatedSolver_u0_set
    __swig_getmethods__["u0"] = _dolfin.ElasticityUpdatedSolver_u0_get
    if _newclass:u0 = property(_dolfin.ElasticityUpdatedSolver_u0_get, _dolfin.ElasticityUpdatedSolver_u0_set)
    __swig_setmethods__["u1"] = _dolfin.ElasticityUpdatedSolver_u1_set
    __swig_getmethods__["u1"] = _dolfin.ElasticityUpdatedSolver_u1_get
    if _newclass:u1 = property(_dolfin.ElasticityUpdatedSolver_u1_get, _dolfin.ElasticityUpdatedSolver_u1_set)
    __swig_setmethods__["sigma0"] = _dolfin.ElasticityUpdatedSolver_sigma0_set
    __swig_getmethods__["sigma0"] = _dolfin.ElasticityUpdatedSolver_sigma0_get
    if _newclass:sigma0 = property(_dolfin.ElasticityUpdatedSolver_sigma0_get, _dolfin.ElasticityUpdatedSolver_sigma0_set)
    __swig_setmethods__["sigma1"] = _dolfin.ElasticityUpdatedSolver_sigma1_set
    __swig_getmethods__["sigma1"] = _dolfin.ElasticityUpdatedSolver_sigma1_get
    if _newclass:sigma1 = property(_dolfin.ElasticityUpdatedSolver_sigma1_get, _dolfin.ElasticityUpdatedSolver_sigma1_set)
    __swig_setmethods__["epsilon1"] = _dolfin.ElasticityUpdatedSolver_epsilon1_set
    __swig_getmethods__["epsilon1"] = _dolfin.ElasticityUpdatedSolver_epsilon1_get
    if _newclass:epsilon1 = property(_dolfin.ElasticityUpdatedSolver_epsilon1_get, _dolfin.ElasticityUpdatedSolver_epsilon1_set)
    __swig_setmethods__["sigmanorm"] = _dolfin.ElasticityUpdatedSolver_sigmanorm_set
    __swig_getmethods__["sigmanorm"] = _dolfin.ElasticityUpdatedSolver_sigmanorm_get
    if _newclass:sigmanorm = property(_dolfin.ElasticityUpdatedSolver_sigmanorm_get, _dolfin.ElasticityUpdatedSolver_sigmanorm_set)
    __swig_setmethods__["Lv"] = _dolfin.ElasticityUpdatedSolver_Lv_set
    __swig_getmethods__["Lv"] = _dolfin.ElasticityUpdatedSolver_Lv_get
    if _newclass:Lv = property(_dolfin.ElasticityUpdatedSolver_Lv_get, _dolfin.ElasticityUpdatedSolver_Lv_set)
    __swig_setmethods__["Lsigma"] = _dolfin.ElasticityUpdatedSolver_Lsigma_set
    __swig_getmethods__["Lsigma"] = _dolfin.ElasticityUpdatedSolver_Lsigma_get
    if _newclass:Lsigma = property(_dolfin.ElasticityUpdatedSolver_Lsigma_get, _dolfin.ElasticityUpdatedSolver_Lsigma_set)
_dolfin.ElasticityUpdatedSolver_swigregister(ElasticityUpdatedSolver)

set = _dolfin.set

assemble = _dolfin.assemble

applyBC = _dolfin.applyBC

ElasticityUpdatedSolver_solve = _dolfin.ElasticityUpdatedSolver_solve

ElasticityUpdatedSolver_finterpolate = _dolfin.ElasticityUpdatedSolver_finterpolate

ElasticityUpdatedSolver_plasticity = _dolfin.ElasticityUpdatedSolver_plasticity

ElasticityUpdatedSolver_initmsigma = _dolfin.ElasticityUpdatedSolver_initmsigma

ElasticityUpdatedSolver_initu0 = _dolfin.ElasticityUpdatedSolver_initu0

ElasticityUpdatedSolver_initJ0 = _dolfin.ElasticityUpdatedSolver_initJ0

ElasticityUpdatedSolver_computeJ = _dolfin.ElasticityUpdatedSolver_computeJ

ElasticityUpdatedSolver_initF0Green = _dolfin.ElasticityUpdatedSolver_initF0Green

ElasticityUpdatedSolver_computeFGreen = _dolfin.ElasticityUpdatedSolver_computeFGreen

ElasticityUpdatedSolver_initF0Euler = _dolfin.ElasticityUpdatedSolver_initF0Euler

ElasticityUpdatedSolver_computeFEuler = _dolfin.ElasticityUpdatedSolver_computeFEuler

ElasticityUpdatedSolver_computeFBEuler = _dolfin.ElasticityUpdatedSolver_computeFBEuler

ElasticityUpdatedSolver_computeBEuler = _dolfin.ElasticityUpdatedSolver_computeBEuler

ElasticityUpdatedSolver_multF = _dolfin.ElasticityUpdatedSolver_multF

ElasticityUpdatedSolver_multB = _dolfin.ElasticityUpdatedSolver_multB

ElasticityUpdatedSolver_deform = _dolfin.ElasticityUpdatedSolver_deform

class ElasticityUpdatedODE(ODE):
    __swig_setmethods__ = {}
    for _s in [ODE]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, ElasticityUpdatedODE, name, value)
    __swig_getmethods__ = {}
    for _s in [ODE]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, ElasticityUpdatedODE, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::ElasticityUpdatedODE instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_ElasticityUpdatedODE(*args)
        try: self.this.append(this)
        except: self.this = this
    def u0(*args): return _dolfin.ElasticityUpdatedODE_u0(*args)
    def fmono(*args): return _dolfin.ElasticityUpdatedODE_fmono(*args)
    def update(*args): return _dolfin.ElasticityUpdatedODE_update(*args)
    def fromArray(*args): return _dolfin.ElasticityUpdatedODE_fromArray(*args)
    def toArray(*args): return _dolfin.ElasticityUpdatedODE_toArray(*args)
    __swig_setmethods__["solver"] = _dolfin.ElasticityUpdatedODE_solver_set
    __swig_getmethods__["solver"] = _dolfin.ElasticityUpdatedODE_solver_get
    if _newclass:solver = property(_dolfin.ElasticityUpdatedODE_solver_get, _dolfin.ElasticityUpdatedODE_solver_set)
_dolfin.ElasticityUpdatedODE_swigregister(ElasticityUpdatedODE)

class UtilBC1(BoundaryCondition):
    __swig_setmethods__ = {}
    for _s in [BoundaryCondition]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UtilBC1, name, value)
    __swig_getmethods__ = {}
    for _s in [BoundaryCondition]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UtilBC1, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::UtilBC1 instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_UtilBC1(*args)
        try: self.this.append(this)
        except: self.this = this
    def eval(*args): return _dolfin.UtilBC1_eval(*args)
_dolfin.UtilBC1_swigregister(UtilBC1)

class UtilBC2(BoundaryCondition):
    __swig_setmethods__ = {}
    for _s in [BoundaryCondition]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, UtilBC2, name, value)
    __swig_getmethods__ = {}
    for _s in [BoundaryCondition]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, UtilBC2, name)
    def __repr__(self):
        try: strthis = "at 0x%x" %( self.this, ) 
        except: strthis = "" 
        return "<%s.%s; proxy of C++ dolfin::UtilBC2 instance %s>" % (self.__class__.__module__, self.__class__.__name__, strthis,)
    def __init__(self, *args):
        this = _dolfin.new_UtilBC2(*args)
        try: self.this.append(this)
        except: self.this = this
    def eval(*args): return _dolfin.UtilBC2_eval(*args)
_dolfin.UtilBC2_swigregister(UtilBC2)



