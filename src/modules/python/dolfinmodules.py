# This file was created automatically by SWIG 1.3.29.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _dolfinmodules
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


class PySwigIterator(_object):
    """Proxy of C++ PySwigIterator class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PySwigIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PySwigIterator, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _dolfinmodules.delete_PySwigIterator
    __del__ = lambda self : None;
    def value(*args):
        """value(self) -> PyObject"""
        return _dolfinmodules.PySwigIterator_value(*args)

    def incr(*args):
        """
        incr(self, size_t n=1) -> PySwigIterator
        incr(self) -> PySwigIterator
        """
        return _dolfinmodules.PySwigIterator_incr(*args)

    def decr(*args):
        """
        decr(self, size_t n=1) -> PySwigIterator
        decr(self) -> PySwigIterator
        """
        return _dolfinmodules.PySwigIterator_decr(*args)

    def distance(*args):
        """distance(self, PySwigIterator x) -> ptrdiff_t"""
        return _dolfinmodules.PySwigIterator_distance(*args)

    def equal(*args):
        """equal(self, PySwigIterator x) -> bool"""
        return _dolfinmodules.PySwigIterator_equal(*args)

    def copy(*args):
        """copy(self) -> PySwigIterator"""
        return _dolfinmodules.PySwigIterator_copy(*args)

    def next(*args):
        """next(self) -> PyObject"""
        return _dolfinmodules.PySwigIterator_next(*args)

    def previous(*args):
        """previous(self) -> PyObject"""
        return _dolfinmodules.PySwigIterator_previous(*args)

    def advance(*args):
        """advance(self, ptrdiff_t n) -> PySwigIterator"""
        return _dolfinmodules.PySwigIterator_advance(*args)

    def __eq__(*args):
        """__eq__(self, PySwigIterator x) -> bool"""
        return _dolfinmodules.PySwigIterator___eq__(*args)

    def __ne__(*args):
        """__ne__(self, PySwigIterator x) -> bool"""
        return _dolfinmodules.PySwigIterator___ne__(*args)

    def __iadd__(*args):
        """__iadd__(self, ptrdiff_t n) -> PySwigIterator"""
        return _dolfinmodules.PySwigIterator___iadd__(*args)

    def __isub__(*args):
        """__isub__(self, ptrdiff_t n) -> PySwigIterator"""
        return _dolfinmodules.PySwigIterator___isub__(*args)

    def __add__(*args):
        """__add__(self, ptrdiff_t n) -> PySwigIterator"""
        return _dolfinmodules.PySwigIterator___add__(*args)

    def __sub__(*args):
        """
        __sub__(self, ptrdiff_t n) -> PySwigIterator
        __sub__(self, PySwigIterator x) -> ptrdiff_t
        """
        return _dolfinmodules.PySwigIterator___sub__(*args)

    def __iter__(self): return self
PySwigIterator_swigregister = _dolfinmodules.PySwigIterator_swigregister
PySwigIterator_swigregister(PySwigIterator)


def new_realArray(*args):
  """new_realArray(size_t nelements) -> real"""
  return _dolfinmodules.new_realArray(*args)

def delete_realArray(*args):
  """delete_realArray(real ary)"""
  return _dolfinmodules.delete_realArray(*args)

def realArray_getitem(*args):
  """realArray_getitem(real ary, size_t index) -> real"""
  return _dolfinmodules.realArray_getitem(*args)

def realArray_setitem(*args):
  """realArray_setitem(real ary, size_t index, real value)"""
  return _dolfinmodules.realArray_setitem(*args)

def new_intArray(*args):
  """new_intArray(size_t nelements) -> int"""
  return _dolfinmodules.new_intArray(*args)

def delete_intArray(*args):
  """delete_intArray(int ary)"""
  return _dolfinmodules.delete_intArray(*args)

def intArray_getitem(*args):
  """intArray_getitem(int ary, size_t index) -> int"""
  return _dolfinmodules.intArray_getitem(*args)

def intArray_setitem(*args):
  """intArray_setitem(int ary, size_t index, int value)"""
  return _dolfinmodules.intArray_setitem(*args)
class intp(_object):
    """Proxy of C++ intp class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, intp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, intp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> intp"""
        this = _dolfinmodules.new_intp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfinmodules.delete_intp
    __del__ = lambda self : None;
    def assign(*args):
        """assign(self, int value)"""
        return _dolfinmodules.intp_assign(*args)

    def value(*args):
        """value(self) -> int"""
        return _dolfinmodules.intp_value(*args)

    def cast(*args):
        """cast(self) -> int"""
        return _dolfinmodules.intp_cast(*args)

    def frompointer(*args):
        """frompointer(int t) -> intp"""
        return _dolfinmodules.intp_frompointer(*args)

    if _newclass:frompointer = staticmethod(frompointer)
    __swig_getmethods__["frompointer"] = lambda x: frompointer
intp_swigregister = _dolfinmodules.intp_swigregister
intp_swigregister(intp)

def intp_frompointer(*args):
  """intp_frompointer(int t) -> intp"""
  return _dolfinmodules.intp_frompointer(*args)

class doublep(_object):
    """Proxy of C++ doublep class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, doublep, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, doublep, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> doublep"""
        this = _dolfinmodules.new_doublep(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _dolfinmodules.delete_doublep
    __del__ = lambda self : None;
    def assign(*args):
        """assign(self, double value)"""
        return _dolfinmodules.doublep_assign(*args)

    def value(*args):
        """value(self) -> double"""
        return _dolfinmodules.doublep_value(*args)

    def cast(*args):
        """cast(self) -> double"""
        return _dolfinmodules.doublep_cast(*args)

    def frompointer(*args):
        """frompointer(double t) -> doublep"""
        return _dolfinmodules.doublep_frompointer(*args)

    if _newclass:frompointer = staticmethod(frompointer)
    __swig_getmethods__["frompointer"] = lambda x: frompointer
doublep_swigregister = _dolfinmodules.doublep_swigregister
doublep_swigregister(doublep)

def doublep_frompointer(*args):
  """doublep_frompointer(double t) -> doublep"""
  return _dolfinmodules.doublep_frompointer(*args)

class ublas_dense_matrix(_object):
    """Proxy of C++ ublas_dense_matrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ublas_dense_matrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ublas_dense_matrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
ublas_dense_matrix_swigregister = _dolfinmodules.ublas_dense_matrix_swigregister
ublas_dense_matrix_swigregister(ublas_dense_matrix)

class ublas_sparse_matrix(_object):
    """Proxy of C++ ublas_sparse_matrix class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ublas_sparse_matrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ublas_sparse_matrix, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
ublas_sparse_matrix_swigregister = _dolfinmodules.ublas_sparse_matrix_swigregister
ublas_sparse_matrix_swigregister(ublas_sparse_matrix)

class ElasticityUpdatedSolver(_object):
    """Proxy of C++ ElasticityUpdatedSolver class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ElasticityUpdatedSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ElasticityUpdatedSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, Mesh mesh, Function f, Function v0, Function rho, real E, 
            real nu, real nuv, real nuplast, BoundaryCondition bc, 
            real k, real T) -> ElasticityUpdatedSolver
        """
        this = _dolfinmodules.new_ElasticityUpdatedSolver(*args)
        try: self.this.append(this)
        except: self.this = this
    def init(*args):
        """init(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_init(*args)

    def step(*args):
        """step(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_step(*args)

    def oldstep(*args):
        """oldstep(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_oldstep(*args)

    def fu(*args):
        """fu(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_fu(*args)

    def gather(*args):
        """gather(Vector x1, Vector x2, VecScatter x1sc)"""
        return _dolfinmodules.ElasticityUpdatedSolver_gather(*args)

    if _newclass:gather = staticmethod(gather)
    __swig_getmethods__["gather"] = lambda x: gather
    def scatter(*args):
        """scatter(Vector x1, Vector x2, VecScatter x1sc)"""
        return _dolfinmodules.ElasticityUpdatedSolver_scatter(*args)

    if _newclass:scatter = staticmethod(scatter)
    __swig_getmethods__["scatter"] = lambda x: scatter
    def createScatterer(*args):
        """createScatterer(Vector x1, Vector x2, int offset, int size) -> VecScatter"""
        return _dolfinmodules.ElasticityUpdatedSolver_createScatterer(*args)

    if _newclass:createScatterer = staticmethod(createScatterer)
    __swig_getmethods__["createScatterer"] = lambda x: createScatterer
    def fromArray(*args):
        """fromArray(real u, Vector x, uint offset, uint size)"""
        return _dolfinmodules.ElasticityUpdatedSolver_fromArray(*args)

    if _newclass:fromArray = staticmethod(fromArray)
    __swig_getmethods__["fromArray"] = lambda x: fromArray
    def toArray(*args):
        """toArray(real y, Vector x, uint offset, uint size)"""
        return _dolfinmodules.ElasticityUpdatedSolver_toArray(*args)

    if _newclass:toArray = staticmethod(toArray)
    __swig_getmethods__["toArray"] = lambda x: toArray
    def fromDense(*args):
        """fromDense(uBlasVector u, Vector x, uint offset, uint size)"""
        return _dolfinmodules.ElasticityUpdatedSolver_fromDense(*args)

    if _newclass:fromDense = staticmethod(fromDense)
    __swig_getmethods__["fromDense"] = lambda x: fromDense
    def toDense(*args):
        """toDense(uBlasVector y, Vector x, uint offset, uint size)"""
        return _dolfinmodules.ElasticityUpdatedSolver_toDense(*args)

    if _newclass:toDense = staticmethod(toDense)
    __swig_getmethods__["toDense"] = lambda x: toDense
    def preparestep(*args):
        """preparestep(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_preparestep(*args)

    def prepareiteration(*args):
        """prepareiteration(self)"""
        return _dolfinmodules.ElasticityUpdatedSolver_prepareiteration(*args)

    def save(*args):
        """save(self, Mesh mesh, File solutionfile, real t)"""
        return _dolfinmodules.ElasticityUpdatedSolver_save(*args)

    def condsave(*args):
        """condsave(self, Mesh mesh, File solutionfile, real t)"""
        return _dolfinmodules.ElasticityUpdatedSolver_condsave(*args)

    def solve(*args):
        """
        solve()
        solve(Mesh mesh, Function f, Function v0, Function rho, real E, 
            real nu, real nuv, real nuplast, BoundaryCondition bc, 
            real k, real T)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_solve(*args)

    if _newclass:solve = staticmethod(solve)
    __swig_getmethods__["solve"] = lambda x: solve
    __swig_setmethods__["dotu_x1sc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1sc_set
    __swig_getmethods__["dotu_x1sc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1sc_get
    if _newclass:dotu_x1sc = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x1sc_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x1sc_set)
    __swig_setmethods__["dotu_x2sc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2sc_set
    __swig_getmethods__["dotu_x2sc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2sc_get
    if _newclass:dotu_x2sc = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x2sc_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x2sc_set)
    __swig_setmethods__["dotu_xsigmasc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmasc_set
    __swig_getmethods__["dotu_xsigmasc"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmasc_get
    if _newclass:dotu_xsigmasc = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmasc_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmasc_set)
    __swig_setmethods__["dotu_x1is"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1is_set
    __swig_getmethods__["dotu_x1is"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1is_get
    if _newclass:dotu_x1is = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x1is_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x1is_set)
    __swig_setmethods__["dotu_x2is"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2is_set
    __swig_getmethods__["dotu_x2is"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2is_get
    if _newclass:dotu_x2is = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x2is_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x2is_set)
    __swig_setmethods__["dotu_xsigmais"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmais_set
    __swig_getmethods__["dotu_xsigmais"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmais_get
    if _newclass:dotu_xsigmais = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmais_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigmais_set)
    __swig_setmethods__["mesh"] = _dolfinmodules.ElasticityUpdatedSolver_mesh_set
    __swig_getmethods__["mesh"] = _dolfinmodules.ElasticityUpdatedSolver_mesh_get
    if _newclass:mesh = property(_dolfinmodules.ElasticityUpdatedSolver_mesh_get, _dolfinmodules.ElasticityUpdatedSolver_mesh_set)
    __swig_setmethods__["f"] = _dolfinmodules.ElasticityUpdatedSolver_f_set
    __swig_getmethods__["f"] = _dolfinmodules.ElasticityUpdatedSolver_f_get
    if _newclass:f = property(_dolfinmodules.ElasticityUpdatedSolver_f_get, _dolfinmodules.ElasticityUpdatedSolver_f_set)
    __swig_setmethods__["v0"] = _dolfinmodules.ElasticityUpdatedSolver_v0_set
    __swig_getmethods__["v0"] = _dolfinmodules.ElasticityUpdatedSolver_v0_get
    if _newclass:v0 = property(_dolfinmodules.ElasticityUpdatedSolver_v0_get, _dolfinmodules.ElasticityUpdatedSolver_v0_set)
    __swig_setmethods__["rho"] = _dolfinmodules.ElasticityUpdatedSolver_rho_set
    __swig_getmethods__["rho"] = _dolfinmodules.ElasticityUpdatedSolver_rho_get
    if _newclass:rho = property(_dolfinmodules.ElasticityUpdatedSolver_rho_get, _dolfinmodules.ElasticityUpdatedSolver_rho_set)
    __swig_setmethods__["E"] = _dolfinmodules.ElasticityUpdatedSolver_E_set
    __swig_getmethods__["E"] = _dolfinmodules.ElasticityUpdatedSolver_E_get
    if _newclass:E = property(_dolfinmodules.ElasticityUpdatedSolver_E_get, _dolfinmodules.ElasticityUpdatedSolver_E_set)
    __swig_setmethods__["nu"] = _dolfinmodules.ElasticityUpdatedSolver_nu_set
    __swig_getmethods__["nu"] = _dolfinmodules.ElasticityUpdatedSolver_nu_get
    if _newclass:nu = property(_dolfinmodules.ElasticityUpdatedSolver_nu_get, _dolfinmodules.ElasticityUpdatedSolver_nu_set)
    __swig_setmethods__["nuv"] = _dolfinmodules.ElasticityUpdatedSolver_nuv_set
    __swig_getmethods__["nuv"] = _dolfinmodules.ElasticityUpdatedSolver_nuv_get
    if _newclass:nuv = property(_dolfinmodules.ElasticityUpdatedSolver_nuv_get, _dolfinmodules.ElasticityUpdatedSolver_nuv_set)
    __swig_setmethods__["nuplast"] = _dolfinmodules.ElasticityUpdatedSolver_nuplast_set
    __swig_getmethods__["nuplast"] = _dolfinmodules.ElasticityUpdatedSolver_nuplast_get
    if _newclass:nuplast = property(_dolfinmodules.ElasticityUpdatedSolver_nuplast_get, _dolfinmodules.ElasticityUpdatedSolver_nuplast_set)
    __swig_setmethods__["bc"] = _dolfinmodules.ElasticityUpdatedSolver_bc_set
    __swig_getmethods__["bc"] = _dolfinmodules.ElasticityUpdatedSolver_bc_get
    if _newclass:bc = property(_dolfinmodules.ElasticityUpdatedSolver_bc_get, _dolfinmodules.ElasticityUpdatedSolver_bc_set)
    __swig_setmethods__["k"] = _dolfinmodules.ElasticityUpdatedSolver_k_set
    __swig_getmethods__["k"] = _dolfinmodules.ElasticityUpdatedSolver_k_get
    if _newclass:k = property(_dolfinmodules.ElasticityUpdatedSolver_k_get, _dolfinmodules.ElasticityUpdatedSolver_k_set)
    __swig_setmethods__["T"] = _dolfinmodules.ElasticityUpdatedSolver_T_set
    __swig_getmethods__["T"] = _dolfinmodules.ElasticityUpdatedSolver_T_get
    if _newclass:T = property(_dolfinmodules.ElasticityUpdatedSolver_T_get, _dolfinmodules.ElasticityUpdatedSolver_T_set)
    __swig_setmethods__["counter"] = _dolfinmodules.ElasticityUpdatedSolver_counter_set
    __swig_getmethods__["counter"] = _dolfinmodules.ElasticityUpdatedSolver_counter_get
    if _newclass:counter = property(_dolfinmodules.ElasticityUpdatedSolver_counter_get, _dolfinmodules.ElasticityUpdatedSolver_counter_set)
    __swig_setmethods__["lastsample"] = _dolfinmodules.ElasticityUpdatedSolver_lastsample_set
    __swig_getmethods__["lastsample"] = _dolfinmodules.ElasticityUpdatedSolver_lastsample_get
    if _newclass:lastsample = property(_dolfinmodules.ElasticityUpdatedSolver_lastsample_get, _dolfinmodules.ElasticityUpdatedSolver_lastsample_set)
    __swig_setmethods__["lmbda"] = _dolfinmodules.ElasticityUpdatedSolver_lmbda_set
    __swig_getmethods__["lmbda"] = _dolfinmodules.ElasticityUpdatedSolver_lmbda_get
    if _newclass:lmbda = property(_dolfinmodules.ElasticityUpdatedSolver_lmbda_get, _dolfinmodules.ElasticityUpdatedSolver_lmbda_set)
    __swig_setmethods__["mu"] = _dolfinmodules.ElasticityUpdatedSolver_mu_set
    __swig_getmethods__["mu"] = _dolfinmodules.ElasticityUpdatedSolver_mu_get
    if _newclass:mu = property(_dolfinmodules.ElasticityUpdatedSolver_mu_get, _dolfinmodules.ElasticityUpdatedSolver_mu_set)
    __swig_setmethods__["t"] = _dolfinmodules.ElasticityUpdatedSolver_t_set
    __swig_getmethods__["t"] = _dolfinmodules.ElasticityUpdatedSolver_t_get
    if _newclass:t = property(_dolfinmodules.ElasticityUpdatedSolver_t_get, _dolfinmodules.ElasticityUpdatedSolver_t_set)
    __swig_setmethods__["rtol"] = _dolfinmodules.ElasticityUpdatedSolver_rtol_set
    __swig_getmethods__["rtol"] = _dolfinmodules.ElasticityUpdatedSolver_rtol_get
    if _newclass:rtol = property(_dolfinmodules.ElasticityUpdatedSolver_rtol_get, _dolfinmodules.ElasticityUpdatedSolver_rtol_set)
    __swig_setmethods__["maxiters"] = _dolfinmodules.ElasticityUpdatedSolver_maxiters_set
    __swig_getmethods__["maxiters"] = _dolfinmodules.ElasticityUpdatedSolver_maxiters_get
    if _newclass:maxiters = property(_dolfinmodules.ElasticityUpdatedSolver_maxiters_get, _dolfinmodules.ElasticityUpdatedSolver_maxiters_set)
    __swig_setmethods__["do_plasticity"] = _dolfinmodules.ElasticityUpdatedSolver_do_plasticity_set
    __swig_getmethods__["do_plasticity"] = _dolfinmodules.ElasticityUpdatedSolver_do_plasticity_get
    if _newclass:do_plasticity = property(_dolfinmodules.ElasticityUpdatedSolver_do_plasticity_get, _dolfinmodules.ElasticityUpdatedSolver_do_plasticity_set)
    __swig_setmethods__["yld"] = _dolfinmodules.ElasticityUpdatedSolver_yld_set
    __swig_getmethods__["yld"] = _dolfinmodules.ElasticityUpdatedSolver_yld_get
    if _newclass:yld = property(_dolfinmodules.ElasticityUpdatedSolver_yld_get, _dolfinmodules.ElasticityUpdatedSolver_yld_set)
    __swig_setmethods__["savesamplefreq"] = _dolfinmodules.ElasticityUpdatedSolver_savesamplefreq_set
    __swig_getmethods__["savesamplefreq"] = _dolfinmodules.ElasticityUpdatedSolver_savesamplefreq_get
    if _newclass:savesamplefreq = property(_dolfinmodules.ElasticityUpdatedSolver_savesamplefreq_get, _dolfinmodules.ElasticityUpdatedSolver_savesamplefreq_set)
    __swig_setmethods__["fevals"] = _dolfinmodules.ElasticityUpdatedSolver_fevals_set
    __swig_getmethods__["fevals"] = _dolfinmodules.ElasticityUpdatedSolver_fevals_get
    if _newclass:fevals = property(_dolfinmodules.ElasticityUpdatedSolver_fevals_get, _dolfinmodules.ElasticityUpdatedSolver_fevals_set)
    __swig_setmethods__["Nv"] = _dolfinmodules.ElasticityUpdatedSolver_Nv_set
    __swig_getmethods__["Nv"] = _dolfinmodules.ElasticityUpdatedSolver_Nv_get
    if _newclass:Nv = property(_dolfinmodules.ElasticityUpdatedSolver_Nv_get, _dolfinmodules.ElasticityUpdatedSolver_Nv_set)
    __swig_setmethods__["Nsigma"] = _dolfinmodules.ElasticityUpdatedSolver_Nsigma_set
    __swig_getmethods__["Nsigma"] = _dolfinmodules.ElasticityUpdatedSolver_Nsigma_get
    if _newclass:Nsigma = property(_dolfinmodules.ElasticityUpdatedSolver_Nsigma_get, _dolfinmodules.ElasticityUpdatedSolver_Nsigma_set)
    __swig_setmethods__["Nsigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_Nsigmanorm_set
    __swig_getmethods__["Nsigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_Nsigmanorm_get
    if _newclass:Nsigmanorm = property(_dolfinmodules.ElasticityUpdatedSolver_Nsigmanorm_get, _dolfinmodules.ElasticityUpdatedSolver_Nsigmanorm_set)
    __swig_setmethods__["ode"] = _dolfinmodules.ElasticityUpdatedSolver_ode_set
    __swig_getmethods__["ode"] = _dolfinmodules.ElasticityUpdatedSolver_ode_get
    if _newclass:ode = property(_dolfinmodules.ElasticityUpdatedSolver_ode_get, _dolfinmodules.ElasticityUpdatedSolver_ode_set)
    __swig_setmethods__["ts"] = _dolfinmodules.ElasticityUpdatedSolver_ts_set
    __swig_getmethods__["ts"] = _dolfinmodules.ElasticityUpdatedSolver_ts_get
    if _newclass:ts = property(_dolfinmodules.ElasticityUpdatedSolver_ts_get, _dolfinmodules.ElasticityUpdatedSolver_ts_set)
    __swig_setmethods__["element1"] = _dolfinmodules.ElasticityUpdatedSolver_element1_set
    __swig_getmethods__["element1"] = _dolfinmodules.ElasticityUpdatedSolver_element1_get
    if _newclass:element1 = property(_dolfinmodules.ElasticityUpdatedSolver_element1_get, _dolfinmodules.ElasticityUpdatedSolver_element1_set)
    __swig_setmethods__["element2"] = _dolfinmodules.ElasticityUpdatedSolver_element2_set
    __swig_getmethods__["element2"] = _dolfinmodules.ElasticityUpdatedSolver_element2_get
    if _newclass:element2 = property(_dolfinmodules.ElasticityUpdatedSolver_element2_get, _dolfinmodules.ElasticityUpdatedSolver_element2_set)
    __swig_setmethods__["element3"] = _dolfinmodules.ElasticityUpdatedSolver_element3_set
    __swig_getmethods__["element3"] = _dolfinmodules.ElasticityUpdatedSolver_element3_get
    if _newclass:element3 = property(_dolfinmodules.ElasticityUpdatedSolver_element3_get, _dolfinmodules.ElasticityUpdatedSolver_element3_set)
    __swig_setmethods__["x1_0"] = _dolfinmodules.ElasticityUpdatedSolver_x1_0_set
    __swig_getmethods__["x1_0"] = _dolfinmodules.ElasticityUpdatedSolver_x1_0_get
    if _newclass:x1_0 = property(_dolfinmodules.ElasticityUpdatedSolver_x1_0_get, _dolfinmodules.ElasticityUpdatedSolver_x1_0_set)
    __swig_setmethods__["x1_1"] = _dolfinmodules.ElasticityUpdatedSolver_x1_1_set
    __swig_getmethods__["x1_1"] = _dolfinmodules.ElasticityUpdatedSolver_x1_1_get
    if _newclass:x1_1 = property(_dolfinmodules.ElasticityUpdatedSolver_x1_1_get, _dolfinmodules.ElasticityUpdatedSolver_x1_1_set)
    __swig_setmethods__["x2_0"] = _dolfinmodules.ElasticityUpdatedSolver_x2_0_set
    __swig_getmethods__["x2_0"] = _dolfinmodules.ElasticityUpdatedSolver_x2_0_get
    if _newclass:x2_0 = property(_dolfinmodules.ElasticityUpdatedSolver_x2_0_get, _dolfinmodules.ElasticityUpdatedSolver_x2_0_set)
    __swig_setmethods__["x2_1"] = _dolfinmodules.ElasticityUpdatedSolver_x2_1_set
    __swig_getmethods__["x2_1"] = _dolfinmodules.ElasticityUpdatedSolver_x2_1_get
    if _newclass:x2_1 = property(_dolfinmodules.ElasticityUpdatedSolver_x2_1_get, _dolfinmodules.ElasticityUpdatedSolver_x2_1_set)
    __swig_setmethods__["b"] = _dolfinmodules.ElasticityUpdatedSolver_b_set
    __swig_getmethods__["b"] = _dolfinmodules.ElasticityUpdatedSolver_b_get
    if _newclass:b = property(_dolfinmodules.ElasticityUpdatedSolver_b_get, _dolfinmodules.ElasticityUpdatedSolver_b_set)
    __swig_setmethods__["m"] = _dolfinmodules.ElasticityUpdatedSolver_m_set
    __swig_getmethods__["m"] = _dolfinmodules.ElasticityUpdatedSolver_m_get
    if _newclass:m = property(_dolfinmodules.ElasticityUpdatedSolver_m_get, _dolfinmodules.ElasticityUpdatedSolver_m_set)
    __swig_setmethods__["msigma"] = _dolfinmodules.ElasticityUpdatedSolver_msigma_set
    __swig_getmethods__["msigma"] = _dolfinmodules.ElasticityUpdatedSolver_msigma_get
    if _newclass:msigma = property(_dolfinmodules.ElasticityUpdatedSolver_msigma_get, _dolfinmodules.ElasticityUpdatedSolver_msigma_set)
    __swig_setmethods__["stepresidual"] = _dolfinmodules.ElasticityUpdatedSolver_stepresidual_set
    __swig_getmethods__["stepresidual"] = _dolfinmodules.ElasticityUpdatedSolver_stepresidual_get
    if _newclass:stepresidual = property(_dolfinmodules.ElasticityUpdatedSolver_stepresidual_get, _dolfinmodules.ElasticityUpdatedSolver_stepresidual_set)
    __swig_setmethods__["xsigma0"] = _dolfinmodules.ElasticityUpdatedSolver_xsigma0_set
    __swig_getmethods__["xsigma0"] = _dolfinmodules.ElasticityUpdatedSolver_xsigma0_get
    if _newclass:xsigma0 = property(_dolfinmodules.ElasticityUpdatedSolver_xsigma0_get, _dolfinmodules.ElasticityUpdatedSolver_xsigma0_set)
    __swig_setmethods__["xsigma1"] = _dolfinmodules.ElasticityUpdatedSolver_xsigma1_set
    __swig_getmethods__["xsigma1"] = _dolfinmodules.ElasticityUpdatedSolver_xsigma1_get
    if _newclass:xsigma1 = property(_dolfinmodules.ElasticityUpdatedSolver_xsigma1_get, _dolfinmodules.ElasticityUpdatedSolver_xsigma1_set)
    __swig_setmethods__["xepsilon1"] = _dolfinmodules.ElasticityUpdatedSolver_xepsilon1_set
    __swig_getmethods__["xepsilon1"] = _dolfinmodules.ElasticityUpdatedSolver_xepsilon1_get
    if _newclass:xepsilon1 = property(_dolfinmodules.ElasticityUpdatedSolver_xepsilon1_get, _dolfinmodules.ElasticityUpdatedSolver_xepsilon1_set)
    __swig_setmethods__["xsigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmanorm_set
    __swig_getmethods__["xsigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmanorm_get
    if _newclass:xsigmanorm = property(_dolfinmodules.ElasticityUpdatedSolver_xsigmanorm_get, _dolfinmodules.ElasticityUpdatedSolver_xsigmanorm_set)
    __swig_setmethods__["xjaumann1"] = _dolfinmodules.ElasticityUpdatedSolver_xjaumann1_set
    __swig_getmethods__["xjaumann1"] = _dolfinmodules.ElasticityUpdatedSolver_xjaumann1_get
    if _newclass:xjaumann1 = property(_dolfinmodules.ElasticityUpdatedSolver_xjaumann1_get, _dolfinmodules.ElasticityUpdatedSolver_xjaumann1_set)
    __swig_setmethods__["xtmp1"] = _dolfinmodules.ElasticityUpdatedSolver_xtmp1_set
    __swig_getmethods__["xtmp1"] = _dolfinmodules.ElasticityUpdatedSolver_xtmp1_get
    if _newclass:xtmp1 = property(_dolfinmodules.ElasticityUpdatedSolver_xtmp1_get, _dolfinmodules.ElasticityUpdatedSolver_xtmp1_set)
    __swig_setmethods__["xtmp2"] = _dolfinmodules.ElasticityUpdatedSolver_xtmp2_set
    __swig_getmethods__["xtmp2"] = _dolfinmodules.ElasticityUpdatedSolver_xtmp2_get
    if _newclass:xtmp2 = property(_dolfinmodules.ElasticityUpdatedSolver_xtmp2_get, _dolfinmodules.ElasticityUpdatedSolver_xtmp2_set)
    __swig_setmethods__["xsigmatmp1"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp1_set
    __swig_getmethods__["xsigmatmp1"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp1_get
    if _newclass:xsigmatmp1 = property(_dolfinmodules.ElasticityUpdatedSolver_xsigmatmp1_get, _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp1_set)
    __swig_setmethods__["xsigmatmp2"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp2_set
    __swig_getmethods__["xsigmatmp2"] = _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp2_get
    if _newclass:xsigmatmp2 = property(_dolfinmodules.ElasticityUpdatedSolver_xsigmatmp2_get, _dolfinmodules.ElasticityUpdatedSolver_xsigmatmp2_set)
    __swig_setmethods__["fcontact"] = _dolfinmodules.ElasticityUpdatedSolver_fcontact_set
    __swig_getmethods__["fcontact"] = _dolfinmodules.ElasticityUpdatedSolver_fcontact_get
    if _newclass:fcontact = property(_dolfinmodules.ElasticityUpdatedSolver_fcontact_get, _dolfinmodules.ElasticityUpdatedSolver_fcontact_set)
    __swig_setmethods__["Dummy"] = _dolfinmodules.ElasticityUpdatedSolver_Dummy_set
    __swig_getmethods__["Dummy"] = _dolfinmodules.ElasticityUpdatedSolver_Dummy_get
    if _newclass:Dummy = property(_dolfinmodules.ElasticityUpdatedSolver_Dummy_get, _dolfinmodules.ElasticityUpdatedSolver_Dummy_set)
    __swig_setmethods__["dotu_x1"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_set
    __swig_getmethods__["dotu_x1"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_get
    if _newclass:dotu_x1 = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x1_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_set)
    __swig_setmethods__["dotu_x2"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_set
    __swig_getmethods__["dotu_x2"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_get
    if _newclass:dotu_x2 = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x2_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_set)
    __swig_setmethods__["dotu_xsigma"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_set
    __swig_getmethods__["dotu_xsigma"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_get
    if _newclass:dotu_xsigma = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_set)
    __swig_setmethods__["dotu"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_set
    __swig_getmethods__["dotu"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_get
    if _newclass:dotu = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_set)
    __swig_setmethods__["dotu_x1_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_indices_set
    __swig_getmethods__["dotu_x1_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_indices_get
    if _newclass:dotu_x1_indices = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x1_indices_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x1_indices_set)
    __swig_setmethods__["dotu_x2_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_indices_set
    __swig_getmethods__["dotu_x2_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_indices_get
    if _newclass:dotu_x2_indices = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_x2_indices_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_x2_indices_set)
    __swig_setmethods__["dotu_xsigma_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_indices_set
    __swig_getmethods__["dotu_xsigma_indices"] = _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_indices_get
    if _newclass:dotu_xsigma_indices = property(_dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_indices_get, _dolfinmodules.ElasticityUpdatedSolver_dotu_xsigma_indices_set)
    __swig_setmethods__["v1"] = _dolfinmodules.ElasticityUpdatedSolver_v1_set
    __swig_getmethods__["v1"] = _dolfinmodules.ElasticityUpdatedSolver_v1_get
    if _newclass:v1 = property(_dolfinmodules.ElasticityUpdatedSolver_v1_get, _dolfinmodules.ElasticityUpdatedSolver_v1_set)
    __swig_setmethods__["u0"] = _dolfinmodules.ElasticityUpdatedSolver_u0_set
    __swig_getmethods__["u0"] = _dolfinmodules.ElasticityUpdatedSolver_u0_get
    if _newclass:u0 = property(_dolfinmodules.ElasticityUpdatedSolver_u0_get, _dolfinmodules.ElasticityUpdatedSolver_u0_set)
    __swig_setmethods__["u1"] = _dolfinmodules.ElasticityUpdatedSolver_u1_set
    __swig_getmethods__["u1"] = _dolfinmodules.ElasticityUpdatedSolver_u1_get
    if _newclass:u1 = property(_dolfinmodules.ElasticityUpdatedSolver_u1_get, _dolfinmodules.ElasticityUpdatedSolver_u1_set)
    __swig_setmethods__["sigma0"] = _dolfinmodules.ElasticityUpdatedSolver_sigma0_set
    __swig_getmethods__["sigma0"] = _dolfinmodules.ElasticityUpdatedSolver_sigma0_get
    if _newclass:sigma0 = property(_dolfinmodules.ElasticityUpdatedSolver_sigma0_get, _dolfinmodules.ElasticityUpdatedSolver_sigma0_set)
    __swig_setmethods__["sigma1"] = _dolfinmodules.ElasticityUpdatedSolver_sigma1_set
    __swig_getmethods__["sigma1"] = _dolfinmodules.ElasticityUpdatedSolver_sigma1_get
    if _newclass:sigma1 = property(_dolfinmodules.ElasticityUpdatedSolver_sigma1_get, _dolfinmodules.ElasticityUpdatedSolver_sigma1_set)
    __swig_setmethods__["epsilon1"] = _dolfinmodules.ElasticityUpdatedSolver_epsilon1_set
    __swig_getmethods__["epsilon1"] = _dolfinmodules.ElasticityUpdatedSolver_epsilon1_get
    if _newclass:epsilon1 = property(_dolfinmodules.ElasticityUpdatedSolver_epsilon1_get, _dolfinmodules.ElasticityUpdatedSolver_epsilon1_set)
    __swig_setmethods__["sigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_sigmanorm_set
    __swig_getmethods__["sigmanorm"] = _dolfinmodules.ElasticityUpdatedSolver_sigmanorm_get
    if _newclass:sigmanorm = property(_dolfinmodules.ElasticityUpdatedSolver_sigmanorm_get, _dolfinmodules.ElasticityUpdatedSolver_sigmanorm_set)
    __swig_setmethods__["Lv"] = _dolfinmodules.ElasticityUpdatedSolver_Lv_set
    __swig_getmethods__["Lv"] = _dolfinmodules.ElasticityUpdatedSolver_Lv_get
    if _newclass:Lv = property(_dolfinmodules.ElasticityUpdatedSolver_Lv_get, _dolfinmodules.ElasticityUpdatedSolver_Lv_set)
    __swig_setmethods__["Lsigma"] = _dolfinmodules.ElasticityUpdatedSolver_Lsigma_set
    __swig_getmethods__["Lsigma"] = _dolfinmodules.ElasticityUpdatedSolver_Lsigma_get
    if _newclass:Lsigma = property(_dolfinmodules.ElasticityUpdatedSolver_Lsigma_get, _dolfinmodules.ElasticityUpdatedSolver_Lsigma_set)
    def finterpolate(*args):
        """finterpolate(Function f1, Function f2, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_finterpolate(*args)

    if _newclass:finterpolate = staticmethod(finterpolate)
    __swig_getmethods__["finterpolate"] = lambda x: finterpolate
    def plasticity(*args):
        """
        plasticity(Vector xsigma, Vector xsigmanorm, real yld, FiniteElement element2, 
            Mesh mesh)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_plasticity(*args)

    if _newclass:plasticity = staticmethod(plasticity)
    __swig_getmethods__["plasticity"] = lambda x: plasticity
    def initmsigma(*args):
        """initmsigma(Vector msigma, FiniteElement element2, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_initmsigma(*args)

    if _newclass:initmsigma = staticmethod(initmsigma)
    __swig_getmethods__["initmsigma"] = lambda x: initmsigma
    def initu0(*args):
        """initu0(Vector x0, FiniteElement element, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_initu0(*args)

    if _newclass:initu0 = staticmethod(initu0)
    __swig_getmethods__["initu0"] = lambda x: initu0
    def initJ0(*args):
        """initJ0(Vector xJ0, FiniteElement element, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_initJ0(*args)

    if _newclass:initJ0 = staticmethod(initJ0)
    __swig_getmethods__["initJ0"] = lambda x: initJ0
    def computeJ(*args):
        """
        computeJ(Vector xJ0, Vector xJ, Vector xJinv, FiniteElement element, 
            Mesh mesh)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_computeJ(*args)

    if _newclass:computeJ = staticmethod(computeJ)
    __swig_getmethods__["computeJ"] = lambda x: computeJ
    def initF0Green(*args):
        """initF0Green(Vector xF0, FiniteElement element1, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_initF0Green(*args)

    if _newclass:initF0Green = staticmethod(initF0Green)
    __swig_getmethods__["initF0Green"] = lambda x: initF0Green
    def computeFGreen(*args):
        """
        computeFGreen(Vector xF, Vector xF0, Vector xF1, FiniteElement element1, 
            Mesh mesh)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_computeFGreen(*args)

    if _newclass:computeFGreen = staticmethod(computeFGreen)
    __swig_getmethods__["computeFGreen"] = lambda x: computeFGreen
    def initF0Euler(*args):
        """initF0Euler(Vector xF0, FiniteElement element1, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_initF0Euler(*args)

    if _newclass:initF0Euler = staticmethod(initF0Euler)
    __swig_getmethods__["initF0Euler"] = lambda x: initF0Euler
    def computeFEuler(*args):
        """
        computeFEuler(Vector xF, Vector xF0, Vector xF1, FiniteElement element1, 
            Mesh mesh)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_computeFEuler(*args)

    if _newclass:computeFEuler = staticmethod(computeFEuler)
    __swig_getmethods__["computeFEuler"] = lambda x: computeFEuler
    def computeFBEuler(*args):
        """
        computeFBEuler(Vector xF, Vector xB, Vector xF0, Vector xF1, FiniteElement element1, 
            Mesh mesh)
        """
        return _dolfinmodules.ElasticityUpdatedSolver_computeFBEuler(*args)

    if _newclass:computeFBEuler = staticmethod(computeFBEuler)
    __swig_getmethods__["computeFBEuler"] = lambda x: computeFBEuler
    def computeBEuler(*args):
        """computeBEuler(Vector xF, Vector xB, FiniteElement element1, Mesh mesh)"""
        return _dolfinmodules.ElasticityUpdatedSolver_computeBEuler(*args)

    if _newclass:computeBEuler = staticmethod(computeBEuler)
    __swig_getmethods__["computeBEuler"] = lambda x: computeBEuler
    def multF(*args):
        """multF(real F0, real F1, real F)"""
        return _dolfinmodules.ElasticityUpdatedSolver_multF(*args)

    if _newclass:multF = staticmethod(multF)
    __swig_getmethods__["multF"] = lambda x: multF
    def multB(*args):
        """multB(real F, real B)"""
        return _dolfinmodules.ElasticityUpdatedSolver_multB(*args)

    if _newclass:multB = staticmethod(multB)
    __swig_getmethods__["multB"] = lambda x: multB
    def deform(*args):
        """deform(Mesh mesh, Function u)"""
        return _dolfinmodules.ElasticityUpdatedSolver_deform(*args)

    if _newclass:deform = staticmethod(deform)
    __swig_getmethods__["deform"] = lambda x: deform
ElasticityUpdatedSolver_swigregister = _dolfinmodules.ElasticityUpdatedSolver_swigregister
ElasticityUpdatedSolver_swigregister(ElasticityUpdatedSolver)

def ElasticityUpdatedSolver_gather(*args):
    """ElasticityUpdatedSolver_gather(Vector x1, Vector x2, VecScatter x1sc)"""
    return _dolfinmodules.ElasticityUpdatedSolver_gather(*args)

def ElasticityUpdatedSolver_scatter(*args):
    """ElasticityUpdatedSolver_scatter(Vector x1, Vector x2, VecScatter x1sc)"""
    return _dolfinmodules.ElasticityUpdatedSolver_scatter(*args)

def ElasticityUpdatedSolver_createScatterer(*args):
    """ElasticityUpdatedSolver_createScatterer(Vector x1, Vector x2, int offset, int size) -> VecScatter"""
    return _dolfinmodules.ElasticityUpdatedSolver_createScatterer(*args)

def ElasticityUpdatedSolver_fromArray(*args):
    """ElasticityUpdatedSolver_fromArray(real u, Vector x, uint offset, uint size)"""
    return _dolfinmodules.ElasticityUpdatedSolver_fromArray(*args)

def ElasticityUpdatedSolver_toArray(*args):
    """ElasticityUpdatedSolver_toArray(real y, Vector x, uint offset, uint size)"""
    return _dolfinmodules.ElasticityUpdatedSolver_toArray(*args)

def ElasticityUpdatedSolver_fromDense(*args):
    """ElasticityUpdatedSolver_fromDense(uBlasVector u, Vector x, uint offset, uint size)"""
    return _dolfinmodules.ElasticityUpdatedSolver_fromDense(*args)

def ElasticityUpdatedSolver_toDense(*args):
    """ElasticityUpdatedSolver_toDense(uBlasVector y, Vector x, uint offset, uint size)"""
    return _dolfinmodules.ElasticityUpdatedSolver_toDense(*args)

def ElasticityUpdatedSolver_solve(*args):
    """
    solve()
    ElasticityUpdatedSolver_solve(Mesh mesh, Function f, Function v0, Function rho, real E, 
        real nu, real nuv, real nuplast, BoundaryCondition bc, 
        real k, real T)
    """
    return _dolfinmodules.ElasticityUpdatedSolver_solve(*args)

def ElasticityUpdatedSolver_finterpolate(*args):
  """ElasticityUpdatedSolver_finterpolate(Function f1, Function f2, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_finterpolate(*args)

def ElasticityUpdatedSolver_plasticity(*args):
  """
    ElasticityUpdatedSolver_plasticity(Vector xsigma, Vector xsigmanorm, real yld, FiniteElement element2, 
        Mesh mesh)
    """
  return _dolfinmodules.ElasticityUpdatedSolver_plasticity(*args)

def ElasticityUpdatedSolver_initmsigma(*args):
  """ElasticityUpdatedSolver_initmsigma(Vector msigma, FiniteElement element2, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_initmsigma(*args)

def ElasticityUpdatedSolver_initu0(*args):
  """ElasticityUpdatedSolver_initu0(Vector x0, FiniteElement element, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_initu0(*args)

def ElasticityUpdatedSolver_initJ0(*args):
  """ElasticityUpdatedSolver_initJ0(Vector xJ0, FiniteElement element, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_initJ0(*args)

def ElasticityUpdatedSolver_computeJ(*args):
  """
    ElasticityUpdatedSolver_computeJ(Vector xJ0, Vector xJ, Vector xJinv, FiniteElement element, 
        Mesh mesh)
    """
  return _dolfinmodules.ElasticityUpdatedSolver_computeJ(*args)

def ElasticityUpdatedSolver_initF0Green(*args):
  """ElasticityUpdatedSolver_initF0Green(Vector xF0, FiniteElement element1, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_initF0Green(*args)

def ElasticityUpdatedSolver_computeFGreen(*args):
  """
    ElasticityUpdatedSolver_computeFGreen(Vector xF, Vector xF0, Vector xF1, FiniteElement element1, 
        Mesh mesh)
    """
  return _dolfinmodules.ElasticityUpdatedSolver_computeFGreen(*args)

def ElasticityUpdatedSolver_initF0Euler(*args):
  """ElasticityUpdatedSolver_initF0Euler(Vector xF0, FiniteElement element1, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_initF0Euler(*args)

def ElasticityUpdatedSolver_computeFEuler(*args):
  """
    ElasticityUpdatedSolver_computeFEuler(Vector xF, Vector xF0, Vector xF1, FiniteElement element1, 
        Mesh mesh)
    """
  return _dolfinmodules.ElasticityUpdatedSolver_computeFEuler(*args)

def ElasticityUpdatedSolver_computeFBEuler(*args):
  """
    ElasticityUpdatedSolver_computeFBEuler(Vector xF, Vector xB, Vector xF0, Vector xF1, FiniteElement element1, 
        Mesh mesh)
    """
  return _dolfinmodules.ElasticityUpdatedSolver_computeFBEuler(*args)

def ElasticityUpdatedSolver_computeBEuler(*args):
  """ElasticityUpdatedSolver_computeBEuler(Vector xF, Vector xB, FiniteElement element1, Mesh mesh)"""
  return _dolfinmodules.ElasticityUpdatedSolver_computeBEuler(*args)

def ElasticityUpdatedSolver_multF(*args):
  """ElasticityUpdatedSolver_multF(real F0, real F1, real F)"""
  return _dolfinmodules.ElasticityUpdatedSolver_multF(*args)

def ElasticityUpdatedSolver_multB(*args):
  """ElasticityUpdatedSolver_multB(real F, real B)"""
  return _dolfinmodules.ElasticityUpdatedSolver_multB(*args)

def ElasticityUpdatedSolver_deform(*args):
  """ElasticityUpdatedSolver_deform(Mesh mesh, Function u)"""
  return _dolfinmodules.ElasticityUpdatedSolver_deform(*args)

class ElasticityUpdatedODE(_object):
    """Proxy of C++ ElasticityUpdatedODE class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ElasticityUpdatedODE, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ElasticityUpdatedODE, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self, ElasticityUpdatedSolver solver) -> ElasticityUpdatedODE"""
        this = _dolfinmodules.new_ElasticityUpdatedODE(*args)
        try: self.this.append(this)
        except: self.this = this
    def u0(*args):
        """u0(self, uBlasVector u)"""
        return _dolfinmodules.ElasticityUpdatedODE_u0(*args)

    def f(*args):
        """
        f(self, uBlasVector u, real t, uBlasVector y)
        f(self, uBlasVector u, real t, uint i) -> real
        f(self, uBlasVector u, real t, uBlasVector y)
        """
        return _dolfinmodules.ElasticityUpdatedODE_f(*args)

    def update(*args):
        """update(self, uBlasVector u, real t, bool end) -> bool"""
        return _dolfinmodules.ElasticityUpdatedODE_update(*args)

    __swig_setmethods__["solver"] = _dolfinmodules.ElasticityUpdatedODE_solver_set
    __swig_getmethods__["solver"] = _dolfinmodules.ElasticityUpdatedODE_solver_get
    if _newclass:solver = property(_dolfinmodules.ElasticityUpdatedODE_solver_get, _dolfinmodules.ElasticityUpdatedODE_solver_set)
ElasticityUpdatedODE_swigregister = _dolfinmodules.ElasticityUpdatedODE_swigregister
ElasticityUpdatedODE_swigregister(ElasticityUpdatedODE)

class UtilBC1(_object):
    """Proxy of C++ UtilBC1 class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UtilBC1, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UtilBC1, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> UtilBC1"""
        this = _dolfinmodules.new_UtilBC1(*args)
        try: self.this.append(this)
        except: self.this = this
    def eval(*args):
        """eval(self, BoundaryValue value, Point p, unsigned int i)"""
        return _dolfinmodules.UtilBC1_eval(*args)

UtilBC1_swigregister = _dolfinmodules.UtilBC1_swigregister
UtilBC1_swigregister(UtilBC1)

class UtilBC2(_object):
    """Proxy of C++ UtilBC2 class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UtilBC2, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UtilBC2, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> UtilBC2"""
        this = _dolfinmodules.new_UtilBC2(*args)
        try: self.this.append(this)
        except: self.this = this
    def eval(*args):
        """eval(self, BoundaryValue value, Point p, unsigned int i)"""
        return _dolfinmodules.UtilBC2_eval(*args)

UtilBC2_swigregister = _dolfinmodules.UtilBC2_swigregister
UtilBC2_swigregister(UtilBC2)

class Resistance(_object):
    """Proxy of C++ Resistance class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Resistance, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Resistance, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """__init__(self) -> Resistance"""
        this = _dolfinmodules.new_Resistance(*args)
        try: self.this.append(this)
        except: self.this = this
    def eval(*args):
        """eval(self, Point p, unsigned int i) -> real"""
        return _dolfinmodules.Resistance_eval(*args)

Resistance_swigregister = _dolfinmodules.Resistance_swigregister
Resistance_swigregister(Resistance)



