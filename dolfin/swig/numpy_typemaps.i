
//--------------------------------------------------------------------------
// Input typemaps (from Python to C++)
//-------------------------------------------------------------------------- 

%typemap(in) double* _array {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            $1  = static_cast<double*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 64 bit floats expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}

%typemap(in) int * _array {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_INT ) {
            $1 = static_cast<int*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 32 bit integers (int32) expected. Make sure that the numpy array use dtype='i'.");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}

%typemap(in) unsigned int* _array {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_UINT ) {
            $1 = static_cast<unsigned int*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 32 bit unsigned integers (uint32) expected. Make sure that the numpy array use dtype='I'.");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}


%typemap(in) long * _array {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_LONG ) {
            $1  = static_cast<long*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 64 bit integers expected" Make sure that the numpy array use dtype='l'.);
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}

%typemap(in) (int _array_dim, int* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_INT ) {
            $1 = static_cast<$1_type>(PyArray_DIM(xa,0));
            $2 = static_cast<int*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 32 bit integers (int32) expected. Make sure that the numpy array use dtype='i'.");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}

%typemap(in) (int _array_dim, unsigned int* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_UINT ) {
            $1 = static_cast<$1_type>(PyArray_DIM(xa,0));
            $2 = static_cast<unsigned int*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Numpy array of 32 bit unsigned integers (uint32) expected. Make sure that the numpy array use dtype='I'.");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Numpy array expected");
    }
}


%typemap(in) (int _array_dim, double* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            $1 = PyArray_DIM(xa,0);
            $2 = static_cast<double*>(PyArray_DATA(xa));
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

/**
 * Generic typemap to expand a two-dimensional Numeric arrays into three
 * C++ arguments: _array_dim_0, _array_dim_1, _array
 */
%typemap(in) (int _array_dim_0, int _array_dim_1, double* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                $1 = PyArray_DIM(xa,0);
                $2 = PyArray_DIM(xa,1);
                $3  = static_cast<double*>(PyArray_DATA(xa));
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

/**
 * Generic typemap to expand a two-dimensional Numeric arrays into three
 * C++ arguments: _array_dim_0, _array_dim_1, _array
 */
%typemap(in) (int _array_dim_0, int _array_dim_1, int* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_INT ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                $1 = PyArray_DIM(xa,0);
                $2 = PyArray_DIM(xa,1);
                $3  = static_cast<int*>(PyArray_DATA(xa));
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of integers expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

%{
namespace __private {
    class dppDeleter {
    public:
        double** amat;
        dppDeleter () {amat = 0;}
        ~dppDeleter ()
        {
            free(amat);
        }
    };
}
%}

%typemap(in) double** (__private::dppDeleter tmp){

    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                const int n = PyArray_DIM(xa,0);
                double **amat = static_cast<double**>(malloc(n*sizeof*amat));
                double *data = reinterpret_cast<double*>((*xa).data);
                for (int i=0;i<n;++i)
                    amat[i] = data + i*n;
                $1 = amat;
                tmp.amat = amat;
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

%typemap(in) (int _matrix_dim_0, int _matrix_dim_1, double** _matrix) (__private::dppDeleter tmp){

    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject *>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                int n = PyArray_DIM(xa,0);
                int m = PyArray_DIM(xa,1);
                $1 = n;
                $2 = m;
                double **amat = static_cast<double **>(malloc(n*sizeof*amat));
                double *data = reinterpret_cast<double *>(PyArray_DATA(xa));
                for (int i=0;i<n;++i)
                    amat[i] = data + i*n;
                $3 = amat;
                tmp.amat = amat;
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

%typemap(in) (int _array_dim_0, int _array_dim_1, int* _array) {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_INT ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                $1 = PyArray_DIM(xa,0);
                $2 = PyArray_DIM(xa,1);
                $3 = reinterpret_cast<int*>(PyArray_DATA(xa));
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

//--------------------------------------------------------------------------
// Various typemaps 
//--------------------------------------------------------------------------

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (int _array_dim_0, int _array_dim_1, double* _array) {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (int _array_dim, double* _array) {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%apply int { unsigned int };

// vim:ft=cpp:
