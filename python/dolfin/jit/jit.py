# -*- coding: utf-8 -*-

import numpy
import hashlib
import dijitso
import dolfin.cpp as cpp

_cpp_math_builtins = [
    # <cmath> functions: from http://www.cplusplus.com/reference/cmath/
    "cos", "sin", "tan", "acos", "asin", "atan", "atan2",
    "cosh", "sinh", "tanh", "exp", "frexp", "ldexp", "log", "log10", "modf",
    "pow", "sqrt", "ceil", "fabs", "floor", "fmod",
    "max", "min"]

_math_header = """
// cmath functions
%s

const double pi = DOLFIN_PI;
""" % "\n".join("using std::%s;" % mf for mf in _cpp_math_builtins)


def compile_class(cpp_data):
    """Compile a user C(++) string or set of statements to a Python object

    cpp_data is a dict containing:
      "name": must be "expression" or "subdomain"
      "statements": must be a string, or list/tuple of strings
      "properties": a dict of float properties
      "jit_generate": callable (generates cpp code with this dict as input)

    """

    import pkgconfig
    if not pkgconfig.exists('dolfin'):
        raise RuntimeError("Could not find DOLFIN pkg-config file. Please make sure appropriate paths are set.")

    # Get DOLFIN pkg-config data
    d = pkgconfig.parse('dolfin')

    # Set compiler/build options
    params = dijitso.params.default_params()
    params['build']['include_dirs'] = d["include_dirs"]
    params['build']['libs'] = d["libraries"]
    params['build']['lib_dirs'] = d["library_dirs"]

    name = cpp_data['name']
    if name not in ('subdomain', 'expression'):
        raise ValueError("DOLFIN JIT only for SubDomain and Expression")
    statements = cpp_data['statements']
    properties = cpp_data['properties']

    if not isinstance(statements, (str, tuple, list)):
        raise RuntimeError("Expression must be a string, or a list or tuple of strings")

    # Flatten tuple of tuples (2D array) and get value_shape
    statement_array = numpy.array(statements)
    cpp_data['statements'] = tuple(statement_array.flatten())
    cpp_data['value_shape'] = statement_array.shape

    # Make a string representing the properties (and distinguish float/GenericFunction)
    # by adding '*' for GenericFunction
    property_str = ''
    for k,v in properties.items():
        property_str += str(k)
        if hasattr(v, '_cpp_object') and isinstance(v._cpp_object, cpp.function.GenericFunction):
            property_str += '*'

    hash_str = str(statements) + str(property_str)
    module_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    module_name = "dolfin_" + name + "_" + module_hash

    try:
        module, signature = dijitso.jit(cpp_data, module_name, params,
                                        generate=cpp_data['jit_generate'])
        submodule = dijitso.extract_factory_function(module, "create_" + module_name)()
    except:
        raise RuntimeError("Unable to compile C++ code with dijitso")

    if name == 'expression':
        python_object = cpp.function.make_dolfin_expression(submodule)
    else:
        python_object = cpp.mesh.make_dolfin_subdomain(submodule)

    # Set properties to initial values
    # FIXME: maybe remove from here (do it in Expression and SubDomain instead)
    for k, v in properties.items():
        python_object.set_property(k, v)

    return python_object
