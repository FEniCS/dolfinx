def import_formfile(filename):
    """Generates and imports a module corresponding to the FFC form file.
    Returns the module."""
    from ffc.parser import simple
    
    from ffc.common.constants import FFC_OPTIONS

    options = FFC_OPTIONS
    language = "dolfin-swig"
        
    outname = simple.parse(filename, language, options)
        
    formname = filename.split(".")[0]
    __import__(formname)

    return generate_form_module(formname)

def import_form(formlist, formname):
    """Generates and imports a module corresponding to the FFC form.
    Returns the module."""

    from ffc.compiler.compiler import compile
    from ffc.common.constants import FFC_OPTIONS

    options = FFC_OPTIONS
    language = "dolfin-swig"
        
    compile(formlist, formname, language, FFC_OPTIONS)

    return generate_form_module(formname)

def generate_form_module(formname):
    """Generates and imports a module corresponding to the FFC/DOLFIN header
    file. Returns the module."""

    from os import system
    from commands import getoutput

    output = getoutput("dolfin-swig " + formname + ".h")
    print output

    formmodulename = formname.lower() + "form"

    return __import__(formmodulename)
