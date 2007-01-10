def import_formfile(filename):
    """Generates and imports a module corresponding to the FFC form file.
    Returns the module."""
    from ffc.parser import simple
    from ffc.common.constants import FFC_OPTIONS

    print "Compiling form: " + filename

    options = FFC_OPTIONS
    language = "dolfin"
        
    outname = simple.parse(filename, language, options)
        
    formname = filename.split(".")[0]
    __import__(formname)

    return import_header(formname + ".h")

def import_form(formlist, formname):
    """Generates and imports a module corresponding to the FFC form.
    Returns the module."""

    from ffc.compiler.compiler import compile
    from ffc.common.constants import FFC_OPTIONS

    options = FFC_OPTIONS
    language = "dolfin"
        
    compile(formlist, formname, language, FFC_OPTIONS)

    return import_header(formname + ".h")

def import_header(headername):
    """Generates and imports a module corresponding to the DOLFIN header
    file (with implementation). Returns the module."""

    from os import system
    from commands import getoutput

    lowername = (headername.split(".")[0]).lower()
    modulename = lowername

    output = getoutput("dolfin-swig " + headername + " " + modulename)
    print output

    return __import__(modulename)
