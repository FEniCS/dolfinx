#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getLibxml2Dir(sconsEnv=None):
    libxml2_dir = getPackageDir("libxml2", sconsEnv=sconsEnv,
                                default=os.path.join(os.path.sep, "usr"))
    return libxml2_dir

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
    arch = get_architecture()
    if arch.startswith("win"):
        xmlversion_h = "xmlwin32version.h"
    else:
        xmlversion_h = "xmlversion.h"
    cpp_test_version_str = r"""
#include <stdio.h>
#include <libxml/%s>

int main() {
  printf(LIBXML_DOTTED_VERSION);
  return 0;
}
""" % xmlversion_h
    cppfile = "libxml2_config_test_version.cpp"
    write_cppfile(cpp_test_version_str, cppfile);

    if not compiler:
        compiler = get_compiler(sconsEnv=sconsEnv)
    if not linker:
        compiler = get_linker(sconsEnv=sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)
    if not libs:
        libs = pkgLibs(sconsEnv=sconsEnv)

    cmdstr = "%s %s -c %s" % (compiler, cflags, cppfile)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cppfile)
        raise UnableToCompileException("libXML2", cmd=cmdstr,
                                       program=cpp_test_version_str,
                                       errormsg=cmdoutput)

    cmdstr = "%s %s -o a.out %s" % (linker, cppfile.replace('.cpp', '.o'), libs)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        remove_cppfile(cppfile, ofile=True)
        raise UnableToLinkException("libXML2", cmd=cmdstr,
                                    program=cpp_test_version_str,
                                    errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cppfile, ofile=True, execfile=True)
        raise UnableToRunException("libXML2", errormsg=cmdoutput)
    version = cmdoutput

    remove_cppfile(cppfile, ofile=True, execfile=True)
    return version

def pkgCflags(sconsEnv=None):
    include_dir = os.path.join(getLibxml2Dir(sconsEnv=sconsEnv), "include")
    if os.path.exists(os.path.join(include_dir, "libxml2",
                                   "libxml", "xmlversion.h")):
        include_dir = os.path.join(include_dir, "libxml2")
    return "-I%s" % include_dir

def pkgLibs(sconsEnv=None):
    if get_architecture().startswith("win"):
        return "-L%s -L%s -lxml2" % \
               (os.path.join(getLibxml2Dir(sconsEnv), "bin"),
                os.path.join(getLibxml2Dir(sconsEnv), "lib"))
    else:
        return "-L%s -lxml2" % os.path.join(getLibxml2Dir(sconsEnv), "lib")

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
    """Run the tests for this package
     
    If Ok, return various variables, if not we will end with an exception.
    forceCompiler, if set, should be a tuple containing (compiler, linker)
    or just a string, which in that case will be used as both
    """

    if not forceCompiler:
        compiler = get_compiler(sconsEnv)
        linker = get_linker(sconsEnv)
    else:
        compiler, linker = set_forced_compiler(forceCompiler)

    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)
    if not libs:
        libs = pkgLibs(sconsEnv=sconsEnv)
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv, compiler=compiler,
                             linker=linker, cflags=cflags, libs=libs)

    # A program that do a real libXML2 test
    cpp_test_lib_str = r"""
/**
 * section: xmlReader
 * synopsis: Parse an XML file with an xmlReader
 * purpose: Demonstrate the use of xmlReaderForFile() to parse an XML file
 *          and dump the informations about the nodes found in the process.
 *          (Note that the XMLReader functions require libxml2 version later
 *          than 2.6.)
 * usage: reader1 <filename>
 * test: reader1 test2.xml > reader1.tmp ; diff reader1.tmp reader1.res ; rm reader1.tmp
 * author: Daniel Veillard
 * copy: see Copyright for the status of this software.
 */

#include <stdio.h>
#include <libxml/xmlreader.h>

#ifdef LIBXML_READER_ENABLED

/**
 * processNode:
 * @reader: the xmlReader
 *
 * Dump information about the current node
 */
static void
processNode(xmlTextReaderPtr reader) {
    const xmlChar *name, *value;

    name = xmlTextReaderConstName(reader);
    if (name == NULL)
	name = BAD_CAST "--";

    value = xmlTextReaderConstValue(reader);

    printf("%d %d %s %d %d", 
	    xmlTextReaderDepth(reader),
	    xmlTextReaderNodeType(reader),
	    name,
	    xmlTextReaderIsEmptyElement(reader),
	    xmlTextReaderHasValue(reader));
    if (value == NULL)
	printf("\n");
    else {
        if (xmlStrlen(value) > 40)
            printf(" %.40s...\n", value);
        else
	    printf(" %s\n", value);
    }
}

/**
 * streamFile:
 * @filename: the file name to parse
 *
 * Parse and print information about an XML file.
 */
static void
streamFile(const char *filename) {
    xmlTextReaderPtr reader;
    int ret;

    reader = xmlReaderForFile(filename, NULL, 0);
    if (reader != NULL) {
        ret = xmlTextReaderRead(reader);
        while (ret == 1) {
            processNode(reader);
            ret = xmlTextReaderRead(reader);
        }
        xmlFreeTextReader(reader);
        if (ret != 0) {
            fprintf(stderr, "%s : failed to parse\n", filename);
        }
    } else {
        fprintf(stderr, "Unable to open %s\n", filename);
    }
}

int main(int argc, char **argv) {
    if (argc != 2)
        return(1);

    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    streamFile(argv[1]);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();
    return(0);
}

#else
int main(void) {
    fprintf(stderr, "XInclude support not compiled in\n");
    exit(1);
}
#endif
"""
    cpp_file = "libxml2_config_test_lib.cpp"
    write_cppfile(cpp_test_lib_str, cpp_file);

    libxml2file = "libxml2_test_xml_file.xml"
    tmp_ = open(libxml2file, "w")
    tmp_.write(r"<doc/>")
    tmp_.close()

    # try to compile and run the libXML2 test program
    cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        os.remove(libxml2file)
        raise UnableToCompileException("libXML2", cmd=cmdstr,
                                       program=cpp_test_lib_str,
                                       errormsg=cmdoutput)

    cmdstr = "%s %s -o a.out %s" % \
             (linker, cpp_file.replace('.cpp', '.o'), libs)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        remove_cppfile(cpp_file, ofile=True)
        os.remove(libxml2file)
        raise UnableToLinkException("libXML2", cmd=cmdstr,
                                    program=cpp_test_lib_str,
                                    errormsg=cmdoutput)

    cmdstr = "%s %s" % (os.path.join(os.getcwd(), "a.out"), libxml2file)
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cpp_file, ofile=True, execfile=True)
        os.remove(libxml2file)
        raise UnableToRunException("libXML2", errormsg=cmdoutput)
    if not cmdoutput == "0 1 doc 1 0":
        errormsg = "libXML2 test does not produce correct result, " \
                   "check your libXML2 installation."
        errormsg += "\n%s" % cmdoutput
        raise UnableToRunException("libXML2", errormsg=errormsg)

    remove_cppfile(cpp_file, ofile=True, execfile=True)
    os.remove(libxml2file)

    return version, cflags, libs

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    
    version, cflags, libs = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: libXML
Version: %s
Description: libXML library version 2.
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])

    pkg_file = open(os.path.join(directory, "libxml-2.0.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found libXML2 and generated pkg-config file in\n '%s'" \
          % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
