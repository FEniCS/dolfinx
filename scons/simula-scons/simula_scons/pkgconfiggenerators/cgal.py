#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getCgalDir(sconsEnv=None):
    return getPackageDir("cgal", sconsEnv=sconsEnv, default="/usr")

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
  # This is a bit special. It is given in the library as
  # a 10 digit number, like 1030511000. We have to do some arithmetics
  # to find the real version:
  # (VERSION / 1000 - 1000001) / 10000 => major version (3 in this case)
  # (VERSION / 1000 - 1000001) / 100 % 100 => minor version (5 in this case)
  # (VERSION / 1000 - 1000001) / 10 % 10 => sub-minor version (1 in this case).
  #
  # The version check also verify that we can include some CGAL headers.
    cpp_test_version_str = r"""
#include <CGAL/version.h>
#include <iostream>

int main() {
  #ifdef CGAL_VERSION_NR
    std::cout << CGAL_VERSION_NR;
  #endif
  return 0;
}
"""
    cppfile = "cgal_config_test_version.cpp"
    write_cppfile(cpp_test_version_str, cppfile);

    if not compiler:
        compiler = get_compiler(sconsEnv=sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)

    cmdstr = "%s -o a.out %s %s" % (compiler, cflags, cppfile)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cppfile)
        raise UnableToCompileException("CGAL", cmd=cmdstr,
                                       program=cpp_test_version_str,
                                       errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cppfile, execfile=True)
        raise UnableToRunException("CGAL", errormsg=cmdoutput)
    cgal_version_nr = int(cmdoutput)
    cgal_major = (cgal_version_nr / 1000 - 1000001) / 10000
    cgal_minor = (cgal_version_nr / 1000 - 1000001) / 100 % 100
    cgal_subminor = (cgal_version_nr / 1000 - 1000001) / 10 % 10
    full_cgal_version = "%s.%s.%s" % (cgal_major, cgal_minor, cgal_subminor)

    remove_cppfile(cppfile, execfile=True)

    return full_cgal_version

def pkgCflags(sconsEnv=None):
    return "-I%s -frounding-math" % \
           os.path.join(getCgalDir(sconsEnv=sconsEnv), "include")

def pkgLibs(sconsEnv=None, compiler=None, linker=None, cflags=None):
    ## libs = "-L%s -lCGAL" % os.path.join(getCgalDir(sconsEnv), "lib")
    ## if get_architecture() == 'darwin':
    ##     libs += " -lmpfr -lboost_thread-mt"
    ## return libs
    if not compiler:
        compiler = get_compiler(sconsEnv)
    if not linker:
        linker = get_linker(sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)

    # create a simple test program that uses CGAL
    cpp_test_libs_str = r"""
// =====================================================================================
//
// Copyright (C) 2010-06-13 André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-06-13
// Last changed: 2010-06-14
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================


#include <CGAL/AABB_tree.h> // *Must* be inserted before kernel!
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <CGAL/Simple_cartesian.h> 

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>

#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>


typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron_3;

typedef SCK::FT FT;
typedef SCK::Ray_3 Ray;
typedef SCK::Line_3 Line;
typedef SCK::Point_3 Point;
typedef SCK::Triangle_3 Triangle;

typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<SCK,Iterator> Primitive;
typedef CGAL::AABB_traits<SCK, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

typedef Nef_polyhedron_3::Aff_transformation_3 Aff_transformation_3;
typedef Nef_polyhedron_3::Plane_3 Plane_3;
typedef Nef_polyhedron_3::Vector_3 Vector_3;
typedef Nef_polyhedron_3::Point_3 Point_3;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron_3;


int main()
{
  //CGAL exact points
  Point_3 p1(0,0,0);
  Point_3 p2(1,0,0);
  Point_3 p3(0,1,0);
  Point_3 p4(0,0,1);
  
  Polyhedron_3 P;
  P.make_tetrahedron(p1,p2,p3,p4);
  Nef_polyhedron_3 NP(P);
  NP.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, 1)));
    
  //Inexact points
  Point a(1.0, 0.0, 0.0);
  Point b(0.0, 1.0, 0.0);
  Point c(0.0, 0.0, 1.0);
  Point d(0.0, 0.0, 0.0);

  std::list<Triangle> triangles;
  triangles.push_back(Triangle(a,b,c));
  triangles.push_back(Triangle(a,b,d));
  triangles.push_back(Triangle(a,d,c));

  // constructs AABB tree
  Tree tree(triangles.begin(),triangles.end());

  // counts #intersections
  Ray ray_query(a,b);
  std::cout << tree.number_of_intersected_primitives(ray_query)
      << " intersections(s) with ray query" << std::endl;

  // compute closest point and squared distance
  Point point_query(2.0, 2.0, 2.0);
  Point closest_point = tree.closest_point(point_query);

  return 0;
}
"""
    cpp_file = "cgal_test_libs.cpp"
    write_cppfile(cpp_test_libs_str, cpp_file);

    # test that we can compile
    cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("CGAL", cmd=cmdstr,
                                       program=cpp_test_libs_str,
                                       errormsg=cmdoutput)

    # test that we can link
    libs = "-L%s -lCGAL" % os.path.join(getCgalDir(sconsEnv=sconsEnv), "lib")
    cmdstr = "%s -o a.out %s %s" % \
           (linker, cpp_file.replace('.cpp', '.o'), libs)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        # try adding -lmpfr -lgmp -lboost_thread
        libs += " -lmpfr -lgmp -lboost_thread"
        if get_architecture() == "darwin":
            # also add -L/opt/local/lib on mac (assume MacPorts)
            libs += " -L/opt/local/lib"
        cmdstr = "%s -o a.out %s %s" % \
                 (linker, cpp_file.replace('.cpp', '.o'), libs)
        linkFailed, cmdoutput = getstatusoutput(cmdstr)
        if linkFailed:
            # try to append -mt to boost_thread lib
            libs += "-mt"
            cmdstr = "%s -o a.out %s %s" % \
                     (linker, cpp_file.replace('.cpp', '.o'), libs)
            linkFailed, cmdoutput = getstatusoutput(cmdstr)
            if linkFailed:
                remove_cppfile(cpp_file, ofile=True)
                raise UnableToCompileException("CGAL", cmd=cmdstr,
                                               program=cpp_test_libs_str,
                                               errormsg=cmdoutput)

    # test that we can run the binary                                           
    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    remove_cppfile(cpp_file, ofile=True, execfile=True)
    if runFailed:
        raise UnableToRunException("CGAL", errormsg=cmdoutput)

    return libs

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
        libs = pkgLibs(sconsEnv=sconsEnv, compiler=compiler,
                       linker=linker, cflags=cflags)
    else:
        # run pkgLibs as this is the current CGAL test
        pkgLibs(sconsEnv=sconsEnv, compiler=compiler,
                linker=linker, cflags=cflags)
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv, compiler=compiler,
                             linker=linker, cflags=cflags, libs=libs)

    return version, libs, cflags

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):
    if directory is None:
        directory = suitablePkgConfDir()

    version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: cgal
Version: %s
Description: Computational Geometry Algorithms Library
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
    # FIXME:      ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
    # Is there a better way to handle this on Windows?

    pkg_file = open(os.path.join(directory, "cgal.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found cgal and generated pkg-config file in\n '%s'" \
          % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
