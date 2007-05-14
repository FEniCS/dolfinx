// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-14
// Last changed: 2007-05-14
//
// Simple macro for DOLFIN unit tests (using cppunit). The macro
// DOLFIN_TEST removes the need of repeating the same 8 lines of
// code in every single unit test.

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/CompilerOutputter.h>

#define DOLFIN_TEST \
CppUnit::TestResult result; \
CppUnit::TestResultCollector collected_results; \
result.addListener(&collected_results); \
CppUnit::TestRunner runner; \
runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest()); \
runner.run(result); \
CppUnit::CompilerOutputter outputter(&collected_results, std::cerr); \
outputter.write (); \
return collected_results.wasSuccessful () ? 0 : 1;
