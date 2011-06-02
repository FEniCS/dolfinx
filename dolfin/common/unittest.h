// Copyright (C) 2007 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2007-05-14
// Last changed: 2007-05-24
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
