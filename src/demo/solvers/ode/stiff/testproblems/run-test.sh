#!/bin/sh

# Run all stiff test problems. Test should be run with compiler flags
# -O3 -Wall -Werror.

echo Running test problem 1
./dolfin-ode-stiff-testproblems 1 > log-1
mv primal.m primal_1.m
mv primal.debug primal_1.debug

echo Running test problem 2
./dolfin-ode-stiff-testproblems 2 > log-2
mv primal.m primal_2.m
mv primal.debug primal_2.debug

echo Running test problem 3
./dolfin-ode-stiff-testproblems 3 > log-3
mv primal.m primal_3.m
mv primal.debug primal_3.debug

echo Running test problem 4
./dolfin-ode-stiff-testproblems 4 > log-4
mv primal.m primal_4.m
mv primal.debug primal_4.debug

echo Running test problem 5
./dolfin-ode-stiff-testproblems 5 > log-5
mv primal.m primal_5.m
mv primal.debug primal_5.debug

echo Running test problem 6
./dolfin-ode-stiff-testproblems 6 > log-6
mv primal.m primal_6.m
mv primal.debug primal_6.debug

echo Running test problem 7
./dolfin-ode-stiff-testproblems 7 > log-7
mv primal.m primal_7.m
mv primal.debug primal_7.debug

echo Running test problem 8
./dolfin-ode-stiff-testproblems 8 > log-8
mv primal.m primal_8.m
mv primal.debug primal_8.debug

echo Running test problem 9
./dolfin-ode-stiff-testproblems 9 > log-9
mv primal.m primal_9.m
mv primal.debug primal_9.debug
