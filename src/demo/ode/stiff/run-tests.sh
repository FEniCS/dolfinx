#!/bin/sh
#
# Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run all stiff test problems.

echo Running test problem 1
./dolfin-ode-stiff 1 > log-1
#mv primal.m primal_1.m

echo Running test problem 2
./dolfin-ode-stiff 2 > log-2
#mv primal.m primal_2.m

echo Running test problem 3
./dolfin-ode-stiff 3 > log-3
#mv primal.m primal_3.m

echo Running test problem 4
./dolfin-ode-stiff 4 > log-4
#mv primal.m primal_4.m

echo Running test problem 5
./dolfin-ode-stiff 5 > log-5
#mv primal.m primal_5.m

echo Running test problem 6
./dolfin-ode-stiff 6 > log-6
#mv primal.m primal_6.m

echo Running test problem 7
./dolfin-ode-stiff 7 > log-7
#mv primal.m primal_7.m

echo Running test problem 8
./dolfin-ode-stiff 8 > log-8
#mv primal.m primal_8.m

echo Running test problem 9
./dolfin-ode-stiff 9 > log-9
#mv primal.m primal_9.m
