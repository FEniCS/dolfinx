#!/bin/sh
#
# Run all stiff test problems.

echo Running test problem 1
./dolfin-ode-stiff 1 > log-1

echo Running test problem 2
./dolfin-ode-stiff 2 > log-2

echo Running test problem 3
./dolfin-ode-stiff 3 > log-3

echo Running test problem 4
./dolfin-ode-stiff 4 > log-4

echo Running test problem 5
./dolfin-ode-stiff 5 > log-5

echo Running test problem 6
./dolfin-ode-stiff 6 > log-6

echo Running test problem 7
./dolfin-ode-stiff 7 > log-7

echo Running test problem 8
./dolfin-ode-stiff 8 > log-8

echo Running test problem 9
./dolfin-ode-stiff 9 > log-9
