% Copyright (C) 2003 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Generate plots from the multi-adaptivity test (for Octave)

primal

grid on
xlabel('t')
ylabel('u')
gset terminal postscript eps color

title('multi-adaptive solution components 1-3')
plot(t, u(1:3, :), [";"]);
gset output "single_u_1-3.eps"
gset output "foo"

title('multi-adaptive solution components 101-103')
plot(t, u(101:103, :), [";"]);
gset output "single_u_101-103.eps"
gset output "foo"

title('multi-adaptive solution component 3')
plot(t, u(3, :), [";"]);
gset output "single_u_3.eps"
gset output "foo"

ylabel('k')

title('multi-adaptive step size all components')
semilogy(t, k(:, :), [";"]);
gset output "single_k.eps"
gset output "foo"

gset terminal x11
