% Copyright (C) 2004 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Plot solution of the Medical Akzo-Nobel problem.

primal

n = size(u, 1) / 2;
h = 1/n;
x = linspace(h, 1, n);
[X, T] = meshgrid(x, t);

% Pick out the solution (stored at even and odd indices)
iU = 1:2:(2*n-1);
iV = 2:2:2*n;
U  = zeros(size(X));
V  = zeros(size(X));
kU = zeros(size(X));
kV = zeros(size(X));

for i=1:length(t)
  U(i,:)  = u(iU, i)';
  V(i,:)  = u(iV, i)';
  kU(i,:) = k(iU, i)';
  kV(i,:) = k(iV, i)';
end

figure(1)
clf
subplot(2,1,1)
surf(X, T, U)
xlabel('x')
ylabel('t')
subplot(2,1,2)
surf(X, T, V)
xlabel('x')
ylabel('t')
title('Solution')

figure(2)
clf
subplot(2,1,1)
surf(X, T, kU)
xlabel('x')
ylabel('t')
subplot(2,1,2)
surf(X, T, kV)
xlabel('x')
ylabel('t')
title('Time steps')
