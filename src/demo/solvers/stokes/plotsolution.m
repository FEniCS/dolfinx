% Plot solution using MATLAB

% Load solution
pressure
velocity

% Plot velocity
figure(1); clf
pdeplot(points, edges, cells, 'flowdata', u)
axis([0 1 0 1])
xlabel('x')
ylabel('y')
title('Stokes velocity')

% Plot pressure
figure(2); clf
pdesurf(points, cells, p)
axis([0 1 0 1])
xlabel('x')
ylabel('y')
title('Stokes pressure')
