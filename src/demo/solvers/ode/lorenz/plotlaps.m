% Copyright (C) 2005 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Plot the number of laps around each fixed point for Lorenz.

load lorenztmp.txt

t     = lorenztmp(:, 1);
n0    = lorenztmp(:, 3);
n1    = lorenztmp(:, 4);
alpha = lorenztmp(:, 5);

figure(1)
clf

subplot(2, 1, 1)
plot(t, n0)
xlabel('t')
ylabel('n1')

subplot(2, 1, 2)
plot(t, n1)
xlabel('t')
ylabel('n2')

figure(2)
clf

subplot(2, 1, 1)
plot(t(1:1000), alpha(1:1000))
xlabel('t')
ylabel('n1 / n2')

subplot(2, 1, 2)
plot(t, alpha)
xlabel('t')
ylabel('n1 / n2')

disp(['alpha at end of integration: ' num2str(alpha(size(alpha)))])
