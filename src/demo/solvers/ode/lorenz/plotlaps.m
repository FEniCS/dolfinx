% Copyright (C) 2005 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Plot the number of laps around each fixed point for Lorenz.

load lorenz1.txt

t     = lorenz1(:, 1);
n0    = lorenz1(:, 3);
n1    = lorenz1(:, 4);
alpha = lorenz1(:, 5);

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
plot(t(1:1339), alpha(1:1339))
xlabel('t')
ylabel('n1 / n2')

subplot(2, 1, 2)
plot(t, alpha)
xlabel('t')
ylabel('n1 / n2')

disp(['alpha at end of integration: ' num2str(alpha(length(alpha)))])
