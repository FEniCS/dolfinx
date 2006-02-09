% Load data
primal_0; t0 = t; u0 = u; k0 = k;
primal_1; t1 = t; u1 = u; k1 = k;
primal_2; t2 = t; u2 = u; k2 = k;

clf

% Plot solutions
subplot(3, 3, 1)
plot(t0', u0')
xlabel('t')
ylabel('Re z, Im z')

subplot(3, 3, 2)
plot(t1', u1')
xlabel('t')

subplot(3, 3, 3)
plot(t2', u2')
xlabel('t')

% Plot time steps
subplot(3, 3, 4)
plot(t0', k0')
xlabel('t')
ylabel('k')

subplot(3, 3, 5)
plot(t1', k1')
xlabel('t')

subplot(3, 3, 6)
plot(t2', k2')
xlabel('t')

% Plot paths
subplot(3, 3, 7)
plot(u0(1, :), u0(2, :))
xlabel('Re z')
ylabel('Im z')

subplot(3, 3, 8)
plot(u1(1, :), u1(2, :))
xlabel('Re z')

subplot(3, 3, 9)
plot(u2(1, :), u2(2, :))
xlabel('Re z')
