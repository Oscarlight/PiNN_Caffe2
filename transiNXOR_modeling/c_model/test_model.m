params = load('params.mat');

N = 1000;
res = zeros(1, N);
res_mat = zeros(1, N);

vds = linspace(0, 0.2, N);
for i = 1:N
    res(i) = wrappedModel(0.2, 0.0, vds(i), 1);
    res_mat(i) = ids(0.2, 0.0, vds(i), 1, params);
end

plot(vds, res, 'LineWidth', 1.5);
hold on
plot(vds, res_mat, 'LineWidth', 1.5);
legend({'c_model', 'matlab'});
hold off