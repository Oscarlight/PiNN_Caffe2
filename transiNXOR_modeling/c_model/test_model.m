vmin_data = -0.1;
vmax_data = 0.3;

model_N = 1000;
vmin_plot = 0.0;
vmax_plot = 0.2;

vtg = 0.2;
vbg = 0.2;

params = load('params.mat');
preproc = load('preproc.mat');

params.VD_SCALE = preproc.vd;
params.VG_SHIFT = preproc.vg_shift;
params.VG_SCALE = preproc.vg;
params.ID_SCALE = preproc.id;

% evaluate using both native c model and native matlab model
res = zeros(1, model_N); % for native c model
res_mat = zeros(1, model_N); % for matlab model
vds_model = linspace(vmin_plot, vmax_plot, model_N);

for i = 1:model_N
    res(i) = wrappedModel(vtg, vbg, vds_model(i), 1);
    res_mat(i) = ids(vtg, vbg, vds_model(i), 1, params);
end

% load actual data
current_data = getfield(load('current.mat'), 'data')*1e-6;
no_points = size(current_data, 1);
vds_data = linspace(vmin_data, vmax_data, no_points);

vds_delta = (vmax_data-vmin_data)/(no_points-1); % vds spacing
vds_plot = vmin_plot:vds_delta:vmax_plot;
index = @(v) index_data(v, vds_data);
sim_data = squeeze(squeeze(current_data(index(vds_plot), index(vbg), index(vtg))))';

plot(vds_model, res, 'LineWidth', 1.5);
hold on
plot(vds_model, res_mat, 'LineWidth', 1.5);
plot(vds_plot, sim_data, 'k+', 'LineWidth', 1.5);
legend({'c\_model', 'matlab', 'data'}, 'Location', 'Best', 'FontSize', 14);
hold off
xlabel('vds (V)')
ylabel('ids');
print(gcf,'model_fit.png','-dpng','-r300');
