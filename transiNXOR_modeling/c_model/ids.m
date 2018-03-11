function [ id ] = ids( vtg, vbg, vds, w, params )
%IDS calculates ids vs vtg, vbg, vds, w
% All voltages are in volts, w is in microns and acts as scale factor
% params should be preloaded from mat file
%   
    vg = [vtg; vbg];
    vg = (vg - 0.1)/0.1; 
    vds = vds/0.2;
    
    % first layer    
    [sig_0_act, tanh_0_act] = pinn_layer(vg, vds, params.tanh_fc_layer_0_w,...
        params.sig_fc_layer_0_w, params.sig_fc_layer_0_b', params.inter_embed_layer_0_w,...
        params.inter_embed_layer_0_b');
    % second layer
    [sig_1_act, tanh_1_act] = pinn_layer(sig_0_act, tanh_0_act, params.tanh_fc_layer_1_w,...
        params.sig_fc_layer_1_w, params.sig_fc_layer_1_b', params.inter_embed_layer_1_w,...
        params.inter_embed_layer_1_b');
    % third layer
    [sig_2_act, tanh_2_act] = pinn_layer(sig_1_act, tanh_1_act, params.tanh_fc_layer_2_w,...
        params.sig_fc_layer_2_w, params.sig_fc_layer_2_b', params.inter_embed_layer_2_w,...
        params.inter_embed_layer_2_b');

    % calculate id
    id = w * sig_2_act * tanh_2_act * 53.65093994 * 1e-6;

end

function [sig_act, tanh_act] = pinn_layer(sig_in, tanh_in, tanh_w, sig_w, sig_b, inter_w, inter_b)
    tanh_z = tanh_w*tanh_in;
    sig_z = fc(sig_in, sig_w, sig_b);
    inter_z = fc(tanh_z, inter_w, inter_b);
    sig_act = sigmoid(sig_z + inter_z);
    tanh_act = tanh(tanh_z);
end

function out = fc(in, w, b)
% fully connected layer
out = w*in + b;
end

function sig = sigmoid(x)
    %sig = 1./(1 + exp(-x));
    sig = (tanh(x/2) + 1)/2;
end

