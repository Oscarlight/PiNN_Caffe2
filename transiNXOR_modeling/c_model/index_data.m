function [ idx ] = index_data( points, data )
%INDEX_DATA find index of closest voltage to input voltage
% for example if data = [0.1 , 0.2, 0.3, 0.4] and point = 0.22, output is 2
    idx = ones(1, length(points));
    for i = 1:length(points)
        [~, idx(i)] = min(abs(data - points(i)));
    end

end
