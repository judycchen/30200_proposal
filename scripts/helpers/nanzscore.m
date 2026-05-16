% Code by Shan Gao


% MATLAB built-in function zscore() will cause the entire vector to be nan if
% there is one nan.
% Therefore, nanzscore() zscores [data] with NaNs along dimension [dim], 
% using only non-NaN data points to prevent setting the entire vector to nan 
% because of some missing data points.
% e.g., for data with shape TxRxS, dim=1, nanzscore() zscores the data for
% each Region and Subject across Time.

function zdata = nanzscore(data, dim)

    zdata = (data - nanmean(data, dim)) ./ std(data, [], dim, "omitnan");

end