% Code by Shan Gao


% input: 3d matrix, each layer is 1 subject's data (timepoints * region)
%        fisherz = 1: do fisher's z transform; 0: no transform, return r.
% output: 2d leave-one-out isc matrix (subj * region)


function subj_by_region_isc_2d = isc(sample_by_region_3d, fisherz)

    fprintf('Computing ISC...\n')

    num_subj = size(sample_by_region_3d, 3);
    num_region = size(sample_by_region_3d, 2);
    
    % create empty matrix for isc to be filled in
    subj_by_region_isc_2d = zeros(num_subj, num_region);

    % compute isc
    for i = 1:num_subj  % for each layer (subj)
        
        fprintf('subj %d\n', i)
        
        % get the data layer for current subj
        curr_subj = sample_by_region_3d(:, :, i);
        
        % get all other subjs' data layers
        others_indices = setdiff(1:num_subj, i);
        others_data = sample_by_region_3d(:, :, others_indices);
        
        % get timepoints * region mean values for all other subjs except subj i
        others_mean = nanmean(others_data, 3);
        
        % compute isc by region: 
        % corr between curr_subj and others_mean across all timepoints
        for region = 1:num_region
            
            r = nancorr(curr_subj(:, region), ...
                        others_mean(:, region));  % handles TC with NaNs
            
            if fisherz == 1
                subj_by_region_isc_2d(i, region) = atanh(r);    % fisher's z transform
            else
                subj_by_region_isc_2d(i, region) = r;
            end
            
        end

    end

end


