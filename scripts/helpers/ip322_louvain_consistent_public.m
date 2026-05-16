function [Q,most_common_module_mat] = ip322_louvain_consistent_public(conn_matrix,initial_vector,lambda,all_weights,predefined_,predefined_vector)
%% Description

%Required additional functions to be in MATLAB path: BCT
%(folder 2017_01_15_BCT)

%Input: 
%conn_matrix:connectivity matrix (weighted/unweighted/full/thresholded, etc.)

%initial_vector: initial assignment of regions to modules (if any) for initialization of the community Louvain
%algorithm. If not known, it could just be (1:length(conn_matrix))

%lambda:parameter for the community Louvain method (see BCT
%documentation of the community_louvain function)-usually this is 1

%all_weights: string taking either 'yes' or 'no' values. If 'yes' then the
%matrix is unthresholded (has positive and negative correlations) and the community louvain algorithm is executed with the
%'negative_asym' option to account for negative correlations. If 'no' then
%there are no negative correlations (e.g. your input matrix has only the
%strongest positive correlations) and the normal community louvain algorithm is
%executed.

%predefined_: string taking either 'yes' or 'no' values, if 'yes' then the
%algorithm expects a predefined partition vector (see next input parameter
%predefined_vector) of length length(conn_matrix) that contains integer
%values corresponding to the different communities (e.g. this could be
%coming from a predefined partition from another dataset-for example the
%Yeo 7-network partition). If 'no' then the community louvain algorithm is
%executed normally and optimally partitions are found from the data.

%Optional
%predefined_vector: if predefined_ == 'yes' then one needs to pass this
%vector as input (this will be a vector of integers showing the assignment of different regions to different communities).
%If predefined_=='no' this is ignored. In this case just set it to 0.

%Output
%Q: the modularity value
%most_common_module_mat: the optimal partition of the matrix

%email ipappas@usc.edu for more information
%% change accordingly
addpath(genpath('../99_BCT/'));
%%
num_iterations = 1000;
num_regions = length(conn_matrix);
if strcmp(predefined_,'no')
    %% there is no predefined partition so run iteratively Louvain
     for current_iteration = 1:num_iterations
            if strcmp(all_weights,'yes')
            [modules_list_mat(:,current_iteration),~] = community_louvain(conn_matrix,lambda,initial_vector,'negative_asym');
            else
            [modules_list_mat(:,current_iteration),~] = community_louvain(conn_matrix,lambda,initial_vector);
            end
            % give the modules a consistent order
            num_modules = max(modules_list_mat(:,current_iteration));
            first_module_appearance = nan(num_modules,1);
            modules_mat = nan(num_regions,num_modules);

            for current_module = 1:num_modules
                first_module_appearance(current_module) = min(find(modules_list_mat(:,current_iteration)==current_module));
                modules_mat(:,current_module) = (modules_list_mat(:,current_iteration)==current_module);

            end

            [~,module_order] = sort(first_module_appearance);
        modules_mat_reordered = modules_mat(:,module_order);
        modules_list_mat(:,current_iteration) = sum(modules_mat_reordered.*repmat(1:num_modules,num_regions,1),2);
        [modules_list, ~, modules_solution] = unique(modules_list_mat(:,:)','rows');
        most_common_modules = mode(modules_solution);
        most_common_module_mat(:,1) = modules_list(most_common_modules,:)';
     end   
    % now that the optimal partition has been calculated, return modularity value 
    if strcmp(all_weights,'yes')
            
              [~,Q]=community_louvain_calcmod_only(conn_matrix,lambda,most_common_module_mat,'negative_asym');
    else
             
              [~,Q]=community_louvain_calcmod_only(conn_matrix,lambda,most_common_module_mat);
    end
    %%
else
    %% there is a predefined partition so just calculate modularity based on that
    if strcmp(all_weights,'yes')
    [~,Q]=community_louvain_calcmod_only(conn_matrix,lambda,predefined_vector,'negative_asym');
    else
    [~,Q]=community_louvain_calcmod_only(conn_matrix,lambda,predefined_vector);
    end

    most_common_module_mat = predefined_vector;
end
 

end