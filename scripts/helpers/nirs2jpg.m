%% plot values on fnirs channels to jpg plot
% dependencies: BrainNetViewer, nirs2img, mni2cor, spm (https://github.com/spm/spm12)

% Author: Shan Gao


function nirs2jpg(outPath, outFilesName,...   % shared params
                  mni, value, doInterp, doXjview, bilateral,...   % nirs2img params
                  surfaceFilePath, configFilePath)   % BrainNet_MapCfg params
%%
% usage: nirs2jpg('../../../results/Yeo114Brainnetome8_undenoised/', 'pc1_loadings_fnirs',...   % shared params
%                 mni_keep_channels, data, 0, 0, 0,...   % nirs2img params
%                 'BrainMesh_ICBM152_smoothed.nv', 'brainnetviewer_config/config_neg10pos1_afnipos_maxvoxel.mat')   % BrainNet_MapCfg params
%%


%% nirs2img

nirs2img(strcat(outPath, outFilesName, '.img'),...    % imgFileName
         mni, ...     % mni
         value, ...      % value
         doInterp, doXjview, bilateral);      % doInterp, doXjview, bilateral


%% img2jpg

% add BrainNetViewer to path
% !will cause error if omitted even though BrainNetViewer_20191031 is under the same path!
addpath(genpath('../zz_help_scripts/BrainNetViewer_20191031/')); 

% % launch gui
% BrainNet;

% img2jpg
BrainNet_MapCfg(surfaceFilePath,...   % surface file
                strcat(outPath, outFilesName, '.img'),...   % mapping file
                configFilePath,...  % config file
                strcat(outPath, outFilesName, '.jpg'));   % output file


end


