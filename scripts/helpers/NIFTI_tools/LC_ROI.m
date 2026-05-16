clear all; close all; clc

path_NM = strcat('/Users/Jadyn/Desktop/1_bids/sub-1001/ses-1/anat/');

fileNM = strcat(path_NM,'MTC_brain_thr0.04.nii');

addpath(genpath('../9_help_scripts'));
addpath(genpath('../9_NIFTI_tools'));

cd(path_NM);
unix('gunzip *.nii.gz');
NM_s=load_untouch_nii(fileNM);
NM = NM_s.img;

% Number of adjacent voxels
NN = 10; 

%% Left
fileROI = strcat(path_NM,'LC_bin_L.nii');

ROI_s=load_untouch_nii(fileROI);
ROI = ROI_s.img;

LC = zeros(size(ROI));
LCmax = zeros(size(ROI));

for ii = 1:size(ROI,3)
    mask = double(NM(:,:,ii)) .* double(ROI(:,:,ii));
    if (find(mask))
        LC_i = zeros(size(mask));
        LC_max = zeros(size(mask));
        
        [sortedValues,sortIndex] = sort(mask(:),'descend');
        
        LC_max(sortIndex(1)) = 1;
        LC_i = LC_max;
        
        nn = 1;
        xx = 2;
        while nn<NN
            LC_temp = LC_i;
            LC_temp(sortIndex(xx)) = 1;
            [L,n] = bwlabel(LC_temp);
            if n == 1
                LC_i(sortIndex(xx)) = 1;
                nn = nn + 1;
            end
            xx = xx + 1;
        end
        
        LC(:,:,ii) = LC_i;
        LCmax(:,:,ii) = LC_max;
    end
end

LC_s = ROI_s;
LC_s.img = LC;
save_untouch_nii(LC_s,strcat(path_NM,'LC_L.nii'));
gzip('LC_L.nii');

%LC_max = ROI_s;
%LC_max.img = LCmax;
%save_untouch_nii(LC_max,strcat(path_NM,'mcn_LC_max_L.nii'));

%% Right

fileROI = strcat(path_NM,'LC_bin_R.nii');

ROI_s=load_untouch_nii(fileROI);
ROI = ROI_s.img;

LC = zeros(size(ROI));
LCmax = zeros(size(ROI));

for ii = 1:size(ROI,3)
    
    mask = double(NM(:,:,ii)) .* double(ROI(:,:,ii));
    if (find(mask))
        LC_i = zeros(size(mask));
        LC_max = zeros(size(mask));
        
        [sortedValues,sortIndex] = sort(mask(:),'descend');
        
        LC_max(sortIndex(1)) = 1;
        LC_i = LC_max;
        
        nn = 1;
        xx = 2;
        while nn<NN
            LC_temp = LC_i;
            LC_temp(sortIndex(xx)) = 1;
            [L,n] = bwlabel(LC_temp);
            if n == 1
                LC_i(sortIndex(xx)) = 1;
                nn = nn + 1;
            end
            xx = xx + 1;
        end
        
        LC(:,:,ii) = LC_i;
        LCmax(:,:,ii) = LC_max;
    end
end

LC_s = ROI_s;
LC_s.img = LC;
save_untouch_nii(LC_s,strcat(path_NM,'LC_R.nii'));
gzip('LC_R.nii');

%LC_max = ROI_s;
%LC_max.img = LCmax;
%save_untouch_nii(LC_max,strcat(path_NM,'LC_9R_max.nii'));
