%% d_mni_brain_viz.m
% Phase 3 visualization: render fNIRS-fMRI correlation r values onto
% brain surface at each fNIRS channel MNI location.
% Produces Figure 2b equivalent from Gao et al. 2025.
%
% Inputs:
%   - data/c_fnirs-fmri-corr/fnirs_fmri_corr.mat  (from c_fnirs_fmri_corr.py)
%   - Standard_Channels.txt                        (MNI coordinates)
%
% Outputs:
%   - data/c_fnirs-fmri-corr/figures/fnirs_fmri_corr.jpg
%   - data/c_fnirs-fmri-corr/figures/fnirs_fmri_corr_pmasked.jpg
%     (same figure but with non-significant channels zeroed out)
%
% Dependencies (all in scripts/helpers/):
%   - nirs2jpg.m
%   - spm12/
%   - BrainNetViewer_20191031/
%   - config_pos0.4_afnipos_interp_lateralmedial.mat

clear all; clc;

%% ── paths ────────────────────────────────────────────────────────────────────

BASE        = '/project/ycleong/users/judycchen/prediction-proj';
HELPERS     = fullfile(BASE, 'scripts/helpers');
IN_FILE     = fullfile(BASE, 'data/c_fnirs-fmri-corr/fnirs_fmri_corr.mat');
STDCHAN     = '/project/ycleong/datasets/CogTasks-NNW_judy/Montage/Standard_Channels.txt';
OUT_DIR     = fullfile(BASE, 'data/c_fnirs-fmri-corr/figures');

if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

%% ── toolbox paths ────────────────────────────────────────────────────────────

addpath(HELPERS);
addpath(genpath(fullfile(HELPERS, 'spm12')));
addpath(genpath(fullfile(HELPERS, 'BrainNetViewer_20191031')));

%% ── params ───────────────────────────────────────────────────────────────────

SHORT_CH_DIST_MM = 15.0;

SURFACE_FILE = 'BrainMesh_ICBM152_smoothed.nv';
CONFIG_FILE  = fullfile(HELPERS, 'config_pos0.4_afnipos_interp_lateralmedial.mat');

%% ── load MNI coordinates for 42 standard channels ───────────────────────────

fprintf('Loading Standard_Channels.txt...\n');

chan_data      = readmatrix(STDCHAN);
standard_mask  = chan_data(:, 7) > SHORT_CH_DIST_MM;
chan_standard  = chan_data(standard_mask, :);
mni_coords     = chan_standard(:, 8:10);   % (42 x 3)

fprintf('Standard channels: %d (expected 42)\n', size(mni_coords, 1));
if size(mni_coords, 1) ~= 42
    warning('Expected 42 standard channels, got %d. Check distance threshold.', ...
            size(mni_coords, 1));
end

%% ── load correlation results ─────────────────────────────────────────────────

fprintf('Loading correlation results from: %s\n', IN_FILE);
if ~exist(IN_FILE, 'file')
    error('Input file not found: %s\nRun c_fnirs_fmri_corr.py first.', IN_FILE);
end

d = load(IN_FILE);

grp_grp_corr_Rx1         = squeeze(d.grp_grp_corr_Rx1);
grp_grp_corr_pmasked_Rx1 = squeeze(d.grp_grp_corr_pmasked_Rx1);

fprintf('r values loaded: %d channels\n', length(grp_grp_corr_Rx1));
fprintf('Significant channels: %d\n', sum(d.grp_grp_corr_sigmask_Rx1));
fprintf('Mean r: %.4f | Median r: %.4f\n', ...
        mean(grp_grp_corr_Rx1, 'omitnan'), median(grp_grp_corr_Rx1, 'omitnan'));

%% ── build output path string (char array with trailing separator) ────────────
% SPM12 requires char arrays throughout — string objects cause
% "SWITCH expression must be scalar or character vector" errors

out_path    = [char(OUT_DIR) filesep];
surf_file   = SURFACE_FILE;          % already a char literal
config_file = char(CONFIG_FILE);

%% ── Figure 1: all channels (unthresholded) ───────────────────────────────────

fprintf('\nGenerating brain map — all channels (unthresholded)...\n');

data_plot = grp_grp_corr_Rx1;
data_plot(isnan(data_plot)) = 0;

nirs2jpg(out_path, 'fnirs_fmri_corr', ...
         mni_coords, data_plot, 1, 0, 0, ...
         surf_file, config_file);

hdr1 = fullfile(OUT_DIR, 'fnirs_fmri_corr.hdr');
img1 = fullfile(OUT_DIR, 'fnirs_fmri_corr.img');
if exist(hdr1, 'file'); delete(hdr1); end
if exist(img1, 'file'); delete(img1); end

fprintf('Saved: %s\n', fullfile(OUT_DIR, 'fnirs_fmri_corr.jpg'));

%% ── Figure 2: FDR-thresholded (non-significant channels zeroed) ──────────────

fprintf('\nGenerating brain map — FDR-thresholded...\n');

data_pmasked = grp_grp_corr_pmasked_Rx1;
data_pmasked(isnan(data_pmasked)) = 0;

nirs2jpg(out_path, 'fnirs_fmri_corr_pmasked', ...
         mni_coords, data_pmasked, 1, 0, 0, ...
         surf_file, config_file);

hdr2 = fullfile(OUT_DIR, 'fnirs_fmri_corr_pmasked.hdr');
img2 = fullfile(OUT_DIR, 'fnirs_fmri_corr_pmasked.img');
if exist(hdr2, 'file'); delete(hdr2); end
if exist(img2, 'file'); delete(img2); end

fprintf('Saved: %s\n', fullfile(OUT_DIR, 'fnirs_fmri_corr_pmasked.jpg'));

fprintf('\nDone: d_mni_brain_viz.m\n');
fprintf('Figures saved to: %s\n', OUT_DIR);