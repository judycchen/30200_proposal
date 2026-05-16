%% i_viz_network_isfc.m
% Figure 4: ISFC heatmaps — observed BOLD (left) vs predicted BOLD (right)
% Rewritten in MATLAB using coolwarm colormap matching Ryleigh's approach.
%
% Also runs diagnostic: tests whether white gaps are caused by only having
% 42-channel fNIRS coverage ROIs in the 122-ROI matrix.
%
% Inputs:
%   data/d_apcr-model/apcr_isfc.mat          true_isfc_tril_RxR, pred_isfc_tril_RxR
%   data/c_fnirs-fmri-corr/channel_roi_matches.csv   matched_roi column
%   Yeo_17N_114_Brainnetome_subcortical_8.csv         ROI network labels
%
% Outputs (results/e_visualization/isfc/):
%   fig4_isfc_heatmap_true.jpg
%   fig4_isfc_heatmap_pred.jpg
%   fig4_isfc_heatmap_both.jpg
%
% Usage:  run from prediction-proj root, or set BASE below.

clear; clc;

% ── paths ─────────────────────────────────────────────────────────────────────
BASE      = "/project/ycleong/users/judycchen/prediction-proj";
MASKS_DIR = "/project/ycleong/users/judycchen/fnirs_fmri_models/data/sherlock" + ...
            "/fmri/masks/Yeo_17N_114_Brainnetome_subcortical_8";

ISFC_FILE = fullfile(BASE, "data/d_apcr-model/apcr_isfc.mat");
ROI_CSV   = fullfile(MASKS_DIR, "Yeo_17N_114_Brainnetome_subcortical_8.csv");
CHAN_CSV   = fullfile(BASE, "data/c_fnirs-fmri-corr/channel_roi_matches.csv");
OUT_DIR   = fullfile(BASE, "results/e_visualization/isfc");
if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end

% ── load ISFC matrices ────────────────────────────────────────────────────────
fprintf("Loading ISFC data...\n");
isfc_data         = load(ISFC_FILE);
true_tril_RxR     = isfc_data.true_isfc_tril_RxR;   % (122, 122) lower-tri filled
pred_tril_RxR     = isfc_data.pred_isfc_tril_RxR;   % (122, 122) lower-tri filled
isfc_r            = isfc_data.isfc_similarity;
isfc_p            = isfc_data.isfc_pval;
fprintf("ISFC matrix similarity: r=%.3f, p=%.3f\n", isfc_r, isfc_p);
fprintf("NaN in true lower-tri : %d\n", sum(isnan(true_tril_RxR(:))));
fprintf("NaN in pred lower-tri : %d\n", sum(isnan(pred_tril_RxR(:))));

% ── load ROI labels and assign networks ───────────────────────────────────────
fprintf("\nLoading ROI labels...\n");
% CSV columns: roi_id, full_name, short_name, R, G, B, network_code
% (no header row)
opts = detectImportOptions(ROI_CSV, "NumHeaderLines", 0);
opts.VariableNames = {'roi_id','full_name','short_name','R','G','B','net_code'};
opts = setvartype(opts, {'roi_id'}, 'double');
opts = setvartype(opts, {'full_name','short_name'}, 'char');
roi_tbl = readtable(ROI_CSV, opts);

% Assign network label for each ROI
NROI = height(roi_tbl);
networks = strings(NROI, 1);
for k = 1:NROI
    sn = string(roi_tbl.short_name{k});
    if     contains(sn, "Subcortical"); networks(k) = "SUBC";
    elseif contains(sn, "Vis");         networks(k) = "VIS";
    elseif contains(sn, "SomMot");      networks(k) = "SM";
    elseif contains(sn, "DorsAttn");    networks(k) = "DAN";
    elseif contains(sn, "SalVentAttn"); networks(k) = "VAN";
    elseif contains(sn, "Limbic");      networks(k) = "LIMB";
    elseif contains(sn, "Cont");        networks(k) = "CONT";
    elseif contains(sn, "Default");     networks(k) = "DMN";
    elseif contains(sn, "TempPar");     networks(k) = "DMN";
    else;                               networks(k) = "OTHER";
    end
end
roi_tbl.network = networks;

% Sort by network order
NETWORK_ORDER = ["VIS", "SM", "DAN", "VAN", "LIMB", "CONT", "DMN", "SUBC"];
[~, net_rank] = ismember(networks, NETWORK_ORDER);
net_rank(net_rank == 0) = numel(NETWORK_ORDER) + 1;   % push OTHER to end
[~, roi_sort_idx] = sort(net_rank);                    % sorted ROI indices (1-based)
roi_tbl_sorted = roi_tbl(roi_sort_idx, :);

fprintf("ROIs per network (sorted order):\n");
for n = 1:numel(NETWORK_ORDER)
    mask = roi_tbl_sorted.network == NETWORK_ORDER(n);
    fprintf("  %s: %d\n", NETWORK_ORDER(n), sum(mask));
end

% Network boundary positions (for drawing grid lines)
net_sizes = zeros(1, numel(NETWORK_ORDER));
for n = 1:numel(NETWORK_ORDER)
    net_sizes(n) = sum(roi_tbl_sorted.network == NETWORK_ORDER(n));
end
net_bounds    = cumsum(net_sizes);            % last index of each network block
net_bounds_ex = [0, net_bounds];
net_midpoints = (net_bounds_ex(1:end-1) + net_bounds_ex(2:end)) / 2 + 0.5;

% ── symmetrize ISFC matrices ──────────────────────────────────────────────────
% tri_average stores values in lower triangle, NaN in upper.
% Symmetrize by copying lower→upper, then set diagonal to 0.

function mat_sym = symmetrize_tril(mat)
    mat_sym = mat;
    [lo_i, lo_j] = find(tril(true(size(mat)), -1));   % strictly lower-tri indices
    for k = 1:numel(lo_i)
        val = mat(lo_i(k), lo_j(k));
        if ~isnan(val)
            mat_sym(lo_j(k), lo_i(k)) = val;   % mirror to upper
        else
            mat_sym(lo_i(k), lo_j(k)) = 0;     % fill NaN with 0 in lower
            mat_sym(lo_j(k), lo_i(k)) = 0;     % fill NaN with 0 in upper
        end
    end
    mat_sym(1:size(mat,1)+1:end) = 0;   % diagonal = 0
end

true_sym = symmetrize_tril(true_tril_RxR);
pred_sym = symmetrize_tril(pred_tril_RxR);

% Reorder by network
true_plot = true_sym(roi_sort_idx, roi_sort_idx);
pred_plot = pred_sym(roi_sort_idx, roi_sort_idx);

% ── coolwarm colormap ─────────────────────────────────────────────────────────
% Matches seaborn coolwarm: blue → white → red
% Control points from seaborn source (approximately):
%   blue  [0.017, 0.443, 0.690]
%   white [1.000, 1.000, 1.000]
%   red   [0.695, 0.097, 0.074]

function cmap = coolwarm_cmap(n)
    if nargin < 1; n = 256; end
    ctrl_pts = [0.017, 0.443, 0.690;
                1.000, 1.000, 1.000;
                0.695, 0.097, 0.074];
    x  = linspace(0, 1, n)';
    xk = [0; 0.5; 1];
    cmap = interp1(xk, ctrl_pts, x, 'linear');
    cmap = max(0, min(1, cmap));
end

cmap = coolwarm_cmap(256);

% ── helper: draw one ISFC panel ───────────────────────────────────────────────

function draw_isfc_panel(ax, mat, title_str, net_sizes, net_midpoints, NETWORK_ORDER, cmap)
    imagesc(ax, mat, [-0.3, 0.3]);
    colormap(ax, cmap);
    axis(ax, 'square');

    % network boundary lines
    net_bounds = cumsum(net_sizes);
    for b = net_bounds(1:end-1)
        xline(ax, b + 0.5, 'k-', 'LineWidth', 0.8);
        yline(ax, b + 0.5, 'k-', 'LineWidth', 0.8);
    end

    % tick labels at network midpoints
    set(ax, 'XTick', net_midpoints, 'XTickLabel', NETWORK_ORDER, ...
            'XTickLabelRotation', 0, 'FontSize', 9, 'FontWeight', 'bold');
    set(ax, 'YTick', net_midpoints, 'YTickLabel', NETWORK_ORDER, ...
            'FontSize', 9, 'FontWeight', 'bold');

    cb = colorbar(ax);
    cb.Label.String = 'r';
    cb.FontSize = 8;
    cb.Ticks = -0.3:0.1:0.3;

    title(ax, title_str, 'FontSize', 13, 'FontWeight', 'bold');
end

% ── FIGURE 4: side-by-side ────────────────────────────────────────────────────
fprintf("\nPlotting Figure 4 (both panels)...\n");

fig4 = figure('Units', 'pixels', 'Position', [100 100 1200 520], 'Visible', 'off');

ax1 = subplot(1, 2, 1);
draw_isfc_panel(ax1, true_plot, 'Observed BOLD ISFC', ...
    net_sizes, net_midpoints, NETWORK_ORDER, cmap);

ax2 = subplot(1, 2, 2);
draw_isfc_panel(ax2, pred_plot, 'Predicted BOLD ISFC', ...
    net_sizes, net_midpoints, NETWORK_ORDER, cmap);

annotation(fig4, 'textbox', [0.1 0.01 0.8 0.04], ...
    'String', sprintf('Matrix similarity: Pearson r = %.3f, p = %.3f', isfc_r, isfc_p), ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontSize', 11, 'FontAngle', 'italic');

sgtitle('NNW fNIRS→fMRI aPCR — ISFC comparison', 'FontSize', 12, 'FontWeight', 'bold');

out_both = fullfile(OUT_DIR, "fig4_isfc_heatmap_both.jpg");
print(fig4, out_both, '-djpeg', '-r150');
fprintf("Saved: %s\n", out_both);
close(fig4);

% ── individual panels ─────────────────────────────────────────────────────────
for pair = {{'true', true_plot, 'Observed BOLD ISFC'}, ...
             {'pred', pred_plot, 'Predicted BOLD ISFC'}}
    tag   = pair{1}{1};
    mat   = pair{1}{2};
    ttl   = pair{1}{3};
    fig_s = figure('Units', 'pixels', 'Position', [100 100 600 520], 'Visible', 'off');
    ax_s  = axes(fig_s);
    draw_isfc_panel(ax_s, mat, ttl, net_sizes, net_midpoints, NETWORK_ORDER, cmap);
    out_s = fullfile(OUT_DIR, sprintf("fig4_isfc_heatmap_%s.jpg", tag));
    print(fig_s, out_s, '-djpeg', '-r150');
    fprintf("Saved: %s\n", out_s);
    close(fig_s);
end

% ══════════════════════════════════════════════════════════════════════════════
%% DIAGNOSTIC: 42-channel coverage hypothesis
%  Hypothesis: white gaps arise because only ~42 of 122 ROIs are spatially
%  covered by fNIRS channels, so the predicted ISFC for non-covered ROIs
%  may be near-zero or NaN, creating white stripes/gaps in the heatmap.
% ══════════════════════════════════════════════════════════════════════════════

fprintf("\n%s\n", repmat("=", 1, 60));
fprintf("DIAGNOSTIC: 42-channel fNIRS coverage hypothesis\n");
fprintf("%s\n", repmat("=", 1, 60));

% Load channel→ROI matching to identify covered ROI IDs
chan_tbl     = readtable(CHAN_CSV);
covered_rois = unique(chan_tbl.matched_roi);   % ROI IDs (1-based) covered by fNIRS
n_covered    = numel(covered_rois);
covered_mask = false(122, 1);
covered_mask(covered_rois) = true;
fprintf("fNIRS-covered ROIs : %d / 122\n", n_covered);
fprintf("Uncovered ROIs     : %d / 122\n", 122 - n_covered);

% ── Check 1: NaN rates in original (pre-symmetrization) lower triangle ────────
fprintf("\n--- Check 1: NaN in original lower triangle (per-ROI row) ---\n");
% For each ROI row, count NaN in its lower-triangle entries
nan_per_row_true = zeros(122, 1);
nan_per_row_pred = zeros(122, 1);
n_entries = zeros(122, 1);   % number of lower-tri entries per row

for r = 1:122
    lo_entries_true = true_tril_RxR(r, 1:r-1);   % columns below diagonal
    lo_entries_pred = pred_tril_RxR(r, 1:r-1);
    n_entries(r)    = numel(lo_entries_true);
    if n_entries(r) > 0
        nan_per_row_true(r) = sum(isnan(lo_entries_true)) / n_entries(r);
        nan_per_row_pred(r) = sum(isnan(lo_entries_pred)) / n_entries(r);
    else
        nan_per_row_true(r) = NaN;   % row 1 has no lower-tri entries
        nan_per_row_pred(r) = NaN;
    end
end

cov_true_nan  = nanmean(nan_per_row_true(covered_mask));
uncov_true_nan = nanmean(nan_per_row_true(~covered_mask));
cov_pred_nan  = nanmean(nan_per_row_pred(covered_mask));
uncov_pred_nan = nanmean(nan_per_row_pred(~covered_mask));

fprintf("  True ISFC NaN rate  — covered ROIs: %.1f%%,  uncovered: %.1f%%\n", ...
        cov_true_nan*100, uncov_true_nan*100);
fprintf("  Pred ISFC NaN rate  — covered ROIs: %.1f%%,  uncovered: %.1f%%\n", ...
        cov_pred_nan*100, uncov_pred_nan*100);

% ── Check 2: Mean absolute ISFC per ROI row (symmetrized, no diagonal) ────────
fprintf("\n--- Check 2: Mean |ISFC| per ROI row (symmetrized, diagonal excluded) ---\n");

abs_true_sym = abs(true_sym);   abs_true_sym(1:123:end) = NaN;   % exclude diagonal
abs_pred_sym = abs(pred_sym);   abs_pred_sym(1:123:end) = NaN;

mean_abs_per_row_true = nanmean(abs_true_sym, 2);
mean_abs_per_row_pred = nanmean(abs_pred_sym, 2);

fprintf("  True |ISFC| per row — covered ROIs: %.4f ± %.4f\n", ...
        nanmean(mean_abs_per_row_true(covered_mask)), ...
        nanstd(mean_abs_per_row_true(covered_mask)));
fprintf("  True |ISFC| per row — uncovered    : %.4f ± %.4f\n", ...
        nanmean(mean_abs_per_row_true(~covered_mask)), ...
        nanstd(mean_abs_per_row_true(~covered_mask)));
fprintf("  Pred |ISFC| per row — covered ROIs: %.4f ± %.4f\n", ...
        nanmean(mean_abs_per_row_pred(covered_mask)), ...
        nanstd(mean_abs_per_row_pred(covered_mask)));
fprintf("  Pred |ISFC| per row — uncovered    : %.4f ± %.4f\n", ...
        nanmean(mean_abs_per_row_pred(~covered_mask)), ...
        nanstd(mean_abs_per_row_pred(~covered_mask)));

% ── Check 3: Visual map — which ROIs are covered and where in the sorted order
fprintf("\n--- Check 3: Coverage location in network-sorted order ---\n");
covered_sorted_pos = find(covered_mask(roi_sort_idx));
fprintf("  fNIRS-covered ROIs appear at sorted positions: ");
fprintf("%d ", covered_sorted_pos);
fprintf("\n");

% Count covered ROIs per network block
net_bounds_arr = [0, cumsum(net_sizes)];
fprintf("  Coverage per network:\n");
for n = 1:numel(NETWORK_ORDER)
    start_pos = net_bounds_arr(n) + 1;
    end_pos   = net_bounds_arr(n+1);
    in_block  = sum(covered_sorted_pos >= start_pos & covered_sorted_pos <= end_pos);
    block_sz  = net_sizes(n);
    fprintf("    %-6s: %d / %d ROIs covered\n", NETWORK_ORDER(n), in_block, block_sz);
end

% ── Check 4: Global NaN count in symmetrized matrices ─────────────────────────
fprintf("\n--- Check 4: Total NaN in symmetrized matrices ---\n");
fprintf("  true_sym NaN count : %d / %d (%.1f%%)\n", ...
        sum(isnan(true_sym(:))), numel(true_sym), mean(isnan(true_sym(:)))*100);
fprintf("  pred_sym NaN count : %d / %d (%.1f%%)\n", ...
        sum(isnan(pred_sym(:))), numel(pred_sym), mean(isnan(pred_sym(:)))*100);

% ── Verdict ───────────────────────────────────────────────────────────────────
fprintf("\n--- VERDICT ---\n");

total_nan_true = sum(isnan(true_sym(:)));
total_nan_pred = sum(isnan(pred_sym(:)));

if total_nan_true == 0 && total_nan_pred == 0
    fprintf("  No NaN values in symmetrized matrices.\n");
    fprintf("  White gaps are NOT caused by missing data (NaN).\n");
    fprintf("  Possible causes: values genuinely near 0 (white in coolwarm at center).\n");
else
    fprintf("  NaN values present — check which ROIs/networks are affected.\n");
end

% Check coverage effect on predicted ISFC magnitude
pred_cov_mean   = nanmean(mean_abs_per_row_pred(covered_mask));
pred_uncov_mean = nanmean(mean_abs_per_row_pred(~covered_mask));
if pred_cov_mean > pred_uncov_mean * 1.5
    fprintf("  Coverage effect DETECTED in predicted ISFC:\n");
    fprintf("    Covered ROIs have %.1fx higher |ISFC| than uncovered.\n", ...
            pred_cov_mean / pred_uncov_mean);
    fprintf("    This SUPPORTS the 42-channel coverage hypothesis.\n");
    fprintf("    Uncovered ROIs predicted fMRI is weaker → near-zero ISFC.\n");
else
    fprintf("  Coverage effect NOT detected (covered vs uncovered |ISFC| similar).\n");
    fprintf("    42-channel coverage is NOT the main cause of white gaps.\n");
end

fprintf("\nDone: i_viz_network_isfc.m\n");
fprintf("Outputs in: %s\n", OUT_DIR);
