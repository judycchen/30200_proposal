%% b2_exclude-fnirs.m
% fNIRS preprocessing pipeline — Part 2 of 2
% Sections 7–8: manual subject exclusion → post-exclusion ISC
%
% Before running:
%   1. Review ISC table in b1 job log
%   2. Update exclS_NNW below with indices of subjects to exclude
%   3. Submit submit_b2_exclude-fnirs.sbatch
%
% Reference file for interactive use: b_preproc-fnirs.m (all sections combined)

clear all; clc;

% ── task selection ────────────────────────────────────────────────────
tasknum    = 6;
task_names = ["Nback", "CPT1", "CPTMEM1", "CPT2", "CPTMEM2", "NNW"];
task       = task_names(tasknum);

% ── manual subject exclusion ──────────────────────────────────────────
% *** UPDATE THESE after reviewing ISC output from b1 job ***
% Index into S dimension of zhbo_TxRxS (1-based).
% The subject index mapping is printed at the start of Section 7 below.
exclS_Nback   = [17];
exclS_CPT1    = [16];
exclS_CPTMEM1 = [16];
exclS_CPT2    = [17];
exclS_CPTMEM2 = [1, 2, 12, 14, 17, 18];   % ISC < -0.02: p01(-0.042), p02(-0.077), p15(-0.036), p17(-0.029), p20(-0.044), p21(-0.023)
exclS_NNW     = [13, 23];   % ISC < 0: p15(-0.0112), p25(-0.0171)
exclS_all = {exclS_Nback, exclS_CPT1, exclS_CPTMEM1, exclS_CPT2, exclS_CPTMEM2, exclS_NNW};

% ── paths ─────────────────────────────────────────────────────────────
dir_helpers     = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers";
dir_base_out    = "/project/ycleong/users/judycchen/prediction-proj/data/b_fnirs-preproc/";
dir_4_resampled = dir_base_out + "4_resampled/";
dir_5_excluded  = dir_base_out + "5_excluded/";

if ~exist(dir_5_excluded, "dir"); mkdir(dir_5_excluded); end

restoredefaultpath;
addpath(dir_helpers);   % for isc.m

outfile_resamp = fullfile(dir_4_resampled, sprintf("all_%s_zhbo_TxRxS.mat", task));
outfile_excl   = fullfile(dir_5_excluded,  sprintf("all_%s_zhbo_TxRxS.mat", task));

fprintf("Task: %s\n", task);

%% ── SECTION 7: MANUAL SUBJECT EXCLUSION ─────────────────────────────

if ~exist(outfile_resamp, "file")
    error("Input file not found: %s\nRun b1_preproc-fnirs.m first.", outfile_resamp);
end

load(outfile_resamp, "zhbo_TxRxS", "subj_ids_included");

fprintf("Subject index mapping for %s:\n", task);
for s = 1:length(subj_ids_included)
    fprintf("  index %d = %s\n", s, subj_ids_included{s});
end

exclS      = exclS_all{tasknum};
nS         = size(zhbo_TxRxS, 3);
keepS_mask = true(1, nS);
keepS_mask(exclS) = false;
keepS = find(keepS_mask);

fprintf("\n%s: nS=%d, excluding [%s], keeping %d subjects\n", ...
        task, nS, num2str(exclS), sum(keepS_mask));

subj_ids_included = subj_ids_included(keepS_mask);
fprintf("Kept: %s\n", strjoin(subj_ids_included, ", "));

% append exclusion info back to resampled file for record
save(outfile_resamp, "exclS", "keepS", "keepS_mask", "-append");

% save post-exclusion file
zhbo_TxRxS = zhbo_TxRxS(:, :, keepS_mask);
save(outfile_excl, "zhbo_TxRxS", "exclS", "keepS", "subj_ids_included");

fprintf("Output: %dx%dx%d (T x R x S)\nSaved: %s\n", ...
        size(zhbo_TxRxS,1), size(zhbo_TxRxS,2), size(zhbo_TxRxS,3), outfile_excl);
fprintf("Done: Section 7 [task=%s]\n", task);

%% ── SECTION 8: ISC ON POST-EXCLUSION DATA ────────────────────────────

load(outfile_excl, "zhbo_TxRxS", "subj_ids_included");
fprintf("\nData: %dx%dx%d (T x R x S)\n", size(zhbo_TxRxS,1), size(zhbo_TxRxS,2), size(zhbo_TxRxS,3));

isc_SxR = isc(zhbo_TxRxS, 0);
isc_Sx1 = tanh(nanmean(atanh(isc_SxR), 2));
isc_Rx1 = tanh(nanmean(atanh(isc_SxR), 1));

fprintf("\nPost-exclusion subject ISC:\n");
for s = 1:length(subj_ids_included)
    fprintf("  %s: %.4f\n", subj_ids_included{s}, isc_Sx1(s));
end
fprintf("Grand mean ISC: %.4f\n", mean(isc_Sx1, "omitnan"));

save(outfile_excl, "isc_SxR", "isc_Sx1", "isc_Rx1", "-append");
fprintf("ISC results saved to: %s\n", outfile_excl);
fprintf("Done: Section 8 [task=%s]\n", task);
