%% b1_preproc-fnirs.m
% fNIRS preprocessing pipeline — continuous-stimuli version
% Sections 1–6: load+extract+inpaint → OD+filter+SCR → OD→Hb+QA → resample → ISC
%
% Based on: /project/ycleong/users/judycchen/scripts_adpt-fr-ryleigh/1_1_preproc_fnirs_judy/
% Replaces trial-based AnalyzIR pipeline with Homer2 pipeline suited for
% continuous natural stimuli (movies, naturalistic tasks).
%
% After this script completes:
%   1. Read the ISC table printed at the end of the .out log
%   2. Update exclS_NNW in b2_exclude-fnirs.m
%   3. Submit submit_b2_exclude-fnirs.sbatch

clear all; clc;

% ── task selection ────────────────────────────────────────────────────
tasknum     = 6;
task_names  = ["Nback", "CPT1", "CPTMEM1", "CPT2", "CPTMEM2", "NNW"];
task_suffix = ["001",   "002",  "003",      "004",  "005",     "006"];
task        = task_names(tasknum);
suffix      = task_suffix(tasknum);

% ── preprocessing params ──────────────────────────────────────────────
lag              = 0;
min_duration_sec = 300;
target_duration_sec = 889;

% NNW (tasknum = 6)
% target_duration_sec = 889;

% N-back (tasknum = 1)
% target_duration_sec = 530;

% CPT1 (tasknum = 2)
% target_duration_sec = 670;

% CPTMEM1 (tasknum = 3)
% target_duration_sec = 275;

% CPT2 (tasknum = 4)
% target_duration_sec = 660;

% CPTMEM2 (tasknum = 5)
% target_duration_sec = 250;

% ── bandpass filter params ─────────────────────────────────────────────
BandpassFilt_l = 0.005;   % Hz highpass
BandpassFilt_h = 0.5;     % Hz lowpass

% ── motion correction params ───────────────────────────────────────────
MotionArtifact_tMotion     = 1;
MotionArtifact_tMask       = 1;
MotionArtifact_STDEVthresh = 5;
MotionArtifact_AMPthresh   = 2;
n_pca                      = 0.8;   % PCA threshold

% ── bad channel removal params (inpainting) ────────────────────────────
satlength       = 2;
QCoDthresh_base = 0.6;   % QCoDthresh = 0.6 - 0.03*samprate (computed per subject)

% ── QA params ─────────────────────────────────────────────────────────
qamethod = 'corr';
thresh   = 0.1;

% ── PPF for OD → Hb ───────────────────────────────────────────────────
ppf = [6 6];

% ── short channel indices ──────────────────────────────────────────────
% Identified from Standard_Channels.txt (column 7, distance < 15 mm)
short_ch_idx    = [5, 12, 21, 24, 28, 32, 39, 50];
all_ch_idx      = 1:50;
standard_ch_idx = setdiff(all_ch_idx, short_ch_idx);   % 42 channels
n_standard_ch   = length(standard_ch_idx);

% MNI coordinates for all 50 channels (from Standard_Channels.txt, cols 8–10)
all_ch_mni = [
    -48.397,  53.595,  -3.640;   % ch 1
    -36.101,  64.716, -11.741;   % ch 2
    -49.133,  46.016,  20.011;   % ch 3
    -32.292,  47.635,  39.196;   % ch 4
    -40.983,  49.860,  28.800;   % ch 5  SHORT
    -43.323,  57.220,  11.174;   % ch 6
    -25.979,  69.913,   1.117;   % ch 7
    -25.942,  59.948,  30.375;   % ch 8
    -15.289,  69.407,  18.903;   % ch 9
    -67.051, -55.811,  -8.560;   % ch 10
    -70.708, -44.929,   4.667;   % ch 11
    -71.356, -39.855, -10.292;   % ch 12 SHORT
    -60.527, -68.587,   7.722;   % ch 13
    -64.798, -57.811,  25.338;   % ch 14
    -53.237, -72.288,  37.074;   % ch 15
    -66.061, -44.683,  41.730;   % ch 16
    -54.221, -59.265,  51.918;   % ch 17
    -42.401, -47.736,  66.231;   % ch 18
    -39.772, -72.949,  53.073;   % ch 19
    -29.566, -59.889,  69.972;   % ch 20
    -28.115, -69.633,  62.610;   % ch 21 SHORT
     30.186, -58.843,  70.135;   % ch 22
     39.571, -72.500,  53.447;   % ch 23
     23.049, -74.015,  62.258;   % ch 24 SHORT
    -13.398,  72.370,  -7.675;   % ch 25
      1.059,  68.377,   8.774;   % ch 26
     13.612,  72.444,  -7.362;   % ch 27
      8.160,  72.446,  -6.622;   % ch 28 SHORT
    -10.534,  50.217,  49.291;   % ch 29
      1.593,  57.715,  37.889;   % ch 30
      9.969,  50.953,  48.589;   % ch 31
      6.957,  50.436,  50.106;   % ch 32 SHORT
     15.082,  69.478,  18.810;   % ch 33
     25.958,  59.688,  30.706;   % ch 34
     25.892,  70.062,   1.215;   % ch 35
     42.576,  57.797,  11.904;   % ch 36
     32.230,  48.803,  38.145;   % ch 37
     48.247,  46.798,  20.402;   % ch 38
     43.966,  45.285,  29.852;   % ch 39 SHORT
     36.290,  64.907, -10.957;   % ch 40
     48.699,  53.534,  -2.917;   % ch 41
     41.616, -46.264,  66.615;   % ch 42
     54.451, -58.185,  52.283;   % ch 43
     66.074, -44.187,  42.081;   % ch 44
     53.553, -71.577,  37.290;   % ch 45
     65.312, -56.526,  25.661;   % ch 46
     61.114, -67.834,   8.169;   % ch 47
     70.328, -44.159,   7.384;   % ch 48
     67.228, -55.810,  -7.129;   % ch 49
     70.285, -45.167, -10.204;   % ch 50 SHORT
];

% For each standard channel, pre-compute its nearest short channel by
% Euclidean distance in MNI space (fixed mapping, same for all subjects)
short_mni    = all_ch_mni(short_ch_idx, :);
standard_mni = all_ch_mni(standard_ch_idx, :);
nearest_short_pos = zeros(n_standard_ch, 1);
for i = 1:n_standard_ch
    diffs = short_mni - standard_mni(i, :);
    dists = sqrt(sum(diffs .^ 2, 2));
    [~, nearest_short_pos(i)] = min(dists);
end
nearest_short_ch = short_ch_idx(nearest_short_pos);

% ── paths ─────────────────────────────────────────────────────────────
dir_raw_base = "/project/ycleong/datasets/CogTasks-NNW_judy/";
dir_helpers  = "/project/ycleong/users/judycchen/prediction-proj/scripts/helpers";
dir_base_out = "/project/ycleong/users/judycchen/prediction-proj/data/b_fnirs-preproc/";

dir_1_extracted = dir_base_out + "1_extracted/";
dir_2_filtered  = dir_base_out + "2_filtered_od/";
dir_3_QAed      = dir_base_out + "3_QAed_hb/";
dir_4_resampled = dir_base_out + "4_resampled/";

for d = {dir_1_extracted, dir_2_filtered, dir_3_QAed, dir_4_resampled}
    if ~exist(d{1}, "dir"); mkdir(d{1}); end
end

restoredefaultpath;
addpath(genpath(fullfile(dir_helpers, "homer2")));
addpath(genpath(fullfile(dir_helpers, "shannon_scripts")));
addpath(dir_helpers);

fprintf("Params loaded. Task: %s (folder suffix: _%s)\n", task, suffix);

%% ── SECTION 2: LOAD RAW DATA, EXTRACT SEGMENT, INPAINT BAD CHANNELS ──

subj_dirs = dir(fullfile(dir_raw_base, "p*_fnirs_cogTasks"));
subj_dirs = subj_dirs([subj_dirs.isdir]);
fprintf("Found %d subject folders.\n", length(subj_dirs));

for i = 1:length(subj_dirs)

    subj_folder = subj_dirs(i).name;
    subj_id     = regexp(subj_folder, "p\d+", "match", "once");
    subj_path   = fullfile(dir_raw_base, subj_folder);

    nirx_path = find_nirx_path(subj_path, suffix);
    if isempty(nirx_path)
        fprintf("  %s: no task _%s folder, skipping.\n", subj_id, suffix);
        continue;
    end

    fprintf("\n--- %s | Task %s ---\n", subj_id, task);

    % load probe info
    probe_fileinfo = dir(fullfile(nirx_path, "*_probeInfo.mat"));
    if isempty(probe_fileinfo)
        fprintf("  WARNING: No probeInfo found for %s, skipping.\n", subj_id);
        continue;
    end
    load(fullfile(probe_fileinfo.folder, probe_fileinfo.name), "probeInfo");

    % load raw NIRx intensity data
    try
        [data, samprate, ~, src_det, ~, sample_time] = extractTechEnData(nirx_path);
    catch ME
        fprintf("  ERROR loading %s: %s\n", subj_id, ME.message);
        continue;
    end

    nT_total = size(data, 1);
    fprintf("  Loaded: %d samples @ %.3f Hz (%.1f sec)\n", nT_total, samprate, nT_total/samprate);

    % find task start AND end from LSL trigger file
    tri_file = dir(fullfile(nirx_path, "*_fixed_lsl.tri"));
    if isempty(tri_file)
        tri_file = dir(fullfile(nirx_path, "*_lsl.tri"));
    end

    if isempty(tri_file)
        fprintf("  WARNING: No .lsl.tri file for %s, using full recording.\n", subj_id);
        start_sample = 1;
        end_sample   = nT_total;
    else
        fprintf("  Using trigger file: %s\n", tri_file(1).name);
        [start_sample, end_sample] = getLSLStartAndEndSample( ...
            fullfile(tri_file(1).folder, tri_file(1).name), samprate, nirx_path);
    end

    % apply lag
    lag_samples   = round(lag * samprate);
    extract_start = start_sample + lag_samples;
    extract_end   = end_sample;

    if extract_start < 1; extract_start = 1; end
    if extract_end > nT_total; extract_end = nT_total; end

    if extract_start > nT_total
        fprintf("  WARNING: start sample %d > recording length %d, skipping.\n", ...
                extract_start, nT_total);
        continue;
    end

    intensity_TxM = data(extract_start:extract_end, :);
    duration_sec  = size(intensity_TxM, 1) / samprate;
    fprintf("  Extracted: samples %d:%d (%.1f sec)\n", extract_start, extract_end, duration_sec);

    if duration_sec < min_duration_sec
        fprintf("  WARNING: only %.1f sec < %d sec min, skipping.\n", duration_sec, min_duration_sec);
        continue;
    end

    % remove bad channels via inpainting
    QCoDthresh = QCoDthresh_base - 0.03 * samprate;
    [intensity_TxM, channelmask] = removeBadChannels(intensity_TxM, samprate, satlength, QCoDthresh);
    src_det.MeasListAct = [channelmask'; channelmask'];
    src_det.MeasListVis = src_det.MeasListAct;

    outfile = fullfile(dir_1_extracted, sprintf("all_%s_%s_TxM.mat", task, subj_id));
    save(outfile, "intensity_TxM", "src_det", "channelmask", "samprate", "probeInfo", "sample_time", "subj_id");
    fprintf("  Saved: %s\n", outfile);
end
fprintf("\nDone: Section 2 [task=%s]\n", task);

%% ── SECTION 3: OD, BANDPASS, MOTION CORRECTION, SHORT-CHANNEL REGRESSION ──

files = dir(fullfile(dir_1_extracted, sprintf("all_%s_*_TxM.mat", task)));
fprintf("Found %d files for task %s\n", length(files), task);

for i = 1:length(files)

    fprintf("\n%s\n", files(i).name);
    load(fullfile(files(i).folder, files(i).name));

    % skip subjects with too little data
    min_samples = round(60 * samprate);
    if size(intensity_TxM, 1) < min_samples
        fprintf("  Skipping: only %.1f sec (< 60 sec minimum)\n", size(intensity_TxM,1)/samprate);
        continue;
    end

    warning off;

    % [1] intensity → optical density
    od_TxM = hmrIntensity2OD(intensity_TxM);

    % [1b] flag non-finite OD channels as bad
    bad_ch = any(~isfinite(od_TxM), 1);
    if any(bad_ch)
        fprintf("  Warning: %d channels with non-finite OD -> marking bad\n", sum(bad_ch));
        channelmask(bad_ch) = 0;
        od_TxM(:, bad_ch) = 0;
    end

    % [2] bandpass filter
    od_TxM = hmrBandpassFilt(od_TxM, samprate, BandpassFilt_l, BandpassFilt_h);

    % [3] motion correction via PCA
    tInc = hmrMotionArtifact(od_TxM, samprate, src_det, ones(size(od_TxM,1),1), ...
                             MotionArtifact_tMotion, MotionArtifact_tMask, ...
                             MotionArtifact_STDEVthresh, MotionArtifact_AMPthresh);
    [od_TxM, ~, ~] = hmrMotionCorrectPCA(src_det, od_TxM, tInc, n_pca);

    % [4] short-channel regression
    % For each standard channel, regress out the signal of its nearest
    % short-separation channel (OLS). Short channels capture scalp/systemic
    % noise rather than brain signal; removing their contribution isolates
    % the neural component before OD→Hb conversion.
    od_TxM_screg = od_TxM;
    n_skipped = 0;
    for j = 1:n_standard_ch
        std_col   = standard_ch_idx(j);
        short_col = nearest_short_ch(j);
        if channelmask(short_col) == 0
            n_skipped = n_skipped + 1;
            continue;
        end
        X = [ones(size(od_TxM,1), 1), od_TxM(:, short_col)];
        y = od_TxM(:, std_col);
        beta = X \ y;
        od_TxM_screg(:, std_col) = y - X * beta;
    end
    if n_skipped > 0
        fprintf("  SCR: %d standard channels skipped (bad short channel neighbor)\n", n_skipped);
    end
    od_TxM = od_TxM_screg;

    warning on;

    outfile = fullfile(dir_2_filtered, files(i).name);
    save(outfile, "od_TxM", "src_det", "channelmask", "samprate", "probeInfo", "sample_time", "subj_id");
    fprintf("  Saved: %s\n", files(i).name);
end
fprintf("\nDone: Section 3 [task=%s]\n", task);

%% ── SECTION 4: OD → HB, Z-SCORE, QA ─────────────────────────────────

files = dir(fullfile(dir_2_filtered, sprintf("all_%s_*_TxM.mat", task)));
fprintf("Found %d files for task %s\n", length(files), task);

for i = 1:length(files)

    fprintf("%s\n", files(i).name);
    load(fullfile(files(i).folder, files(i).name));

    % [1] OD → concentration
    conc_Tx3xR = hmrOD2Conc(od_TxM, src_det, ppf);

    % [2] z-score
    zconc_Tx3xR = zscore(conc_Tx3xR);

    % [3] QA masks (on HbO only)
    qamask  = qualityAssessment(conc_Tx3xR(:, 1, :),  samprate, qamethod, thresh);
    zqamask = qualityAssessment(zconc_Tx3xR(:, 1, :), samprate, qamethod, thresh);

    totalmask  = channelmask; totalmask(~qamask)   = 0;
    ztotalmask = channelmask; ztotalmask(~zqamask) = 0;

    fprintf("  QA: %d/%d channels pass (raw), %d/%d channels pass (z-scored)\n", ...
            sum(totalmask), length(totalmask), sum(ztotalmask), length(ztotalmask));

    % [4] extract and apply masks
    hbo_TxR  = squeeze(conc_Tx3xR(:, 1, :));
    hbr_TxR  = squeeze(conc_Tx3xR(:, 2, :));
    hbt_TxR  = squeeze(conc_Tx3xR(:, 3, :));
    zhbo_TxR = squeeze(zconc_Tx3xR(:, 1, :));
    zhbr_TxR = squeeze(zconc_Tx3xR(:, 2, :));
    zhbt_TxR = squeeze(zconc_Tx3xR(:, 3, :));

    hbo_TxR(:, ~totalmask)   = NaN;
    hbr_TxR(:, ~totalmask)   = NaN;
    hbt_TxR(:, ~totalmask)   = NaN;
    zhbo_TxR(:, ~ztotalmask) = NaN;
    zhbr_TxR(:, ~ztotalmask) = NaN;
    zhbt_TxR(:, ~ztotalmask) = NaN;

    % save (replace _TxM suffix with _noUncertain_TxR)
    [~, fname, ~] = fileparts(files(i).name);
    fname_out = strrep(fname, '_TxM', '') + "_noUncertain_TxR";
    outfile = fullfile(dir_3_QAed, fname_out);
    save(outfile, "hbo_TxR", "hbr_TxR", "hbt_TxR", "zhbo_TxR", "zhbr_TxR", "zhbt_TxR", ...
         "src_det", "samprate", "probeInfo", "sample_time", "subj_id");
    fprintf("  Saved: %s\n", fname_out);
end
fprintf("\nDone: Section 4 [task=%s]\n", task);

%% ── SECTION 5: RESAMPLE TO 1 Hz, TRIM, CONCATENATE ──────────────────

files = dir(fullfile(dir_3_QAed, sprintf("all_%s_*_noUncertain_TxR.mat", task)));
fprintf("Found %d files for task %s\n", length(files), task);

zhbo_rs_all = {};

for j = 1:length(files)
    fprintf("%s\n", files(j).name);
    load(fullfile(files(j).folder, files(j).name));

    nT_orig   = size(zhbo_TxR, 1);
    nT_target = round(nT_orig / samprate);

    p = nT_target; q = nT_orig; g = gcd(p, q);
    zhbo_rs_TxR = resample(zhbo_TxR, p/g, q/g);

    zhbo_rs_all{j} = zhbo_rs_TxR;
    fprintf("  %s: %d samples -> %d samples (%.1f min)\n", ...
            regexp(files(j).name, "p\d+", "match", "once"), ...
            nT_orig, size(zhbo_rs_TxR,1), size(zhbo_rs_TxR,1)/60);
end

min_R = min(cellfun(@(x) size(x,2), zhbo_rs_all));
fprintf("Channel count: %d | Target length: %d sec\n", min_R, target_duration_sec);

zhbo_TxRxS = []; subj_ids_included = {};

for j = 1:length(files)
    rs   = zhbo_rs_all{j};
    subj = regexp(files(j).name, "p\d+", "match", "once");

    if size(rs, 1) < target_duration_sec
        fprintf("  Skipping %s: only %d sec < %d sec target\n", subj, size(rs,1), target_duration_sec);
        continue;
    end

    zhbo_TxRxS = cat(3, zhbo_TxRxS, rs(1:target_duration_sec, 1:min_R));
    subj_ids_included{end+1} = subj;
end

fprintf("Included %d subjects: %s\n", length(subj_ids_included), strjoin(subj_ids_included, ", "));
fprintf("Output: %dx%dx%d (T x R x S)\n", size(zhbo_TxRxS,1), size(zhbo_TxRxS,2), size(zhbo_TxRxS,3));

load(fullfile(files(1).folder, files(1).name), "src_det", "probeInfo");
outfile_resamp = fullfile(dir_4_resampled, sprintf("all_%s_zhbo_TxRxS.mat", task));
save(outfile_resamp, "zhbo_TxRxS", "src_det", "probeInfo", "subj_ids_included");
fprintf("Saved: %s\nDone: Section 5 [task=%s]\n", outfile_resamp, task);

%% ── SECTION 6: ISC ───────────────────────────────────────────────────

load(outfile_resamp, "zhbo_TxRxS", "subj_ids_included");
fprintf("Data: %dx%dx%d\n", size(zhbo_TxRxS,1), size(zhbo_TxRxS,2), size(zhbo_TxRxS,3));

isc_SxR = isc(zhbo_TxRxS, 0);
isc_Sx1 = tanh(nanmean(atanh(isc_SxR), 2));
isc_Rx1 = tanh(nanmean(atanh(isc_SxR), 1));

fprintf("\n%-10s  %-8s  %s\n", "Index", "SubjID", "Mean ISC");
fprintf("%s\n", repmat('-', 1, 35));
for s = 1:length(subj_ids_included)
    fprintf("%-10d  %-8s  %.4f\n", s, subj_ids_included{s}, isc_Sx1(s));
end
fprintf("Grand mean ISC: %.4f\n", mean(isc_Sx1, "omitnan"));

save(outfile_resamp, "isc_SxR", "isc_Sx1", "isc_Rx1", "-append");
fprintf("ISC results saved.\n");
fprintf("\n%s\n*** REVIEW ISC TABLE ABOVE.                            ***\n", repmat('*',1,60));
fprintf("*** Update exclS_NNW in b2_exclude-fnirs.m            ***\n");
fprintf("*** then submit submit_b2_exclude-fnirs.sbatch         ***\n");
fprintf("%s\n", repmat('*',1,60));
fprintf("Done: Section 6 [task=%s]\n", task);


%% ── HELPER FUNCTIONS ─────────────────────────────────────────────────

function nirx_path = find_nirx_path(subj_path, suffix)
    nirx_path = '';
    if ~isfolder(subj_path); return; end
    subdirs = dir(subj_path); subdirs = subdirs([subdirs.isdir]);
    for k = 1:length(subdirs)
        name = subdirs(k).name;
        if endsWith(name, "_" + suffix)
            cand = fullfile(subj_path, name);
            wl1  = dir(fullfile(cand, "*.wl1"));
            if ~isempty(wl1) && wl1(1).bytes > 0; nirx_path = char(cand); return; end
        end
    end
end

function [start_sample, end_sample] = getLSLStartAndEndSample(tri_path, samprate, nirx_path)
% Find start (trigger value=1) and end (trigger value=2) from LSL trigger file.
    fid = fopen(tri_path, "r");
    if fid < 0
        start_sample = 1; end_sample = Inf; return;
    end

    start_sample = 1; end_sample = Inf;
    found_start = false; found_end = false;
    rec_start_dt = getRecordingStartTime(nirx_path);

    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line); break; end
        parts = strsplit(line, ";");
        if length(parts) < 3; continue; end
        val = str2double(parts{3});
        if isnan(val); continue; end

        try
            trigger_dt  = datetime(parts{1}, "InputFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
            time_offset = seconds(trigger_dt - rec_start_dt);
            sample_idx  = max(1, round(time_offset * samprate) + 1);
        catch; continue; end

        if val == 1 && ~found_start
            start_sample = sample_idx; found_start = true;
            fprintf("  Start trigger (val=1) at sample %d (%.2f sec)\n", start_sample, (start_sample-1)/samprate);
        end
        if val == 2 && ~found_end
            end_sample = sample_idx; found_end = true;
            fprintf("  End trigger (val=2) at sample %d (%.2f sec)\n", end_sample, (end_sample-1)/samprate);
        end
        if found_start && found_end; break; end
    end
    fclose(fid);

    if ~found_start; fprintf("  WARNING: No start trigger (val=1) found, using sample 1.\n"); end
    if ~found_end; fprintf("  WARNING: No end trigger (val=2) found, using end of recording.\n"); end
end

function rec_start_dt = getRecordingStartTime(nirx_path)
    hdr_file = dir(fullfile(nirx_path, "*_config.hdr")); rec_start_dt = NaT;
    if isempty(hdr_file); return; end
    fid = fopen(fullfile(hdr_file.folder, hdr_file.name), "r");
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line); break; end
        if startsWith(line, "Date=")
            try; rec_start_dt = datetime(strtrim(strrep(line,"Date=","")), ...
                    "InputFormat", "yyyy-MM-dd HH:mm:ss.SSSSSS");
            catch; rec_start_dt = NaT; end
            break;
        end
    end
    fclose(fid);
end
