% load Sherlock valence 
% load('/Users/cablab/Desktop/VAM/data/beh/preprocessed/group_average/conv_slidingBeh/FNL_valence.mat')

% phase randomize convolved behavioral timecourse
% surriter = 1000;
surriter = 1500;
surr_beh = phaserandomization(sliding_beh, surriter);

% save data
savepath = '/Users/cablab/Desktop/VAM/data/beh/preprocessed/group_average/z_phase_randomized/';
save([savepath,'Sher_arousal_surr_more.mat'],'surr_beh');
%save([savepath,'/arousal/surr_arousal.mat'],'sliding_surr_beh');



function surrts = phaserandomization(realts, nsurr)

% source code: mathworks, created by Carlos Gias (Date: 21/08/2011)
% https://kr.mathworks.com/matlabcentral/fileexchange/32621-phase-randomization

[nfrms,nts] = size(realts);
if rem(nfrms,2)==0
    nfrms = nfrms-1;
    realts=realts(1:nfrms,:);
end

% Get parameters
len_ser = (nfrms-1)/2;
interv1 = 2:len_ser+1;
interv2 = len_ser+2:nfrms;

% Fourier transform of the realts dataset
fft_realts = fft(realts);
% Create the surrogate recording bl
% ocks one by one
surrts = zeros(nfrms, nsurr);
for k = 1:nsurr
    ph_rnd = rand([len_ser 1]);
    
    % Create the random phases for all the time series
    ph_interv1 = repmat(exp( 2*pi*1i*ph_rnd),1,nts);
    ph_interv2 = conj( flipud( ph_interv1));
    
    % Replace randomly created phases to FFT result
    fft_realts_surr = fft_realts;
    fft_realts_surr(interv1,:) = fft_realts(interv1,:).*ph_interv1;
    fft_realts_surr(interv2,:) = fft_realts(interv2,:).*ph_interv2;
    
    % Inverse transform
    surrts(:,k)= real(ifft(fft_realts_surr));
end
surrts = [zeros(1,nsurr); surrts];
end

function sliding_ts = slidingwindow(ts, nT, wsize, sigma)
% code created by Bo-yong Park: https://by9433.wixsite.com/boyongpark
if size(ts,1)~=nT
    error('number of rows of variable "ts" should match a variable "nT".');
end

% tapered sliding window: generate gaussian function for convolution
% Allen et al. 2014, Cerebral Cortex
if mod(nT,2) ~= 0
    m = ceil(nT/2);
    x = 0:nT;
else
    m = nT/2;
    x = 0:nT-1;
end

w = round(wsize/2);
gw = exp(- ((x-m).^2) / (2*sigma*sigma))';
b = zeros(nT,1); b((m-w+1):(m+w)) = 1;
c = conv(gw, b); c = c/max(c); c = c(m+1:end-m+1);
c = c(1:nT);

A = repmat(c,1,1);
Nwin = nT - wsize;
FNCdyn = zeros(Nwin, 1);

% apply sliding window
tcwin = zeros(Nwin, nT);
for ii = 1:Nwin
    % slide gaussian centered on [1+wsize/2, nT-wsize/2]
    Ashift = circshift(A, round(-nT/2) + round(wsize/2) + ii);
    
    % when using "circshift", prevent spillover of the gaussian
    % to either the beginning or an end of the timeseries
    if ii<floor(Nwin/2) & Ashift(end,1)~=0
        Ashift(ceil(Nwin/2):end,:) = 0;
        Ashift = Ashift.*(sum(A(:,1))/sum(Ashift(1:floor(Nwin/2),1)));
    elseif ii>floor(Nwin/2) & Ashift(1,1)~=0
        Ashift(1:floor(Nwin/2),:) = 0;
        Ashift = Ashift.*(sum(A(:,1))/sum(Ashift(ceil(Nwin/2):end,1)));
    end
    
    % apply gaussian weighted sliding window of the timeseries
    tcwin(ii, :) = squeeze(ts).*Ashift;
end

% normalize for a final round after sliding-window
sliding_ts = zscore(nansum(tcwin,2));
end
