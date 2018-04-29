% This function can be used to compute the sum-squared-voltage for each
% frequency band of an ECoG signal that is passed into the function. Each
% set of ECoG data has been decomposed into three bands: 1) sub band
% (1-60Hz), 2) gamma band (60-100Hz), 3) fast gamma band (100-200Hz). Each
% band is separately passed in as the "ecogData" argument in this function.
% We calculate the same sum-squared-voltage function for all the windows in
% ecogData.
% 
% Inputs are:
% - ecogData = band-pass-filtered ECoG data for this subject
% - numChannels = number of channels for this subject
% - sr = sample rate for 'ecogData'
% - winLen = window length in seconds
% - winDisp = window displacement in seconds
%
% Output is: 
% - 1 feature matrix for this subject (feat1), which contains
% the output of the sum-squared-voltage for each time window within
% ecogData

function [feat1] = CalcFeaturesLiang(ecogBandData, numChannels, sr, winLen, winDisp)

    % Empty arrays for all features
    [feat1] = deal([]);
    
    % Feature
    SumSqVoltage = @(signal) sum(signal.^2); % function
    for i = 1:numChannels
        % Use MovingWinFeats to calculate SumSqVoltage for each channel for
        % this subject
        result1 = MovingWinFeats(ecogBandData(:,i),sr,winLen,winDisp,SumSqVoltage);
        feat1 = [feat1; result1];
    end
    
end