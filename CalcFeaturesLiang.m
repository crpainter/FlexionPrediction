% This function can be used to calculate the features matrices for each
% subject for whom we have ECoG data.
% 
% The six features we are using are the same as what are recommended in the
% assignment:
% 1) average time-domain voltage
% 2) average frequency-domain magnitude in the 5-15 Hz frequency band
% 3) average frequency-domain magnitude in the 20-25 Hz frequency band
% 4) average frequency-domain magnitude in the 75-115 Hz frequency band
% 5) average frequency-domain magnitude in the 125-160 Hz frequency band
% 6) average frequency-domain magnitude in the 160-175 Hz frequency band
%
% Inputs are:
% - ecogData = raw ECoG data for this subject
% - numChannels = number of channels for this subject
% - sr = sample rate for 'ecogData'
% - winLen = window length in seconds
% - winDisp = window displacement in seconds
%
% Outputs are: 6 feature matrices for this subject (feat1, feat2, feat3,
% feat4, feat5, feat6), where each feature corresponds to the feature
% descriptions above

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