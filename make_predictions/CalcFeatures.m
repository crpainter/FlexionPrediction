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

function [feat1,feat2,feat3,feat4,feat5,feat6] = CalcFeatures(ecogData, numChannels, sr, winLen, winDisp)

    % Empty arrays for all features
    [feat1,feat2,feat3,feat4,feat5,feat6] = deal([]);
    
    % Feature 1: Average time-domain voltage
    AvgVoltage = @(signal) mean(signal); % function
    for i = 1:numChannels    
        % Use MovingWinFeats to calculate AvgVoltage for each channel for
        % this subject
        result1 = MovingWinFeats(ecogData(:,i),sr,winLen,winDisp,AvgVoltage);
        feat1 = [feat1; result1];  
    end
    
    % Features 2-6: Average frequency-domain magnitudes in various
    % frequency bands
    winSamples = winLen*sr;
    overlapSamples = winSamples - (winDisp*sr); % # samples in window minus # samples overlapped
    for i = 1:numChannels
    
        % Compute spectral features for this channel for each frequency band
        [s, f, t] = spectrogram(ecogData(:,i), winSamples, overlapSamples, 1:1:sr, sr);
        result2 = mean(abs(s(5:15,:))); % feature 2
        result3 = mean(abs(s(20:25,:))); % feature 3
        result4 = mean(abs(s(75:115,:))); % feature 4
        result5 = mean(abs(s(125:160,:))); % feature 5
        result6 = mean(abs(s(160:175,:))); % feature 6
        %result7 = sum(abs(s(100:200,:)).^2); % feature 7
        %result8 = sum(abs(s(75:115,:)).^2); % feature 8

        % Add result to each feature matrix
        feat2 = [feat2; result2];
        feat3 = [feat3; result3];
        feat4 = [feat4; result4];
        feat5 = [feat5; result5];
        feat6 = [feat6; result6];
        %feat7 = [feat7; result7];
        %feat8 = [feat8; result8];
    
    end
%     
%     % Feature 7: Sum squared voltage in each window
%     SumSqVoltage = @(signal) sum(signal.^2); % function
%     for i = 1:numChannels    
%         % Use MovingWinFeats to calculate SumSqVoltage for each channel for
%         % this subject
%         result1 = MovingWinFeats(ecogData(:,i),sr,winLen,winDisp,SumSqVoltage);
%         feat7 = [feat7; result1];  
%     end
    
end