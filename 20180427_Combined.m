% In this script, we take the inputs from the Liang method and our SVM
% model and combine in order to produce more robust output


%% Combine outputs

% Subject 1
sub1Fing1Combined = mean([sub1Fing1Filtered; sub1Fing1FilteredLiang],1);
sub1Fing2Combined = mean([sub1Fing2Filtered; sub1Fing2FilteredLiang],1);
sub1Fing3Combined = mean([sub1Fing3Filtered; sub1Fing3FilteredLiang],1);
sub1Fing4Combined = mean([sub1Fing4Filtered; sub1Fing4FilteredLiang],1);
sub1Fing5Combined = mean([sub1Fing5Filtered; sub1Fing5FilteredLiang],1);

% Subject 2
sub2Fing1Combined = mean([sub2Fing1Filtered; sub2Fing1FilteredLiang],1);
sub2Fing2Combined = mean([sub2Fing2Filtered; sub2Fing2FilteredLiang],1);
sub2Fing3Combined = mean([sub2Fing3Filtered; sub2Fing3FilteredLiang],1);
sub2Fing4Combined = mean([sub2Fing4Filtered; sub2Fing4FilteredLiang],1);
sub2Fing5Combined = mean([sub2Fing5Filtered; sub2Fing5FilteredLiang],1);

% Subject 3
sub3Fing1Combined = mean([sub3Fing1Filtered; sub3Fing1FilteredLiang],1);
sub3Fing2Combined = mean([sub3Fing2Filtered; sub3Fing2FilteredLiang],1);
sub3Fing3Combined = mean([sub3Fing3Filtered; sub3Fing3FilteredLiang],1);
sub3Fing4Combined = mean([sub3Fing4Filtered; sub3Fing4FilteredLiang],1);
sub3Fing5Combined = mean([sub3Fing5Filtered; sub3Fing5FilteredLiang],1);

%% Filter combined outputs

sr = 1000; % sample rate
winLen = 150/1e3; % 150ms window
winSamples = winLen*sr; % number of samples in window

% filter coefficient vectors
a = 1;
b = ones(1,winSamples)/winSamples;

% filter all finger data
% Subject 1
sub1Fing1CombFiltered = filtfilt(b,a,sub1Fing1Combined);
sub1Fing2CombFiltered = filtfilt(b,a,sub1Fing2Combined);
sub1Fing3CombFiltered = filtfilt(b,a,sub1Fing3Combined);
sub1Fing4CombFiltered = filtfilt(b,a,sub1Fing4Combined);
sub1Fing5CombFiltered = filtfilt(b,a,sub1Fing5Combined);
% Subject 2
sub2Fing1CombFiltered = filtfilt(b,a,sub2Fing1Combined);
sub2Fing2CombFiltered = filtfilt(b,a,sub2Fing2Combined);
sub2Fing3CombFiltered = filtfilt(b,a,sub2Fing3Combined);
sub2Fing4CombFiltered = filtfilt(b,a,sub2Fing4Combined);
sub2Fing5CombFiltered = filtfilt(b,a,sub2Fing5Combined);
% Subject 3
sub3Fing1CombFiltered = filtfilt(b,a,sub3Fing1Combined);
sub3Fing2CombFiltered = filtfilt(b,a,sub3Fing2Combined);
sub3Fing3CombFiltered = filtfilt(b,a,sub3Fing3Combined);
sub3Fing4CombFiltered = filtfilt(b,a,sub3Fing4Combined);
sub3Fing5CombFiltered = filtfilt(b,a,sub3Fing5Combined);


%% Save into predicted_dg

predicted_dg = {};
predicted_dg{1} = [sub1Fing1CombFiltered' sub1Fing2CombFiltered' ...
    sub1Fing3CombFiltered' sub1Fing4CombFiltered' sub1Fing5CombFiltered'];
predicted_dg{2} = [sub2Fing1CombFiltered' sub2Fing2CombFiltered' ...
    sub2Fing3CombFiltered' sub2Fing4CombFiltered' sub2Fing5CombFiltered'];
predicted_dg{3} = [sub3Fing1CombFiltered' sub3Fing2CombFiltered' ...
    sub3Fing3CombFiltered' sub3Fing4CombFiltered' sub3Fing5CombFiltered'];
predicted_dg = predicted_dg';

% Save predicted_dg to .mat file for submission
save('readyPlayerOne_predictions_Combined.mat','predicted_dg');
