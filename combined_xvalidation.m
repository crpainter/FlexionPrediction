%% Cross-validation (combination)
% In this script, we take the inputs from the Liang method and our SVM
% model and combine in order to produce more robust output

%% Combine outputs

% Subject 1
xv1Fing1Combined = mean([xv1Fing1Filtered; xv1Fing1FilteredLiang],1);
xv1Fing2Combined = mean([xv1Fing2Filtered; xv1Fing2FilteredLiang],1);
xv1Fing3Combined = mean([xv1Fing3Filtered; xv1Fing3FilteredLiang],1);
xv1Fing4Combined = mean([xv1Fing4Filtered; xv1Fing4FilteredLiang],1);
xv1Fing5Combined = mean([xv1Fing5Filtered; xv1Fing5FilteredLiang],1);

% Subject 2
xv2Fing1Combined = mean([xv2Fing1Filtered; xv2Fing1FilteredLiang],1);
xv2Fing2Combined = mean([xv2Fing2Filtered; xv2Fing2FilteredLiang],1);
xv2Fing3Combined = mean([xv2Fing3Filtered; xv2Fing3FilteredLiang],1);
xv2Fing4Combined = mean([xv2Fing4Filtered; xv2Fing4FilteredLiang],1);
xv2Fing5Combined = mean([xv2Fing5Filtered; xv2Fing5FilteredLiang],1);

% Subject 3
xv3Fing1Combined = mean([xv3Fing1Filtered; xv3Fing1FilteredLiang],1);
xv3Fing2Combined = mean([xv3Fing2Filtered; xv3Fing2FilteredLiang],1);
xv3Fing3Combined = mean([xv3Fing3Filtered; xv3Fing3FilteredLiang],1);
xv3Fing4Combined = mean([xv3Fing4Filtered; xv3Fing4FilteredLiang],1);
xv3Fing5Combined = mean([xv3Fing5Filtered; xv3Fing5FilteredLiang],1);




%% Moving average (on combined output)

sr = 1000; % sample rate
winLen = 150/1e3; % 150ms window
winSamples = winLen*sr; % number of samples in window

% filter coefficient vectors
a = 1;
b = ones(1,winSamples)/winSamples;

% filter all finger data
% Subject 1
xv1Fing1CombFiltered = filtfilt(b,a,xv1Fing1Combined);
xv1Fing2CombFiltered = filtfilt(b,a,xv1Fing2Combined);
xv1Fing3CombFiltered = filtfilt(b,a,xv1Fing3Combined);
xv1Fing4CombFiltered = filtfilt(b,a,xv1Fing4Combined);
xv1Fing5CombFiltered = filtfilt(b,a,xv1Fing5Combined);
% Subject 2
xv2Fing1CombFiltered = filtfilt(b,a,xv2Fing1Combined);
xv2Fing2CombFiltered = filtfilt(b,a,xv2Fing2Combined);
xv2Fing3CombFiltered = filtfilt(b,a,xv2Fing3Combined);
xv2Fing4CombFiltered = filtfilt(b,a,xv2Fing4Combined);
xv2Fing5CombFiltered = filtfilt(b,a,xv2Fing5Combined);
% Subject 3
xv3Fing1CombFiltered = filtfilt(b,a,xv3Fing1Combined);
xv3Fing2CombFiltered = filtfilt(b,a,xv3Fing2Combined);
xv3Fing3CombFiltered = filtfilt(b,a,xv3Fing3Combined);
xv3Fing4CombFiltered = filtfilt(b,a,xv3Fing4Combined);
xv3Fing5CombFiltered = filtfilt(b,a,xv3Fing5Combined);



%% Plot original versus predicted versus predicted/filtered

% Subject 1 finger 2
figure
hold on
plot(xvTestingDataGlove1(:,2)) % actual data
%plot(xv1Fing2Filtered)
%plot(xv1Fing2Combined)
plot(xv1Fing2CombFiltered)
hold off