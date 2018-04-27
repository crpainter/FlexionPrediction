%% Downloading data

% Download all data from the website

% Initiate session
% Open datasets for subject 1
session = IEEGSession('I521_Sub1_Training_ecog','pkshah','pks_ieeglogin.bin');
session.openDataSet('I521_Sub1_Training_dg');
session.openDataSet('I521_Sub1_Leaderboard_ecog');

% Open datasets for subject 2
session.openDataSet('I521_Sub2_Training_ecog');
session.openDataSet('I521_Sub2_Training_dg');
session.openDataSet('I521_Sub2_Leaderboard_ecog');

% Open datasets for subject 3
session.openDataSet('I521_Sub3_Training_ecog');
session.openDataSet('I521_Sub3_Training_dg');
session.openDataSet('I521_Sub3_Leaderboard_ecog');

% Download data for subject 1
ecogData1 = session.data(1).getvalues(1:300000,1:62);
dgData1 = session.data(2).getvalues(1:300000,1:5);
testECOG1 = session.data(3).getvalues(1:147500,1:62);

% Download data for subject 2
ecogData2 = session.data(4).getvalues(1:300000,1:48);
dgData2 = session.data(5).getvalues(1:300000,1:5);
testECOG2 = session.data(6).getvalues(1:147500,1:48);

% Download data for subject 3
ecogData3 = session.data(7).getvalues(1:300000,1:64);
dgData3 = session.data(8).getvalues(1:300000,1:5);
testECOG3 = session.data(9).getvalues(1:147500,1:64);

%% Downsampling

% Vars to hold downsampled data glove data for each subject
dsDG1 = zeros(6000,5);
dsDG2 = zeros(6000,5);
dsDG3 = zeros(6000,5);

for i = 1:5
    dsDG1(:,i) = decimate(dgData1(:,i), 50);
    dsDG2(:,i) = decimate(dgData2(:,i), 50);
    dsDG3(:,i) = decimate(dgData3(:,i), 50);
end
clear i;

%% Saving data

save('finalProjectData.mat','dgData1','dgData2','dgData3','dsDG1','dsDG2',...
    'dsDG3','ecogData1','ecogData2','ecogData3','testECOG1','testECOG2','testECOG3');






%% General algorithm for each subject

% https://www.frontiersin.org/articles/10.3389/fnins.2012.00091/full

% Six steps
% (1/6): Filter training data into 3 frequency bands (sub, gamma, fast
% gamma)
% (2/6): Calculate features and X matrix for training data
% (3/6): Filter testing data into 3 frequency bands (sub, gamma, fast
% gamma)
% (4/6): Calculate features and X matrix for testing data
% (5/6): Train linear regression model and make predictions
% (6/6): Interpolate and pad predictions to produce output of the correct
% length for each subject

% Load filters for sub bands (1-60Hz), gamma bands (60-100Hz), and fast
% gamma bands (100-200Hz)
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/sub.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/gamma.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/fastGamma.mat');


%% Subject 1 (1/6): filter training data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub1SubBand = filtfilt(subCoeffs, 1, ecogData1);
sub1GammaBand = filtfilt(gammaCoeffs, 1, ecogData1);
sub1FastGammaBand = filtfilt(fastGammaCoeffs, 1, ecogData1);

%% Subject 1 (2/6): calculate features and X matrix for training data

% Inputs into feature and X matrix calculations
numChannels = 62; % for subject 1
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
sub1SubBandFeat = CalcFeaturesLiang(sub1SubBand, numChannels, sr, winLen, winDisp);
sub1GammaBandFeat = CalcFeaturesLiang(sub1GammaBand, numChannels, sr, winLen, winDisp);
sub1FastGammaBandFeat = CalcFeaturesLiang(sub1FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
trainX1 = CalcXMatrixLiang(sub1SubBandFeat, sub1GammaBandFeat, sub1FastGammaBandFeat, ...
    N, numChannels);

%% Subject 1 (3/6): filter testing data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub1SubBandT = filtfilt(subCoeffs, 1, testECOG1);
sub1GammaBandT = filtfilt(gammaCoeffs, 1, testECOG1);
sub1FastGammaBandT = filtfilt(fastGammaCoeffs, 1, testECOG1);

%% Subject 1 (4/6): calculate features and X matrix for testing data

% Calculate testing features for each band
sub1SubBandFeatT = CalcFeaturesLiang(sub1SubBandT, numChannels, sr, winLen, winDisp);
sub1GammaBandFeatT = CalcFeaturesLiang(sub1GammaBandT, numChannels, sr, winLen, winDisp);
sub1FastGammaBandFeatT = CalcFeaturesLiang(sub1FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
testX1 = CalcXMatrixLiang(sub1SubBandFeatT, sub1GammaBandFeatT, sub1FastGammaBandFeatT,...
    N, numChannels);

%% Subject 1 (5/6): train linear regression model and make predictions

% Train beta coefficients using training data and make predictions using
% testing data for all 5 fingers simultaneously
sub1Beta = mldivide((trainX1'*trainX1),(trainX1'*dsDG1((N+1):end,:)));
yhat1 = testX1*sub1Beta;

%% Subject 1 (6/6): interpolate and pad predictions to produce final output for subject 1

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat1);
[sub1Fing1, sub1Fing2, sub1Fing3, sub1Fing4, sub1Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub1Fing1 = spline(1:lenPredictions, yhat1(:,1), 0:1/50:(lenPredictions-1/50));
sub1Fing2 = spline(1:lenPredictions, yhat1(:,2), 0:1/50:(lenPredictions-1/50));
sub1Fing3 = spline(1:lenPredictions, yhat1(:,3), 0:1/50:(lenPredictions-1/50));
sub1Fing4 = spline(1:lenPredictions, yhat1(:,4), 0:1/50:(lenPredictions-1/50));
sub1Fing5 = spline(1:lenPredictions, yhat1(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 1
sub1Fing1 = [zeros(1,(N)*50) sub1Fing1];
sub1Fing2 = [zeros(1,(N)*50) sub1Fing2];
sub1Fing3 = [zeros(1,(N)*50) sub1Fing3];
sub1Fing4 = [zeros(1,(N)*50) sub1Fing4];
sub1Fing5 = [zeros(1,(N)*50) sub1Fing5];






%% Subject 2 (1/6): filter training data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub2SubBand = filtfilt(subCoeffs, 1, ecogData2);
sub2GammaBand = filtfilt(gammaCoeffs, 1, ecogData2);
sub2FastGammaBand = filtfilt(fastGammaCoeffs, 1, ecogData2);

%% Subject 2 (2/6): calculate features and X matrix for training data

% Inputs into feature and X matrix calculations
numChannels = 48; % for subject 2
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
sub2SubBandFeat = CalcFeaturesLiang(sub2SubBand, numChannels, sr, winLen, winDisp);
sub2GammaBandFeat = CalcFeaturesLiang(sub2GammaBand, numChannels, sr, winLen, winDisp);
sub2FastGammaBandFeat = CalcFeaturesLiang(sub2FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
trainX2 = CalcXMatrixLiang(sub2SubBandFeat, sub2GammaBandFeat, sub2FastGammaBandFeat, ...
    N, numChannels);

%% Subject 2 (3/6): filter testing data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub2SubBandT = filtfilt(subCoeffs, 1, testECOG2);
sub2GammaBandT = filtfilt(gammaCoeffs, 1, testECOG2);
sub2FastGammaBandT = filtfilt(fastGammaCoeffs, 1, testECOG2);

%% Subject 2 (4/6): calculate features and X matrix for testing data

% Calculate testing features for each band
sub2SubBandFeatT = CalcFeaturesLiang(sub2SubBandT, numChannels, sr, winLen, winDisp);
sub2GammaBandFeatT = CalcFeaturesLiang(sub2GammaBandT, numChannels, sr, winLen, winDisp);
sub2FastGammaBandFeatT = CalcFeaturesLiang(sub2FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
testX2 = CalcXMatrixLiang(sub2SubBandFeatT, sub2GammaBandFeatT, sub2FastGammaBandFeatT,...
    N, numChannels);

%% Subject 2 (5/6): train linear regression model and make predictions

% Train beta coefficients using training data and make predictions using
% testing data for all 5 fingers simultaneously
sub2Beta = mldivide((trainX2'*trainX2),(trainX2'*dsDG2((N+1):end,:)));
yhat2 = testX2*sub2Beta;

%% Subject 2 (6/6): interpolate and pad predictions to produce final output for subject 2

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat2);
[sub2Fing1, sub2Fing2, sub2Fing3, sub2Fing4, sub2Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub2Fing1 = spline(1:lenPredictions, yhat2(:,1), 0:1/50:(lenPredictions-1/50));
sub2Fing2 = spline(1:lenPredictions, yhat2(:,2), 0:1/50:(lenPredictions-1/50));
sub2Fing3 = spline(1:lenPredictions, yhat2(:,3), 0:1/50:(lenPredictions-1/50));
sub2Fing4 = spline(1:lenPredictions, yhat2(:,4), 0:1/50:(lenPredictions-1/50));
sub2Fing5 = spline(1:lenPredictions, yhat2(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 2
sub2Fing1 = [zeros(1,(N)*50) sub2Fing1];
sub2Fing2 = [zeros(1,(N)*50) sub2Fing2];
sub2Fing3 = [zeros(1,(N)*50) sub2Fing3];
sub2Fing4 = [zeros(1,(N)*50) sub2Fing4];
sub2Fing5 = [zeros(1,(N)*50) sub2Fing5];









%% Subject 3 (1/6): filter training data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub3SubBand = filtfilt(subCoeffs, 1, ecogData3);
sub3GammaBand = filtfilt(gammaCoeffs, 1, ecogData3);
sub3FastGammaBand = filtfilt(fastGammaCoeffs, 1, ecogData3);

%% Subject 3 (2/6): calculate features and X matrix for training data

% Inputs into feature and X matrix calculations
numChannels = 64; % for subject 3
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
sub3SubBandFeat = CalcFeaturesLiang(sub3SubBand, numChannels, sr, winLen, winDisp);
sub3GammaBandFeat = CalcFeaturesLiang(sub3GammaBand, numChannels, sr, winLen, winDisp);
sub3FastGammaBandFeat = CalcFeaturesLiang(sub3FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
trainX3 = CalcXMatrixLiang(sub3SubBandFeat, sub3GammaBandFeat, sub3FastGammaBandFeat, ...
    N, numChannels);

%% Subject 3 (3/6): filter testing data into 3 frequency bands

% Sub-bands (1-60Hz), gamma bands (60-100Hz), fast gamma bands (100-200Hz)
sub3SubBandT = filtfilt(subCoeffs, 1, testECOG3);
sub3GammaBandT = filtfilt(gammaCoeffs, 1, testECOG3);
sub3FastGammaBandT = filtfilt(fastGammaCoeffs, 1, testECOG3);

%% Subject 3 (4/6): calculate features and X matrix for testing data

% Calculate testing features for each band
sub3SubBandFeatT = CalcFeaturesLiang(sub3SubBandT, numChannels, sr, winLen, winDisp);
sub3GammaBandFeatT = CalcFeaturesLiang(sub3GammaBandT, numChannels, sr, winLen, winDisp);
sub3FastGammaBandFeatT = CalcFeaturesLiang(sub3FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
testX3 = CalcXMatrixLiang(sub3SubBandFeatT, sub3GammaBandFeatT, sub3FastGammaBandFeatT,...
    N, numChannels);

%% Subject 3 (5/6): train linear regression model and make predictions

% Train beta coefficients using training data and make predictions using
% testing data for all 5 fingers simultaneously
sub3Beta = mldivide((trainX3'*trainX3),(trainX3'*dsDG3((N+1):end,:)));
yhat3 = testX3*sub3Beta;

%% Subject 3 (6/6): interpolate and pad predictions to produce final output for subject 3

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat3);
[sub3Fing1, sub3Fing2, sub3Fing3, sub3Fing4, sub3Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub3Fing1 = spline(1:lenPredictions, yhat3(:,1), 0:1/50:(lenPredictions-1/50));
sub3Fing2 = spline(1:lenPredictions, yhat3(:,2), 0:1/50:(lenPredictions-1/50));
sub3Fing3 = spline(1:lenPredictions, yhat3(:,3), 0:1/50:(lenPredictions-1/50));
sub3Fing4 = spline(1:lenPredictions, yhat3(:,4), 0:1/50:(lenPredictions-1/50));
sub3Fing5 = spline(1:lenPredictions, yhat3(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 3
sub3Fing1 = [zeros(1,(N)*50) sub3Fing1];
sub3Fing2 = [zeros(1,(N)*50) sub3Fing2];
sub3Fing3 = [zeros(1,(N)*50) sub3Fing3];
sub3Fing4 = [zeros(1,(N)*50) sub3Fing4];
sub3Fing5 = [zeros(1,(N)*50) sub3Fing5];








%% Combine output from three subjects into one variable (predicted_dg)

predicted_dg = {};
predicted_dg{1} = [sub1Fing1' sub1Fing2' sub1Fing3' sub1Fing4' sub1Fing5'];
predicted_dg{2} = [sub2Fing1' sub2Fing2' sub2Fing3' sub2Fing4' sub2Fing5'];
predicted_dg{3} = [sub3Fing1' sub3Fing2' sub3Fing3' sub3Fing4' sub3Fing5'];
predicted_dg = predicted_dg';

% Save predicted_dg to .mat file for submission
save('readyPlayerOne_predictions.mat','predicted_dg');


%% Save training features and X matrices

save('trainingFeatsX.mat','sub1SubBandFeat','sub1GammaBandFeat','sub1FastGammaBandFeat',...
    'sub2SubBandFeat','sub2GammaBandFeat','sub2FastGammaBandFeat',...
    'sub3SubBandFeat','sub3GammaBandFeat','sub3FastGammaBandFeat',...
    'trainX1','trainX2','trainX3');


%% Save testing features and X matrices

save('testingFeatsX.mat','sub1SubBandFeatT','sub1GammaBandFeatT','sub1FastGammaBandFeatT',...
    'sub2SubBandFeatT','sub2GammaBandFeatT','sub2FastGammaBandFeatT',...
    'sub3SubBandFeatT','sub3GammaBandFeatT','sub3FastGammaBandFeatT',...
    'testX1','testX2','testX3');


%% Save workspace

save finalProjectAll.mat;
