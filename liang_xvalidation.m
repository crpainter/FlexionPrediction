%% Cross validation (xv) (general algorithm)

% Start by splitting the raw data into half, 50% for "training" and 50% for
% "testing." Then, run the training calculations on the training set
% (caclulate features and X matrices) and run the testing calculations on
% the testing set (calculate features and X matrices). Then use an SVM
% model to train and predict the output. Interpolate and expand
% the output, then calculate correlations with the true finger flexion
% data.

%% Cross validation (Subject 1)

% Split the raw ECoG data in half
xvTrainingECOG1 = ecogData1(1:150000,:); % training data
xvTestingECOG1 = ecogData1(150001:end,:); % testing data

% Split the response variable dataset in half
xvTrainingDataGlove1 = dgData1(1:150000,:); % raw data glove data
xvTestingDataGlove1 = dgData1(150001:end,:); % raw data glove data
xvTrainingDS1 = dsDG1(1:3000,:); % downsampled data glove data
xvTestingDS1 = dsDG1(3001:end,:); % downsampled data glove data

% Load filters for sub bands (1-60Hz), gamma bands (60-100Hz), and fast
% gamma bands (100-200Hz)
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/sub2.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/gamma2.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/fastGamma2.mat');

% Filter training data into three frequency bands
xv1SubBand = filtfilt(subCoeffs2, 1, xvTrainingECOG1);
xv1GammaBand = filtfilt(gammaCoeffs2, 1, xvTrainingECOG1);
xv1FastGammaBand = filtfilt(fastGammaCoeffs2, 1, xvTrainingECOG1);

% Inputs into feature and X matrix calculations
numChannels = 62; % for subject 1
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
xv1SubBandFeat = CalcFeaturesLiang(xv1SubBand, numChannels, sr, winLen, winDisp);
xv1GammaBandFeat = CalcFeaturesLiang(xv1GammaBand, numChannels, sr, winLen, winDisp);
xv1FastGammaBandFeat = CalcFeaturesLiang(xv1FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
xvTrainX1Liang = CalcXMatrixLiang(xv1SubBandFeat, xv1GammaBandFeat, xv1FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv1SubBandT = filtfilt(subCoeffs2, 1, xvTestingECOG1);
xv1GammaBandT = filtfilt(gammaCoeffs2, 1, xvTestingECOG1);
xv1FastGammaBandT = filtfilt(fastGammaCoeffs2, 1, xvTestingECOG1);

% Calculate testing features for each band
xv1SubBandFeatT = CalcFeaturesLiang(xv1SubBandT, numChannels, sr, winLen, winDisp);
xv1GammaBandFeatT = CalcFeaturesLiang(xv1GammaBandT, numChannels, sr, winLen, winDisp);
xv1FastGammaBandFeatT = CalcFeaturesLiang(xv1FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX1Liang = CalcXMatrixLiang(xv1SubBandFeatT, xv1GammaBandFeatT, xv1FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv1BetaLiang = mldivide((xvTrainX1Liang'*xvTrainX1Liang),(xvTrainX1Liang'*xvTrainingDS1((N+1):end,:)));
xvYHat1Liang = xvTestX1Liang*xv1BetaLiang;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove1
lenPredictions = length(xvYHat1Liang);
[xv1Fing1Liang, xv1Fing2Liang, xv1Fing3Liang, xv1Fing4Liang, xv1Fing5Liang] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv1Fing1Liang = spline(1:lenPredictions, xvYHat1Liang(:,1), 0:1/50:(lenPredictions-1/50));
xv1Fing2Liang = spline(1:lenPredictions, xvYHat1Liang(:,2), 0:1/50:(lenPredictions-1/50));
xv1Fing3Liang = spline(1:lenPredictions, xvYHat1Liang(:,3), 0:1/50:(lenPredictions-1/50));
xv1Fing4Liang = spline(1:lenPredictions, xvYHat1Liang(:,4), 0:1/50:(lenPredictions-1/50));
xv1Fing5Liang = spline(1:lenPredictions, xvYHat1Liang(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv1Fing1Liang = [zeros(1,(N)*50) xv1Fing1Liang];
xv1Fing2Liang = [zeros(1,(N)*50) xv1Fing2Liang];
xv1Fing3Liang = [zeros(1,(N)*50) xv1Fing3Liang];
xv1Fing4Liang = [zeros(1,(N)*50) xv1Fing4Liang];
xv1Fing5Liang = [zeros(1,(N)*50) xv1Fing5Liang];

% Compute correlation between predicted values and actual values
xv1Fing1CorrLiang = corr(xv1Fing1Liang', xvTestingDataGlove1(:,1));
xv1Fing2CorrLiang = corr(xv1Fing2Liang', xvTestingDataGlove1(:,2));
xv1Fing3CorrLiang = corr(xv1Fing3Liang', xvTestingDataGlove1(:,3));
xv1Fing4CorrLiang = corr(xv1Fing4Liang', xvTestingDataGlove1(:,4));
xv1Fing5CorrLiang = corr(xv1Fing5Liang', xvTestingDataGlove1(:,5));
% Compute average correlation
xv1CorrLiang = mean([xv1Fing1CorrLiang xv1Fing2CorrLiang xv1Fing3CorrLiang xv1Fing5CorrLiang])





%% Cross validation (Subject 2)

% Split the raw ECoG data in half
xvTrainingECOG2 = ecogData2(1:150000,:); % training data
xvTestingECOG2 = ecogData2(150001:end,:); % testing data

% Split the response variable dataset in half
xvTrainingDataGlove2 = dgData2(1:150000,:); % raw data glove data
xvTestingDataGlove2 = dgData2(150001:end,:); % raw data glove data
xvTrainingDS2 = dsDG2(1:3000,:); % downsampled data glove data
xvTestingDS2 = dsDG2(3001:end,:); % downsampled data glove data

% Filter training data into three frequency bands
xv2SubBand = filtfilt(subCoeffs2, 1, xvTrainingECOG2);
xv2GammaBand = filtfilt(gammaCoeffs2, 1, xvTrainingECOG2);
xv2FastGammaBand = filtfilt(fastGammaCoeffs2, 1, xvTrainingECOG2);

% Inputs into feature and X matrix calculations
numChannels = 48; % for subject 2
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
xv2SubBandFeat = CalcFeaturesLiang(xv2SubBand, numChannels, sr, winLen, winDisp);
xv2GammaBandFeat = CalcFeaturesLiang(xv2GammaBand, numChannels, sr, winLen, winDisp);
xv2FastGammaBandFeat = CalcFeaturesLiang(xv2FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
xvTrainX2Liang = CalcXMatrixLiang(xv2SubBandFeat, xv2GammaBandFeat, xv2FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv2SubBandT = filtfilt(subCoeffs2, 1, xvTestingECOG2);
xv2GammaBandT = filtfilt(gammaCoeffs2, 1, xvTestingECOG2);
xv2FastGammaBandT = filtfilt(fastGammaCoeffs2, 1, xvTestingECOG2);

% Calculate testing features for each band
xv2SubBandFeatT = CalcFeaturesLiang(xv2SubBandT, numChannels, sr, winLen, winDisp);
xv2GammaBandFeatT = CalcFeaturesLiang(xv2GammaBandT, numChannels, sr, winLen, winDisp);
xv2FastGammaBandFeatT = CalcFeaturesLiang(xv2FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX2Liang = CalcXMatrixLiang(xv2SubBandFeatT, xv2GammaBandFeatT, xv2FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv2BetaLiang = mldivide((xvTrainX2Liang'*xvTrainX2Liang),(xvTrainX2Liang'*xvTrainingDS2((N+1):end,:)));
xvYHat2Liang = xvTestX2Liang*xv2BetaLiang;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove2
lenPredictions = length(xvYHat2Liang);
[xv2Fing1Liang, xv2Fing2Liang, xv2Fing3Liang, xv2Fing4Liang, xv2Fing5Liang] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv2Fing1Liang = spline(1:lenPredictions, xvYHat2Liang(:,1), 0:1/50:(lenPredictions-1/50));
xv2Fing2Liang = spline(1:lenPredictions, xvYHat2Liang(:,2), 0:1/50:(lenPredictions-1/50));
xv2Fing3Liang = spline(1:lenPredictions, xvYHat2Liang(:,3), 0:1/50:(lenPredictions-1/50));
xv2Fing4Liang = spline(1:lenPredictions, xvYHat2Liang(:,4), 0:1/50:(lenPredictions-1/50));
xv2Fing5Liang = spline(1:lenPredictions, xvYHat2Liang(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv2Fing1Liang = [zeros(1,(N)*50) xv2Fing1Liang];
xv2Fing2Liang = [zeros(1,(N)*50) xv2Fing2Liang];
xv2Fing3Liang = [zeros(1,(N)*50) xv2Fing3Liang];
xv2Fing4Liang = [zeros(1,(N)*50) xv2Fing4Liang];
xv2Fing5Liang = [zeros(1,(N)*50) xv2Fing5Liang];

% Compute correlation between predicted values and actual values
xv2Fing1CorrLiang = corr(xv2Fing1Liang', xvTestingDataGlove2(:,1));
xv2Fing2CorrLiang = corr(xv2Fing2Liang', xvTestingDataGlove2(:,2));
xv2Fing3CorrLiang = corr(xv2Fing3Liang', xvTestingDataGlove2(:,3));
xv2Fing4CorrLiang = corr(xv2Fing4Liang', xvTestingDataGlove2(:,4));
xv2Fing5CorrLiang = corr(xv2Fing5Liang', xvTestingDataGlove2(:,5));
% Compute average correlation
xv2CorrLiang = mean([xv2Fing1CorrLiang xv2Fing2CorrLiang xv2Fing3CorrLiang xv2Fing5CorrLiang])





%% Cross validation (Subject 3)

% Split the raw ECoG data in half
xvTrainingECOG3 = ecogData3(1:150000,:); % training data
xvTestingECOG3 = ecogData3(150001:end,:); % testing data

% Split the response variable dataset in half
xvTrainingDataGlove3 = dgData3(1:150000,:); % raw data glove data
xvTestingDataGlove3 = dgData3(150001:end,:); % raw data glove data
xvTrainingDS3 = dsDG3(1:3000,:); % downsampled data glove data
xvTestingDS3 = dsDG3(3001:end,:); % downsampled data glove data

% Filter training data into three frequency bands
xv3SubBand = filtfilt(subCoeffs2, 1, xvTrainingECOG3);
xv3GammaBand = filtfilt(gammaCoeffs2, 1, xvTrainingECOG3);
xv3FastGammaBand = filtfilt(fastGammaCoeffs2, 1, xvTrainingECOG3);

% Inputs into feature and X matrix calculations
numChannels = 64; % for subject 3
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% Calculate training features for each band
xv3SubBandFeat = CalcFeaturesLiang(xv3SubBand, numChannels, sr, winLen, winDisp);
xv3GammaBandFeat = CalcFeaturesLiang(xv3GammaBand, numChannels, sr, winLen, winDisp);
xv3FastGammaBandFeat = CalcFeaturesLiang(xv3FastGammaBand, numChannels, sr, winLen, winDisp);

% Calculate training X matrix
xvTrainX3Liang = CalcXMatrixLiang(xv3SubBandFeat, xv3GammaBandFeat, xv3FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv3SubBandT = filtfilt(subCoeffs2, 1, xvTestingECOG3);
xv3GammaBandT = filtfilt(gammaCoeffs2, 1, xvTestingECOG3);
xv3FastGammaBandT = filtfilt(fastGammaCoeffs2, 1, xvTestingECOG3);

% Calculate testing features for each band
xv3SubBandFeatT = CalcFeaturesLiang(xv3SubBandT, numChannels, sr, winLen, winDisp);
xv3GammaBandFeatT = CalcFeaturesLiang(xv3GammaBandT, numChannels, sr, winLen, winDisp);
xv3FastGammaBandFeatT = CalcFeaturesLiang(xv3FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX3Liang = CalcXMatrixLiang(xv3SubBandFeatT, xv3GammaBandFeatT, xv3FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv3BetaLiang = mldivide((xvTrainX3Liang'*xvTrainX3Liang),(xvTrainX3Liang'*xvTrainingDS3((N+1):end,:)));
xvYHat3Liang = xvTestX3Liang*xv3BetaLiang;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove3
lenPredictions = length(xvYHat3Liang);
[xv3Fing1Liang, xv3Fing2Liang, xv3Fing3Liang, xv3Fing4Liang, xv3Fing5Liang] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv3Fing1Liang = spline(1:lenPredictions, xvYHat3Liang(:,1), 0:1/50:(lenPredictions-1/50));
xv3Fing2Liang = spline(1:lenPredictions, xvYHat3Liang(:,2), 0:1/50:(lenPredictions-1/50));
xv3Fing3Liang = spline(1:lenPredictions, xvYHat3Liang(:,3), 0:1/50:(lenPredictions-1/50));
xv3Fing4Liang = spline(1:lenPredictions, xvYHat3Liang(:,4), 0:1/50:(lenPredictions-1/50));
xv3Fing5Liang = spline(1:lenPredictions, xvYHat3Liang(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv3Fing1Liang = [zeros(1,(N)*50) xv3Fing1Liang];
xv3Fing2Liang = [zeros(1,(N)*50) xv3Fing2Liang];
xv3Fing3Liang = [zeros(1,(N)*50) xv3Fing3Liang];
xv3Fing4Liang = [zeros(1,(N)*50) xv3Fing4Liang];
xv3Fing5Liang = [zeros(1,(N)*50) xv3Fing5Liang];

% Compute correlation between predicted values and actual values
xv3Fing1CorrLiang = corr(xv3Fing1Liang', xvTestingDataGlove3(:,1));
xv3Fing2CorrLiang = corr(xv3Fing2Liang', xvTestingDataGlove3(:,2));
xv3Fing3CorrLiang = corr(xv3Fing3Liang', xvTestingDataGlove3(:,3));
xv3Fing4CorrLiang = corr(xv3Fing4Liang', xvTestingDataGlove3(:,4));
xv3Fing5CorrLiang = corr(xv3Fing5Liang', xvTestingDataGlove3(:,5));
% Compute average correlation
xv3CorrLiang = mean([xv3Fing1CorrLiang xv3Fing2CorrLiang xv3Fing3CorrLiang xv3Fing5CorrLiang])





%% Moving average (post-processing)

sr = 1000; % sample rate
winLen = 150/1e3; % 150ms window
winSamples = winLen*sr; % number of samples in window

% filter coefficient vectors
a = 1;
b = ones(1,winSamples)/winSamples;

% filter all finger data
% Subject 1
xv1Fing1FilteredLiang = filtfilt(b,a,xv1Fing1Liang);
xv1Fing2FilteredLiang = filtfilt(b,a,xv1Fing2Liang);
xv1Fing3FilteredLiang = filtfilt(b,a,xv1Fing3Liang);
xv1Fing4FilteredLiang = filtfilt(b,a,xv1Fing4Liang);
xv1Fing5FilteredLiang = filtfilt(b,a,xv1Fing5Liang);
% Subject 2
xv2Fing1FilteredLiang = filtfilt(b,a,xv2Fing1Liang);
xv2Fing2FilteredLiang = filtfilt(b,a,xv2Fing2Liang);
xv2Fing3FilteredLiang = filtfilt(b,a,xv2Fing3Liang);
xv2Fing4FilteredLiang = filtfilt(b,a,xv2Fing4Liang);
xv2Fing5FilteredLiang = filtfilt(b,a,xv2Fing5Liang);
% Subject 3
xv3Fing1FilteredLiang = filtfilt(b,a,xv3Fing1Liang);
xv3Fing2FilteredLiang = filtfilt(b,a,xv3Fing2Liang);
xv3Fing3FilteredLiang = filtfilt(b,a,xv3Fing3Liang);
xv3Fing4FilteredLiang = filtfilt(b,a,xv3Fing4Liang);
xv3Fing5FilteredLiang = filtfilt(b,a,xv3Fing5Liang);



%% New correlations (filtered)

% Subject 1
% Compute correlation between predicted/filtered values and actual values
xv1Fing1FilteredLiangCorr = corr(xv1Fing1FilteredLiang', xvTestingDataGlove1(:,1));
xv1Fing2FilteredLiangCorr = corr(xv1Fing2FilteredLiang', xvTestingDataGlove1(:,2));
xv1Fing3FilteredLiangCorr = corr(xv1Fing3FilteredLiang', xvTestingDataGlove1(:,3));
xv1Fing4FilteredLiangCorr = corr(xv1Fing4FilteredLiang', xvTestingDataGlove1(:,4));
xv1Fing5FilteredLiangCorr = corr(xv1Fing5FilteredLiang', xvTestingDataGlove1(:,5));
% Compute average correlation
xv1CorrFilteredLiang = mean([xv1Fing1FilteredLiangCorr xv1Fing2FilteredLiangCorr...
    xv1Fing3FilteredLiangCorr xv1Fing5FilteredLiangCorr])

% Subject 2
% Compute correlation between predicted values and actual values
xv2Fing1FilteredLiangCorr = corr(xv2Fing1FilteredLiang', xvTestingDataGlove2(:,1));
xv2Fing2FilteredLiangCorr = corr(xv2Fing2FilteredLiang', xvTestingDataGlove2(:,2));
xv2Fing3FilteredLiangCorr = corr(xv2Fing3FilteredLiang', xvTestingDataGlove2(:,3));
xv2Fing4FilteredLiangCorr = corr(xv2Fing4FilteredLiang', xvTestingDataGlove2(:,4));
xv2Fing5FilteredLiangCorr = corr(xv2Fing5FilteredLiang', xvTestingDataGlove2(:,5));
% Compute average correlation
xv2CorrFilteredLiang = mean([xv2Fing1FilteredLiangCorr xv2Fing2FilteredLiangCorr...
    xv2Fing3FilteredLiangCorr xv2Fing5FilteredLiangCorr])

% Subject 3
xv3Fing1FilteredLiangCorr = corr(xv3Fing1FilteredLiang', xvTestingDataGlove3(:,1));
xv3Fing2FilteredLiangCorr = corr(xv3Fing2FilteredLiang', xvTestingDataGlove3(:,2));
xv3Fing3FilteredLiangCorr = corr(xv3Fing3FilteredLiang', xvTestingDataGlove3(:,3));
xv3Fing4FilteredLiangCorr = corr(xv3Fing4FilteredLiang', xvTestingDataGlove3(:,4));
xv3Fing5FilteredLiangCorr = corr(xv3Fing5FilteredLiang', xvTestingDataGlove3(:,5));
% Compute average correlation
xv3CorrFilteredLiang = mean([xv3Fing1FilteredLiangCorr xv3Fing2FilteredLiangCorr...
    xv3Fing3FilteredLiangCorr xv3Fing5FilteredLiangCorr])


%% Overall cross-validation score (filtered)
xvCorrFilteredLiang = mean([xv3Fing1FilteredLiangCorr xv3Fing2FilteredLiangCorr xv3Fing3FilteredLiangCorr...
    xv3Fing5FilteredLiangCorr xv2Fing1FilteredLiangCorr xv2Fing2FilteredLiangCorr xv2Fing3FilteredLiangCorr...
    xv2Fing5FilteredLiangCorr xv1Fing1FilteredLiangCorr xv1Fing2FilteredLiangCorr xv1Fing3FilteredLiangCorr...
    xv1Fing5FilteredLiangCorr])




%% Overall cross-validation score
xvCorrLiang = mean([xv3Fing1CorrLiang xv3Fing2CorrLiang xv3Fing3CorrLiang xv3Fing5CorrLiang xv2Fing1CorrLiang...
    xv2Fing2CorrLiang xv2Fing3CorrLiang xv2Fing5CorrLiang xv1Fing1CorrLiang xv1Fing2CorrLiang ...
    xv1Fing3CorrLiang xv1Fing5CorrLiang])






%% Plot original versus predicted

% Subject 1 finger 1
figure
plot(xv1Fing2Liang)
hold on
plot(xvTestingDataGlove1(:,2))
hold off


