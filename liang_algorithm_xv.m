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
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/sub.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/gamma.mat');
load('/Users/pooja/Dropbox/1-PENN/WHARTON/2018-SPRING/BE521/final-project/filters/fastGamma.mat');

% Filter training data into three frequency bands
xv1SubBand = filtfilt(subCoeffs, 1, xvTrainingECOG1);
xv1GammaBand = filtfilt(gammaCoeffs, 1, xvTrainingECOG1);
xv1FastGammaBand = filtfilt(fastGammaCoeffs, 1, xvTrainingECOG1);

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
xvTrainX1 = CalcXMatrixLiang(xv1SubBandFeat, xv1GammaBandFeat, xv1FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv1SubBandT = filtfilt(subCoeffs, 1, xvTestingECOG1);
xv1GammaBandT = filtfilt(gammaCoeffs, 1, xvTestingECOG1);
xv1FastGammaBandT = filtfilt(fastGammaCoeffs, 1, xvTestingECOG1);

% Calculate testing features for each band
xv1SubBandFeatT = CalcFeaturesLiang(xv1SubBandT, numChannels, sr, winLen, winDisp);
xv1GammaBandFeatT = CalcFeaturesLiang(xv1GammaBandT, numChannels, sr, winLen, winDisp);
xv1FastGammaBandFeatT = CalcFeaturesLiang(xv1FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX1 = CalcXMatrixLiang(xv1SubBandFeatT, xv1GammaBandFeatT, xv1FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv1Beta = mldivide((xvTrainX1'*xvTrainX1),(xvTrainX1'*xvTrainingDS1((N+1):end,:)));
xvYHat1 = xvTestX1*xv1Beta;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove1
lenPredictions = length(xvYHat1);
[xv1Fing1, xv1Fing2, xv1Fing3, xv1Fing4, xv1Fing5] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv1Fing1 = spline(1:lenPredictions, xvYHat1(:,1), 0:1/50:(lenPredictions-1/50));
xv1Fing2 = spline(1:lenPredictions, xvYHat1(:,2), 0:1/50:(lenPredictions-1/50));
xv1Fing3 = spline(1:lenPredictions, xvYHat1(:,3), 0:1/50:(lenPredictions-1/50));
xv1Fing4 = spline(1:lenPredictions, xvYHat1(:,4), 0:1/50:(lenPredictions-1/50));
xv1Fing5 = spline(1:lenPredictions, xvYHat1(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv1Fing1 = [zeros(1,(N)*50) xv1Fing1];
xv1Fing2 = [zeros(1,(N)*50) xv1Fing2];
xv1Fing3 = [zeros(1,(N)*50) xv1Fing3];
xv1Fing4 = [zeros(1,(N)*50) xv1Fing4];
xv1Fing5 = [zeros(1,(N)*50) xv1Fing5];

% Compute correlation between predicted values and actual values
xv1Fing1Corr = corr(xv1Fing1', xvTestingDataGlove1(:,1))
xv1Fing2Corr = corr(xv1Fing2', xvTestingDataGlove1(:,2))
xv1Fing3Corr = corr(xv1Fing3', xvTestingDataGlove1(:,3))
xv1Fing4Corr = corr(xv1Fing4', xvTestingDataGlove1(:,4))
xv1Fing5Corr = corr(xv1Fing5', xvTestingDataGlove1(:,5))
% Compute average correlation
xv1Corr = mean([xv1Fing1Corr xv1Fing2Corr xv1Fing3Corr xv1Fing5Corr])





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
xv2SubBand = filtfilt(subCoeffs, 1, xvTrainingECOG2);
xv2GammaBand = filtfilt(gammaCoeffs, 1, xvTrainingECOG2);
xv2FastGammaBand = filtfilt(fastGammaCoeffs, 1, xvTrainingECOG2);

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
xvTrainX2 = CalcXMatrixLiang(xv2SubBandFeat, xv2GammaBandFeat, xv2FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv2SubBandT = filtfilt(subCoeffs, 1, xvTestingECOG2);
xv2GammaBandT = filtfilt(gammaCoeffs, 1, xvTestingECOG2);
xv2FastGammaBandT = filtfilt(fastGammaCoeffs, 1, xvTestingECOG2);

% Calculate testing features for each band
xv2SubBandFeatT = CalcFeaturesLiang(xv2SubBandT, numChannels, sr, winLen, winDisp);
xv2GammaBandFeatT = CalcFeaturesLiang(xv2GammaBandT, numChannels, sr, winLen, winDisp);
xv2FastGammaBandFeatT = CalcFeaturesLiang(xv2FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX2 = CalcXMatrixLiang(xv2SubBandFeatT, xv2GammaBandFeatT, xv2FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv2Beta = mldivide((xvTrainX2'*xvTrainX2),(xvTrainX2'*xvTrainingDS2((N+1):end,:)));
xvYHat2 = xvTestX2*xv2Beta;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove2
lenPredictions = length(xvYHat2);
[xv2Fing1, xv2Fing2, xv2Fing3, xv2Fing4, xv2Fing5] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv2Fing1 = spline(1:lenPredictions, xvYHat2(:,1), 0:1/50:(lenPredictions-1/50));
xv2Fing2 = spline(1:lenPredictions, xvYHat2(:,2), 0:1/50:(lenPredictions-1/50));
xv2Fing3 = spline(1:lenPredictions, xvYHat2(:,3), 0:1/50:(lenPredictions-1/50));
xv2Fing4 = spline(1:lenPredictions, xvYHat2(:,4), 0:1/50:(lenPredictions-1/50));
xv2Fing5 = spline(1:lenPredictions, xvYHat2(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv2Fing1 = [zeros(1,(N)*50) xv2Fing1];
xv2Fing2 = [zeros(1,(N)*50) xv2Fing2];
xv2Fing3 = [zeros(1,(N)*50) xv2Fing3];
xv2Fing4 = [zeros(1,(N)*50) xv2Fing4];
xv2Fing5 = [zeros(1,(N)*50) xv2Fing5];

% Compute correlation between predicted values and actual values
xv2Fing1Corr = corr(xv2Fing1', xvTestingDataGlove2(:,1))
xv2Fing2Corr = corr(xv2Fing2', xvTestingDataGlove2(:,2))
xv2Fing3Corr = corr(xv2Fing3', xvTestingDataGlove2(:,3))
xv2Fing4Corr = corr(xv2Fing4', xvTestingDataGlove2(:,4))
xv2Fing5Corr = corr(xv2Fing5', xvTestingDataGlove2(:,5))
% Compute average correlation
xv2Corr = mean([xv2Fing1Corr xv2Fing2Corr xv2Fing3Corr xv2Fing5Corr])





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
xv3SubBand = filtfilt(subCoeffs, 1, xvTrainingECOG3);
xv3GammaBand = filtfilt(gammaCoeffs, 1, xvTrainingECOG3);
xv3FastGammaBand = filtfilt(fastGammaCoeffs, 1, xvTrainingECOG3);

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
xvTrainX3 = CalcXMatrixLiang(xv3SubBandFeat, xv3GammaBandFeat, xv3FastGammaBandFeat, ...
    N, numChannels);

% Filter testing data into three frequency bands
xv3SubBandT = filtfilt(subCoeffs, 1, xvTestingECOG3);
xv3GammaBandT = filtfilt(gammaCoeffs, 1, xvTestingECOG3);
xv3FastGammaBandT = filtfilt(fastGammaCoeffs, 1, xvTestingECOG3);

% Calculate testing features for each band
xv3SubBandFeatT = CalcFeaturesLiang(xv3SubBandT, numChannels, sr, winLen, winDisp);
xv3GammaBandFeatT = CalcFeaturesLiang(xv3GammaBandT, numChannels, sr, winLen, winDisp);
xv3FastGammaBandFeatT = CalcFeaturesLiang(xv3FastGammaBandT, numChannels, sr, winLen, winDisp);

% Calculate testing X matrix
xvTestX3 = CalcXMatrixLiang(xv3SubBandFeatT, xv3GammaBandFeatT, xv3FastGammaBandFeatT,...
    N, numChannels);

% Train linear model and make predictions
xv3Beta = mldivide((xvTrainX3'*xvTrainX3),(xvTrainX3'*xvTrainingDS3((N+1):end,:)));
xvYHat3 = xvTestX3*xv3Beta;

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove3
lenPredictions = length(xvYHat3);
[xv3Fing1, xv3Fing2, xv3Fing3, xv3Fing4, xv3Fing5] = deal([]);
% Interpolate (should end up with output that is (N-1)*50 samples too short)
xv3Fing1 = spline(1:lenPredictions, xvYHat3(:,1), 0:1/50:(lenPredictions-1/50));
xv3Fing2 = spline(1:lenPredictions, xvYHat3(:,2), 0:1/50:(lenPredictions-1/50));
xv3Fing3 = spline(1:lenPredictions, xvYHat3(:,3), 0:1/50:(lenPredictions-1/50));
xv3Fing4 = spline(1:lenPredictions, xvYHat3(:,4), 0:1/50:(lenPredictions-1/50));
xv3Fing5 = spline(1:lenPredictions, xvYHat3(:,5), 0:1/50:(lenPredictions-1/50));
% Add padding
xv3Fing1 = [zeros(1,(N)*50) xv3Fing1];
xv3Fing2 = [zeros(1,(N)*50) xv3Fing2];
xv3Fing3 = [zeros(1,(N)*50) xv3Fing3];
xv3Fing4 = [zeros(1,(N)*50) xv3Fing4];
xv3Fing5 = [zeros(1,(N)*50) xv3Fing5];

% Compute correlation between predicted values and actual values
xv3Fing1Corr = corr(xv3Fing1', xvTestingDataGlove3(:,1))
xv3Fing2Corr = corr(xv3Fing2', xvTestingDataGlove3(:,2))
xv3Fing3Corr = corr(xv3Fing3', xvTestingDataGlove3(:,3))
xv3Fing4Corr = corr(xv3Fing4', xvTestingDataGlove3(:,4))
xv3Fing5Corr = corr(xv3Fing5', xvTestingDataGlove3(:,5))
% Compute average correlation
xv3Corr = mean([xv3Fing1Corr xv3Fing2Corr xv3Fing3Corr xv3Fing5Corr])







%% Overall cross-validation score
xvCorr = mean([xv3Fing1Corr xv3Fing2Corr xv3Fing3Corr xv3Fing5Corr xv2Fing1Corr...
    xv2Fing2Corr xv2Fing3Corr xv2Fing5Corr xv1Fing1Corr xv1Fing2Corr ...
    xv1Fing3Corr xv1Fing5Corr])



%% Plot original versus predicted

% Subject 1 finger 1
figure
plot(xv1Fing2)
hold on
plot(xvTestingDataGlove1(:,2))
hold off


%% 

subplot(5,1,1)
plot(0:12/200:(24-12/200),dsDG1(1:400,1))
xlim([0 24])
ax = gca;
ax.XTick = [0 2 4 6 8 10 12 14 16 18 20 22 24];
subplot(5,1,2)
plot(0:12/200:(24-12/200),dsDG1(1:400,2))
xlim([0 24])
ax = gca;
ax.XTick = [0 2 4 6 8 10 12 14 16 18 20 22 24];
subplot(5,1,3)
plot(0:12/200:(24-12/200),dsDG1(1:400,3))
xlim([0 24])
ax = gca;
ax.XTick = [0 2 4 6 8 10 12 14 16 18 20 22 24];
subplot(5,1,4)
plot(0:12/200:(24-12/200),dsDG1(1:400,4))
xlim([0 24])
ax = gca;
ax.XTick = [0 2 4 6 8 10 12 14 16 18 20 22 24];
subplot(5,1,5)
plot(0:12/200:(24-12/200),dsDG1(1:400,5))
xlim([0 24])
ax = gca;
ax.XTick = [0 2 4 6 8 10 12 14 16 18 20 22 24];


%%
plot(0:1/50:(lenPredictions-1/50),spline(1:lenPredictions, yhat3Fing1, 0:1/50:(lenPredictions-1/50)))
hold on
plot(yhat3Fing1)
hold off
legend('Splined','Original')
