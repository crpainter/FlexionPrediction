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

% Inputs into feature and X matrix calculations
numChannels = 62; % for subject 1
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% For training set, calculate features and X matrix
[xv1TrainFeat1, xv1TrainFeat2, xv1TrainFeat3, xv1TrainFeat4, xv1TrainFeat5, xv1TrainFeat6] = ...
    CalcFeatures(xvTrainingECOG1, numChannels, sr, winLen, winDisp);
xvTrainX1 = CalcXMatrix(xv1TrainFeat1, xv1TrainFeat2, xv1TrainFeat3, xv1TrainFeat4, ...
    xv1TrainFeat5, xv1TrainFeat6, N, numChannels);

% For testing set, calculate features and X matrix
[xv1TestFeat1, xv1TestFeat2, xv1TestFeat3, xv1TestFeat4, xv1TestFeat5, xv1TestFeat6] = ...
    CalcFeatures(xvTestingECOG1, numChannels, sr, winLen, winDisp);
xvTestX1 = CalcXMatrix(xv1TestFeat1, xv1TestFeat2, xv1TestFeat3, xv1TestFeat4, ...
    xv1TestFeat5, xv1TestFeat6, N, numChannels);

% Use SVM model for each finger individually
% Finger 1
xv1SVMFing1 = fitrsvm(xvTrainX1,xvTrainingDS1((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
xv1YHatFing1 = predict(xv1SVMFing1,xvTestX1);
% Finger 2
xv1SVMFing2 = fitrsvm(xvTrainX1,xvTrainingDS1((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
xv1YHatFing2 = predict(xv1SVMFing2,xvTestX1);
% Finger 3
xv1SVMFing3 = fitrsvm(xvTrainX1,xvTrainingDS1((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
xv1YHatFing3 = predict(xv1SVMFing3,xvTestX1);
% Finger 4
xv1SVMFing4 = fitrsvm(xvTrainX1,xvTrainingDS1((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
xv1YHatFing4 = predict(xv1SVMFing4,xvTestX1);
% Finger 5
xv1SVMFing5 = fitrsvm(xvTrainX1,xvTrainingDS1((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
xv1YHatFing5 = predict(xv1SVMFing5,xvTestX1);

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove1
lenPredictions = length(xv1YHatFing1);
[xv1Fing1, xv1Fing2, xv1Fing3, xv1Fing4, xv1Fing5] = deal([]);
% Interpolate (should end up with output that is N*50 samples too short)
xv1Fing1 = spline(1:lenPredictions, xv1YHatFing1, 0:1/50:(lenPredictions-1/50));
xv1Fing2 = spline(1:lenPredictions, xv1YHatFing2, 0:1/50:(lenPredictions-1/50));
xv1Fing3 = spline(1:lenPredictions, xv1YHatFing3, 0:1/50:(lenPredictions-1/50));
xv1Fing4 = spline(1:lenPredictions, xv1YHatFing4, 0:1/50:(lenPredictions-1/50));
xv1Fing5 = spline(1:lenPredictions, xv1YHatFing5, 0:1/50:(lenPredictions-1/50));
% Add padding
xv1Fing1 = [zeros(1,N*50) xv1Fing1];
xv1Fing2 = [zeros(1,N*50) xv1Fing2];
xv1Fing3 = [zeros(1,N*50) xv1Fing3];
xv1Fing4 = [zeros(1,N*50) xv1Fing4];
xv1Fing5 = [zeros(1,N*50) xv1Fing5];

% Compute correlation between predicted values and actual values
xv1Fing1Corr = corr(xv1Fing1', xvTestingDataGlove1(:,1));
xv1Fing2Corr = corr(xv1Fing2', xvTestingDataGlove1(:,2));
xv1Fing3Corr = corr(xv1Fing3', xvTestingDataGlove1(:,3));
xv1Fing4Corr = corr(xv1Fing4', xvTestingDataGlove1(:,4));
xv1Fing5Corr = corr(xv1Fing5', xvTestingDataGlove1(:,5));
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

% Inputs into feature and X matrix calculations
numChannels = 48; % for subject 2
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% For training set, calculate features and X matrix
[xv2TrainFeat1, xv2TrainFeat2, xv2TrainFeat3, xv2TrainFeat4, xv2TrainFeat5, xv2TrainFeat6] = ...
    CalcFeatures(xvTrainingECOG2, numChannels, sr, winLen, winDisp);
xvTrainX2 = CalcXMatrix(xv2TrainFeat1, xv2TrainFeat2, xv2TrainFeat3, xv2TrainFeat4, ...
    xv2TrainFeat5, xv2TrainFeat6, N, numChannels);

% For testing set, calculate features and X matrix
[xv2TestFeat1, xv2TestFeat2, xv2TestFeat3, xv2TestFeat4, xv2TestFeat5, xv2TestFeat6] = ...
    CalcFeatures(xvTestingECOG2, numChannels, sr, winLen, winDisp);
xvTestX2 = CalcXMatrix(xv2TestFeat1, xv2TestFeat2, xv2TestFeat3, xv2TestFeat4, ...
    xv2TestFeat5, xv2TestFeat6, N, numChannels);

% Use SVM model for each finger individually
% Finger 1
xv2SVMFing1 = fitrsvm(xvTrainX2,xvTrainingDS2((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
xv2YHatFing1 = predict(xv2SVMFing1,xvTestX2);
% Finger 2
xv2SVMFing2 = fitrsvm(xvTrainX2,xvTrainingDS2((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
xv2YHatFing2 = predict(xv2SVMFing2,xvTestX2);
% Finger 3
xv2SVMFing3 = fitrsvm(xvTrainX2,xvTrainingDS2((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
xv2YHatFing3 = predict(xv2SVMFing3,xvTestX2);
% Finger 4
xv2SVMFing4 = fitrsvm(xvTrainX2,xvTrainingDS2((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
xv2YHatFing4 = predict(xv2SVMFing4,xvTestX2);
% Finger 5
xv2SVMFing5 = fitrsvm(xvTrainX2,xvTrainingDS2((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
xv2YHatFing5 = predict(xv2SVMFing5,xvTestX2);

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove2
lenPredictions = length(xv2YHatFing1);
[xv2Fing1, xv2Fing2, xv2Fing3, xv2Fing4, xv2Fing5] = deal([]);
% Interpolate (should end up with output that is N*50 samples too short)
xv2Fing1 = spline(1:lenPredictions, xv2YHatFing1, 0:1/50:(lenPredictions-1/50));
xv2Fing2 = spline(1:lenPredictions, xv2YHatFing2, 0:1/50:(lenPredictions-1/50));
xv2Fing3 = spline(1:lenPredictions, xv2YHatFing3, 0:1/50:(lenPredictions-1/50));
xv2Fing4 = spline(1:lenPredictions, xv2YHatFing4, 0:1/50:(lenPredictions-1/50));
xv2Fing5 = spline(1:lenPredictions, xv2YHatFing5, 0:1/50:(lenPredictions-1/50));
% Add padding
xv2Fing1 = [zeros(1,N*50) xv2Fing1];
xv2Fing2 = [zeros(1,N*50) xv2Fing2];
xv2Fing3 = [zeros(1,N*50) xv2Fing3];
xv2Fing4 = [zeros(1,N*50) xv2Fing4];
xv2Fing5 = [zeros(1,N*50) xv2Fing5];

% Compute correlation between predicted values and actual values
xv2Fing1Corr = corr(xv2Fing1', xvTestingDataGlove2(:,1));
xv2Fing2Corr = corr(xv2Fing2', xvTestingDataGlove2(:,2));
xv2Fing3Corr = corr(xv2Fing3', xvTestingDataGlove2(:,3));
xv2Fing4Corr = corr(xv2Fing4', xvTestingDataGlove2(:,4));
xv2Fing5Corr = corr(xv2Fing5', xvTestingDataGlove2(:,5));
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

% Inputs into feature and X matrix calculations
numChannels = 64; % for subject 3
sr = 1000; % sample rate
winLen = 100/1e3; % 100ms window
winDisp = 50/1e3; % 50ms displacement
N = 3; % number of time bins before

% For training set, calculate features and X matrix
[xv3TrainFeat1, xv3TrainFeat2, xv3TrainFeat3, xv3TrainFeat4, xv3TrainFeat5, xv3TrainFeat6] = ...
    CalcFeatures(xvTrainingECOG3, numChannels, sr, winLen, winDisp);
xvTrainX3 = CalcXMatrix(xv3TrainFeat1, xv3TrainFeat2, xv3TrainFeat3, xv3TrainFeat4, ...
    xv3TrainFeat5, xv3TrainFeat6, N, numChannels);

% For testing set, calculate features and X matrix
[xv3TestFeat1, xv3TestFeat2, xv3TestFeat3, xv3TestFeat4, xv3TestFeat5, xv3TestFeat6] = ...
    CalcFeatures(xvTestingECOG3, numChannels, sr, winLen, winDisp);
xvTestX3 = CalcXMatrix(xv3TestFeat1, xv3TestFeat2, xv3TestFeat3, xv3TestFeat4, ...
    xv3TestFeat5, xv3TestFeat6, N, numChannels);

% Use SVM model for each finger individually
% Finger 1
xv3SVMFing1 = fitrsvm(xvTrainX3,xvTrainingDS3((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
xv3YHatFing1 = predict(xv3SVMFing1,xvTestX3);
% Finger 2
xv3SVMFing2 = fitrsvm(xvTrainX3,xvTrainingDS3((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
xv3YHatFing2 = predict(xv3SVMFing2,xvTestX3);
% Finger 3
xv3SVMFing3 = fitrsvm(xvTrainX3,xvTrainingDS3((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
xv3YHatFing3 = predict(xv3SVMFing3,xvTestX3);
% Finger 4
xv3SVMFing4 = fitrsvm(xvTrainX3,xvTrainingDS3((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
xv3YHatFing4 = predict(xv3SVMFing4,xvTestX3);
% Finger 5
xv3SVMFing5 = fitrsvm(xvTrainX3,xvTrainingDS3((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
xv3YHatFing5 = predict(xv3SVMFing5,xvTestX3);

% Interpolate and pad predictions to get them to be the right length to
% compare with xvTestingDataGlove3
lenPredictions = length(xv3YHatFing1);
[xv3Fing1, xv3Fing2, xv3Fing3, xv3Fing4, xv3Fing5] = deal([]);
% Interpolate (should end up with output that is N*50 samples too short)
xv3Fing1 = spline(1:lenPredictions, xv3YHatFing1, 0:1/50:(lenPredictions-1/50));
xv3Fing2 = spline(1:lenPredictions, xv3YHatFing2, 0:1/50:(lenPredictions-1/50));
xv3Fing3 = spline(1:lenPredictions, xv3YHatFing3, 0:1/50:(lenPredictions-1/50));
xv3Fing4 = spline(1:lenPredictions, xv3YHatFing4, 0:1/50:(lenPredictions-1/50));
xv3Fing5 = spline(1:lenPredictions, xv3YHatFing5, 0:1/50:(lenPredictions-1/50));
% Add padding
xv3Fing1 = [zeros(1,N*50) xv3Fing1];
xv3Fing2 = [zeros(1,N*50) xv3Fing2];
xv3Fing3 = [zeros(1,N*50) xv3Fing3];
xv3Fing4 = [zeros(1,N*50) xv3Fing4];
xv3Fing5 = [zeros(1,N*50) xv3Fing5];

% Compute correlation between predicted values and actual values
xv3Fing1Corr = corr(xv3Fing1', xvTestingDataGlove3(:,1));
xv3Fing2Corr = corr(xv3Fing2', xvTestingDataGlove3(:,2));
xv3Fing3Corr = corr(xv3Fing3', xvTestingDataGlove3(:,3));
xv3Fing4Corr = corr(xv3Fing4', xvTestingDataGlove3(:,4));
xv3Fing5Corr = corr(xv3Fing5', xvTestingDataGlove3(:,5));
% Compute average correlation
xv3Corr = mean([xv3Fing1Corr xv3Fing2Corr xv3Fing3Corr xv3Fing5Corr])






%% Moving average (post-processing)

sr = 1000; % sample rate
winLen = 150/1e3; % 100ms window
winSamples = winLen*sr; % number of samples in window

% filter coefficient vectors
a = 1;
b = ones(1,winSamples)/winSamples;

% filter all finger data
% Subject 1
xv1Fing1Filtered = filtfilt(b,a,xv1Fing1);
xv1Fing2Filtered = filtfilt(b,a,xv1Fing2);
xv1Fing3Filtered = filtfilt(b,a,xv1Fing3);
xv1Fing4Filtered = filtfilt(b,a,xv1Fing4);
xv1Fing5Filtered = filtfilt(b,a,xv1Fing5);
% Subject 2
xv2Fing1Filtered = filtfilt(b,a,xv2Fing1);
xv2Fing2Filtered = filtfilt(b,a,xv2Fing2);
xv2Fing3Filtered = filtfilt(b,a,xv2Fing3);
xv2Fing4Filtered = filtfilt(b,a,xv2Fing4);
xv2Fing5Filtered = filtfilt(b,a,xv2Fing5);
% Subject 3
xv3Fing1Filtered = filtfilt(b,a,xv3Fing1);
xv3Fing2Filtered = filtfilt(b,a,xv3Fing2);
xv3Fing3Filtered = filtfilt(b,a,xv3Fing3);
xv3Fing4Filtered = filtfilt(b,a,xv3Fing4);
xv3Fing5Filtered = filtfilt(b,a,xv3Fing5);


%% New correlations (filtered)

% Subject 1
% Compute correlation between predicted/filtered values and actual values
xv1Fing1FilteredCorr = corr(xv1Fing1Filtered', xvTestingDataGlove1(:,1));
xv1Fing2FilteredCorr = corr(xv1Fing2Filtered', xvTestingDataGlove1(:,2));
xv1Fing3FilteredCorr = corr(xv1Fing3Filtered', xvTestingDataGlove1(:,3));
xv1Fing4FilteredCorr = corr(xv1Fing4Filtered', xvTestingDataGlove1(:,4));
xv1Fing5FilteredCorr = corr(xv1Fing5Filtered', xvTestingDataGlove1(:,5));
% Compute average correlation
xv1CorrFiltered = mean([xv1Fing1FilteredCorr xv1Fing2FilteredCorr...
    xv1Fing3FilteredCorr xv1Fing5FilteredCorr])

% Subject 2
% Compute correlation between predicted values and actual values
xv2Fing1FilteredCorr = corr(xv2Fing1Filtered', xvTestingDataGlove2(:,1));
xv2Fing2FilteredCorr = corr(xv2Fing2Filtered', xvTestingDataGlove2(:,2));
xv2Fing3FilteredCorr = corr(xv2Fing3Filtered', xvTestingDataGlove2(:,3));
xv2Fing4FilteredCorr = corr(xv2Fing4Filtered', xvTestingDataGlove2(:,4));
xv2Fing5FilteredCorr = corr(xv2Fing5Filtered', xvTestingDataGlove2(:,5));
% Compute average correlation
xv2CorrFiltered = mean([xv2Fing1FilteredCorr xv2Fing2FilteredCorr...
    xv2Fing3FilteredCorr xv2Fing5FilteredCorr])

% Subject 3
xv3Fing1FilteredCorr = corr(xv3Fing1Filtered', xvTestingDataGlove3(:,1));
xv3Fing2FilteredCorr = corr(xv3Fing2Filtered', xvTestingDataGlove3(:,2));
xv3Fing3FilteredCorr = corr(xv3Fing3Filtered', xvTestingDataGlove3(:,3));
xv3Fing4FilteredCorr = corr(xv3Fing4Filtered', xvTestingDataGlove3(:,4));
xv3Fing5FilteredCorr = corr(xv3Fing5Filtered', xvTestingDataGlove3(:,5));
% Compute average correlation
xv3CorrFiltered = mean([xv3Fing1FilteredCorr xv3Fing2FilteredCorr...
    xv3Fing3FilteredCorr xv3Fing5FilteredCorr])


%% Overall cross-validation score (filtered)
xvCorrFiltered = mean([xv3Fing1FilteredCorr xv3Fing2FilteredCorr xv3Fing3FilteredCorr...
    xv3Fing5FilteredCorr xv2Fing1FilteredCorr xv2Fing2FilteredCorr xv2Fing3FilteredCorr...
    xv2Fing5FilteredCorr xv1Fing1FilteredCorr xv1Fing2FilteredCorr xv1Fing3FilteredCorr...
    xv1Fing5FilteredCorr])




%% Overall cross-validation score
xvCorr = mean([xv3Fing1Corr xv3Fing2Corr xv3Fing3Corr xv3Fing5Corr xv2Fing1Corr...
    xv2Fing2Corr xv2Fing3Corr xv2Fing5Corr xv1Fing1Corr xv1Fing2Corr ...
    xv1Fing3Corr xv1Fing5Corr])


%% Plot original versus predicted versus predicted/filtered

% Subject 1 finger 1
figure
hold on
plot(xvTestingDataGlove1(:,2))
%plot(xv1Fing2)
plot(xv1Fing2Filtered)
hold off