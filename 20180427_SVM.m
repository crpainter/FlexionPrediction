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

% Four steps
% (1/4): Calculate features and X matrix for training data
% (2/4): Calculate features and X matrix for testing data
% (3/4): Train SVM model for each finger and make predictions
% (4/4): Interpolate and pad predictions to produce output of the correct
% length for each subject


%% Subject 1 (1/4): calculate features and X matrix for training data

% Inputs for feature and X matrix calculations
numChannels = 62; % for subject 1
sr = 1000; % 1000 Hz
winLen = 100/1e3; % window length is 100ms
winDisp = 50/1e3; % window displacement is 50ms
N = 3; % number of time bins before

% Calculate features and X matrix
% The six features we are using are the same as what are recommended in the
% assignment:
% 1) average time-domain voltage
% 2) average frequency-domain magnitude in the 5-15 Hz frequency band
% 3) average frequency-domain magnitude in the 20-25 Hz frequency band
% 4) average frequency-domain magnitude in the 75-115 Hz frequency band
% 5) average frequency-domain magnitude in the 125-160 Hz frequency band
% 6) average frequency-domain magnitude in the 160-175 Hz frequency band
[sub1Feat1, sub1Feat2, sub1Feat3, sub1Feat4, sub1Feat5, sub1Feat6] = ...
    CalcFeatures(ecogData1, numChannels, sr, winLen, winDisp);
trainX1 = CalcXMatrix(sub1Feat1, sub1Feat2, sub1Feat3, sub1Feat4, sub1Feat5, ...
    sub1Feat6, N, numChannels);


%% Subject 1 (2/4): calculate features and X matrix for testing data

[sub1Feat1T, sub1Feat2T, sub1Feat3T, sub1Feat4T, sub1Feat5T, sub1Feat6T] = ...
    CalcFeatures(testECOG1, numChannels, sr, winLen, winDisp);
testX1 = CalcXMatrix(sub1Feat1T, sub1Feat2T, sub1Feat3T, sub1Feat4T, sub1Feat5T,...
    sub1Feat6T, N, numChannels);


%% Subject 1 (3/4): train SVM model for each finger and make predictions

% Use SVM model for each finger individually
% Finger 1
svm1Fing1 = fitrsvm(trainX1,dsDG1((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
yhat1Fing1 = predict(svm1Fing1,testX1);
% Finger 2
svm1Fing2 = fitrsvm(trainX1,dsDG1((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
yhat1Fing2 = predict(svm1Fing2,testX1);
% Finger 3
svm1Fing3 = fitrsvm(trainX1,dsDG1((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
yhat1Fing3 = predict(svm1Fing3,testX1);
% Finger 4
svm1Fing4 = fitrsvm(trainX1,dsDG1((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
yhat1Fing4 = predict(svm1Fing4,testX1);
% Finger 5
svm1Fing5 = fitrsvm(trainX1,dsDG1((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',6,'KernelScale','auto','Standardize',true);
yhat1Fing5 = predict(svm1Fing5,testX1);


%% Subject 1 (4/4): interpolate and pad predictions to produce final output for subject 1

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat1Fing1);
[sub1Fing1, sub1Fing2, sub1Fing3, sub1Fing4, sub1Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub1Fing1 = spline(1:lenPredictions, yhat1Fing1, 0:1/50:(lenPredictions-1/50));
sub1Fing2 = spline(1:lenPredictions, yhat1Fing2, 0:1/50:(lenPredictions-1/50));
sub1Fing3 = spline(1:lenPredictions, yhat1Fing3, 0:1/50:(lenPredictions-1/50));
sub1Fing4 = spline(1:lenPredictions, yhat1Fing4, 0:1/50:(lenPredictions-1/50));
sub1Fing5 = spline(1:lenPredictions, yhat1Fing5, 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 1
sub1Fing1 = [zeros(1,N*50) sub1Fing1];
sub1Fing2 = [zeros(1,N*50) sub1Fing2];
sub1Fing3 = [zeros(1,N*50) sub1Fing3];
sub1Fing4 = [zeros(1,N*50) sub1Fing4];
sub1Fing5 = [zeros(1,N*50) sub1Fing5];






%% Subject 2 (1/4): calculate features and X matrix for training data

% Inputs for feature and X matrix calculations
numChannels = 48; % for subject 2
sr = 1000; % 1000 Hz
winLen = 100/1e3; % window length is 100ms
winDisp = 50/1e3; % window displacement is 50ms
N = 3; % number of time bins before

% Calculate features and X matrix
[sub2Feat1, sub2Feat2, sub2Feat3, sub2Feat4, sub2Feat5, sub2Feat6] = ...
    CalcFeatures(ecogData2, numChannels, sr, winLen, winDisp);
trainX2 = CalcXMatrix(sub2Feat1, sub2Feat2, sub2Feat3, sub2Feat4, sub2Feat5, ...
    sub2Feat6, N, numChannels);


%% Subject 2 (2/4): calculate features and X matrix for testing data

[sub2Feat1T, sub2Feat2T, sub2Feat3T, sub2Feat4T, sub2Feat5T, sub2Feat6T] = ...
    CalcFeatures(testECOG2, numChannels, sr, winLen, winDisp);
testX2 = CalcXMatrix(sub2Feat1T, sub2Feat2T, sub2Feat3T, sub2Feat4T, sub2Feat5T,...
    sub2Feat6T, N, numChannels);


%% Subject 2 (3/4): train SVM model for each finger and make predictions

% Use SVM model for each finger individually
% Finger 1
svm2Fing1 = fitrsvm(trainX2,dsDG2((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
yhat2Fing1 = predict(svm2Fing1,testX2);
% Finger 2
svm2Fing2 = fitrsvm(trainX2,dsDG2((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
yhat2Fing2 = predict(svm2Fing2,testX2);
% Finger 3
svm2Fing3 = fitrsvm(trainX2,dsDG2((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
yhat2Fing3 = predict(svm2Fing3,testX2);
% Finger 4
svm2Fing4 = fitrsvm(trainX2,dsDG2((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
yhat2Fing4 = predict(svm2Fing4,testX2);
% Finger 5
svm2Fing5 = fitrsvm(trainX2,dsDG2((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',4,'KernelScale','auto','Standardize',true);
yhat2Fing5 = predict(svm2Fing5,testX2);


%% Subject 2 (4/4): interpolate and pad predictions to produce final output for subject 2

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat2Fing1);
[sub2Fing1, sub2Fing2, sub2Fing3, sub2Fing4, sub2Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub2Fing1 = spline(1:lenPredictions, yhat2Fing1, 0:1/50:(lenPredictions-1/50));
sub2Fing2 = spline(1:lenPredictions, yhat2Fing2, 0:1/50:(lenPredictions-1/50));
sub2Fing3 = spline(1:lenPredictions, yhat2Fing3, 0:1/50:(lenPredictions-1/50));
sub2Fing4 = spline(1:lenPredictions, yhat2Fing4, 0:1/50:(lenPredictions-1/50));
sub2Fing5 = spline(1:lenPredictions, yhat2Fing5, 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 1
sub2Fing1 = [zeros(1,N*50) sub2Fing1];
sub2Fing2 = [zeros(1,N*50) sub2Fing2];
sub2Fing3 = [zeros(1,N*50) sub2Fing3];
sub2Fing4 = [zeros(1,N*50) sub2Fing4];
sub2Fing5 = [zeros(1,N*50) sub2Fing5];





%% Subject 3 (1/4): calculate features and X matrix for training data

% Inputs for feature and X matrix calculations
numChannels = 64; % for subject 3
sr = 1000; % 1000 Hz
winLen = 100/1e3; % window length is 100ms
winDisp = 50/1e3; % window displacement is 50ms
N = 3; % number of time bins before

% Calculate features and X matrix
[sub3Feat1, sub3Feat2, sub3Feat3, sub3Feat4, sub3Feat5, sub3Feat6] = ...
    CalcFeatures(ecogData3, numChannels, sr, winLen, winDisp);
trainX3 = CalcXMatrix(sub3Feat1, sub3Feat2, sub3Feat3, sub3Feat4, sub3Feat5, ...
    sub3Feat6, N, numChannels);


%% Subject 3 (2/4): calculate features and X matrix for testing data

[sub3Feat1T, sub3Feat2T, sub3Feat3T, sub3Feat4T, sub3Feat5T, sub3Feat6T] = ...
    CalcFeatures(testECOG3, numChannels, sr, winLen, winDisp);
testX3 = CalcXMatrix(sub3Feat1T, sub3Feat2T, sub3Feat3T, sub3Feat4T, sub3Feat5T,...
    sub3Feat6T, N, numChannels);


%% Subject 3 (3/4): train SVM model for each finger and make predictions

% Use SVM model for each finger individually
% Finger 1
svm3Fing1 = fitrsvm(trainX3,dsDG3((N+1):end,1),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
yhat3Fing1 = predict(svm3Fing1,testX3);
% Finger 2
svm3Fing2 = fitrsvm(trainX3,dsDG3((N+1):end,2),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
yhat3Fing2 = predict(svm3Fing2,testX3);
% Finger 3
svm3Fing3 = fitrsvm(trainX3,dsDG3((N+1):end,3),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
yhat3Fing3 = predict(svm3Fing3,testX3);
% Finger 4
svm3Fing4 = fitrsvm(trainX3,dsDG3((N+1):end,4),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
yhat3Fing4 = predict(svm3Fing4,testX3);
% Finger 5
svm3Fing5 = fitrsvm(trainX3,dsDG3((N+1):end,5),'KernelFunction',...
    'polynomial','PolynomialOrder',7,'KernelScale','auto','Standardize',true);
yhat3Fing5 = predict(svm3Fing5,testX3);


%% Subject 3 (4/4): interpolate and pad predictions to produce final output for subject 3

% Interpolate and pad predictions to get them to be the right length
lenPredictions = length(yhat3Fing1);
[sub3Fing1, sub3Fing2, sub3Fing3, sub3Fing4, sub3Fing5] = deal([]); 
% Interpolate (should end up with output that is N*50 samples too short)
sub3Fing1 = spline(1:lenPredictions, yhat3Fing1, 0:1/50:(lenPredictions-1/50));
sub3Fing2 = spline(1:lenPredictions, yhat3Fing2, 0:1/50:(lenPredictions-1/50));
sub3Fing3 = spline(1:lenPredictions, yhat3Fing3, 0:1/50:(lenPredictions-1/50));
sub3Fing4 = spline(1:lenPredictions, yhat3Fing4, 0:1/50:(lenPredictions-1/50));
sub3Fing5 = spline(1:lenPredictions, yhat3Fing5, 0:1/50:(lenPredictions-1/50));
% Add padding to produce final output for subject 1
sub3Fing1 = [zeros(1,N*50) sub3Fing1];
sub3Fing2 = [zeros(1,N*50) sub3Fing2];
sub3Fing3 = [zeros(1,N*50) sub3Fing3];
sub3Fing4 = [zeros(1,N*50) sub3Fing4];
sub3Fing5 = [zeros(1,N*50) sub3Fing5];







%% Combine output from three subjects into one variable (predicted_dg)

predicted_dg = {};
predicted_dg{1} = [sub1Fing1' sub1Fing2' sub1Fing3' sub1Fing4' sub1Fing5'];
predicted_dg{2} = [sub2Fing1' sub2Fing2' sub2Fing3' sub2Fing4' sub2Fing5'];
predicted_dg{3} = [sub3Fing1' sub3Fing2' sub3Fing3' sub3Fing4' sub3Fing5'];
predicted_dg = predicted_dg';

% Save predicted_dg to .mat file for submission
save('readyPlayerOne_predictions.mat','predicted_dg');


%% Save training features and X matrices

save('trainingFeatsX.mat','sub1Feat1','sub1Feat2','sub1Feat3','sub1Feat4','sub1Feat5','sub1Feat6',...
    'sub2Feat1','sub2Feat2','sub2Feat3','sub2Feat4','sub2Feat5','sub2Feat6',...
    'sub3Feat1','sub3Feat2','sub3Feat3','sub3Feat4','sub3Feat5','sub3Feat6',...
    'trainX1','trainX2','trainX3');


%% Save testing features and X matrices

save('testingFeatsX.mat','sub1Feat1T','sub1Feat2T','sub1Feat3T','sub1Feat4T','sub1Feat5T','sub1Feat6T',...
    'sub2Feat1T','sub2Feat2T','sub2Feat3T','sub2Feat4T','sub2Feat5T','sub2Feat6T',...
    'sub3Feat1T','sub3Feat2T','sub3Feat3T','sub3Feat4T','sub3Feat5T','sub3Feat6T',...
    'testX1','testX2','testX3');


%% Save workspace

save finalProjectAll.mat;
