function [predicted_dg] = make_predictions(test_ecog)

%
% Inputs: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.
% Outputs: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)
% Run time: The script has to run less than 1 hour.
%
% The following is a sample script.

% Load Model
% Imagine this mat file has the following variables:
% winDisp, filtTPs, trainFeats (cell array), 

N = 3;
numC{1} = 62;
numC{2} = 48;
numC{3} = 64;
sr = 1000;
winLen = 0.1
winDisp = 0.05;

[sub1Feat1T, sub1Feat2T, sub1Feat3T, sub1Feat4T, sub1Feat5T, sub1Feat6T] = ...
    CalcFeatures(test_ecog{1}, numC{1}, sr, winLen, winDisp);
testX{1} = CalcXMatrix(sub1Feat1T, sub1Feat2T, sub1Feat3T, sub1Feat4T, sub1Feat5T,...
    sub1Feat6T, N, numC{1});

[sub2Feat1T, sub2Feat2T, sub2Feat3T, sub2Feat4T, sub2Feat5T, sub2Feat6T] = ...
    CalcFeatures(test_ecog{2}, numC{2}, sr, winLen, winDisp);
testX{2} = CalcXMatrix(sub2Feat1T, sub2Feat2T, sub2Feat3T, sub2Feat4T, sub2Feat5T,...
    sub2Feat6T, N, numC{2});

[sub3Feat1T, sub3Feat2T, sub3Feat3T, sub3Feat4T, sub3Feat5T, sub3Feat6T] = ...
    CalcFeatures(test_ecog{3}, numC{3}, sr, winLen, winDisp);
testX{3} = CalcXMatrix(sub3Feat1T, sub3Feat2T, sub3Feat3T, sub3Feat4T, sub3Feat5T,...
    sub3Feat6T, N, numC{3});


%load weights for each subject and each finger
%w is a 3 x 5 cell array, containing the weights for each subject per row,
%and model for each finger per column

% Predict using linear predictor for each subject
%create cell array with one element for each subject
predicted_dg = cell(3,1);

%for each subject
for subj = 1:3 
    
    %get the testing ecog
    testset = testX{subj}; 
    
    %initialize the predicted dataglove matrix
    %yhat = zeros(size(testset,1),5);
    
    %for each finger
    for i = 1:5 
        
        numChannels = numC{subj};
        fname = strcat( 'svm', num2str(subj), 'Fing', num2str(1) );
        svmmodel = load(fname);
        cellversionmodel = struct2cell(svmmodel);
        yhatFing{i} = predict(cellversionmodel{1},testset);
        
    end
    predicted_dg{subj} = [yhatFing{1}' yhatFing{2}' yhatFing{3}' yhatFing{4}' yhatFing{5}'];
    clear yhatFing
end

