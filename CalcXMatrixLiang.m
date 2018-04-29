% This function calculates the X feature matrix for use in a linear
% regression (optimal linear decoder) model. 
%
% Inputs are:
% - band1Feat = sub band (1-60Hz) features; make sure that the number of 
% rows of 'band1Feat' equals 'numChannels'
% - band2Feat = gamma band (60-100Hz) features; make sure that the number of 
% rows of 'band2Feat' equals 'numChannels'
% - band3Feat = fast gamma band (100-200Hz) features; make sure that the number of 
% rows of 'band3Feat' equals 'numChannels'
% - nTimeBins = the number of time bins preceding the current moment
% - numChannels = the number of channels for this subject
%
% Output is: X matrix

function x = CalcXMatrixLiang(band1Feat,band2Feat,band3Feat,nTimeBins,numChannels)

    N = nTimeBins; % number of time bins before
    M = size(band1Feat,2)-N+1; % number of rows for X matrix
    v = numChannels; % number of channels for subject
    nF = 3; % number of features
    nC = v*N*nF+1; % number of columns
    x = []; % empty X matrix
    % Construct X matrix
    for i = 1:M

        currRow = [1];

        % Iterate through each channel
        for j = 1:v

            % Add N values from each feature matrix, at row j
            currRow = [currRow band1Feat(j,i:(i+N-1))];
            currRow = [currRow band2Feat(j,i:(i+N-1))];
            currRow = [currRow band3Feat(j,i:(i+N-1))];

        end

        % Add current row to matrix
        x = [x; currRow];

    end
    
end