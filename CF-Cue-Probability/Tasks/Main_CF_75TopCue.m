%% author: Matt & Haiyang; updated by Luyan, Jan 7, 2020
% face images are divided into 10 groups (5 for each gender)
clear all;
experimentAbbv = 'CF_Complete_75TopCue';

%if ~exist('theParticipant','var'), theParticipant = []; end
%while isempty(theParticipant)
%    theParticipant = input('Please enter the participant name: ','s');
%end

try 
%Normally, only the statements between the try and CATCH are executed.
%However, if an error occurs while executing any of the statements, the error is captured into an object, ME, of class MException, and the statements between the CATCH and END are executed. 
%% experiment inforamtion input
prompt = {'Enter the participant code: ', ...
    'Enter the age of participant:',...
    'Enter the gender of participant',...
    'Enter the ethnicity of participant: ', ... % (Caucasian: 1, Chinese: 2)
    'Enter the ethnicity of stimuli used: (Caucasian: 1, Chinese: 2)'};
title = 'Please input the related information';
dims = [1 50];
definput = {'000', '18','female','2', '2'};
inputs = inputdlg(prompt, title, dims, definput);

participantNum = inputs{1};
participantAge = inputs{2};
participantGender = inputs{3};
participantRace = str2double(inputs{4});
stimRace = str2double(inputs{5});

%% Load files
Initialize; % startup the screen.
ExperimentVars_CF;
GlobalVars_CF; % load some shared variables.

%% Experiment Design
% clear conditionsArray;
% conditionsArray = {...
%    'isAligned', 0:1; ... % 0 = misaligned; 1 = aligned
%    'topIsCued', 0:1;... %  0 = bottom is cued; 1 = top is cued
%    'isCongruent', 0:1;...  % 0 = incongruent; 1 = congruent
%    'cuedIsSame', 0:1; ... % 0 = different; 1 = same
%    'faceGroup', 1:10; ... % 10 groups in total   'F1','F2','F3','F4','F5','M1','M2','M3','M4','M5'
%    'faceIndex', 1:4;... % 4 images in each group
%    'withinBlockReps', 1; ... % 
%    'blockNumber', 0:1; ... % 
%    };
% blockByCondition = 'blockNumber'; % Which condition is used to block the trials? (balance the randomization so that unique conditions appear in different blocks). just use doublequotes (='') if you don't want to use.
% ed = BuildExperimentDesign(conditionsArray,blockByCondition);

% prepare stimuli list in advance, read it into the variable design_table 
design_table = readtable('StimuliList_0.75TopCue.csv');
% randomize and sort by sortCondition (Probability).
sortCondition = 'BlockNumber';
design_table = sortrows( design_table(randperm(size(design_table,1)),:) ,...  % randperm: Random permutation
    find(strcmpi(design_table.Properties.VariableNames,sortCondition)));  % strcmpi: Compare strings or character vectors ignoring case.
                                                                          % find: Find indices of nonzero elements.
                                                                          % sortrows: Sort rows of a matrix. sortrows(A,COL) sorts the mtatrix according to the columns specified by the vector COL.

% convert table to structure and create ed
ed = table2struct(design_table);
                                                                          
facesPerGroup = 4;
numTrials = size(ed,1);  % 320

%% Face Selection
% select face based on condition
% columns: studyTop; studyBottom; targetTop; targetBottom
% rows: top(CS,CD,IS,ID), bottom(CS,CD,IS,ID)                %% SS, DD, SD, DS
faceSelector = ...
    [0 1 0 1 ; % TCS
     0 1 2 3 ; % TCD
     0 1 0 2 ; % TIS
     0 1 3 1 ; % TID
     0 1 0 1 ; % BCS
     0 1 2 3 ; % BCD
     0 1 3 1 ; % BIS
     0 1 0 2]; % BID
% Usage: 
%   trialType = 1 + 4*(1-ed(ttn).topIsCued) + 2*(1-ed(ttn).isCongruent) + (1-ed(ttn).cuedIsSame);
%   thisFaceSet = mod((ed(ttn).faceIndex + faceSelector(trialType,:)-1),4)+1;

%% Instruction
%Screen('DrawText',window,instructionText,screenX/2,screenY/2,instructionColor);
DrawFormattedText(window,instructionText,'center','center',instructionColor);
Screen('Flip',window);
RestrictKeysForKbCheck(instructionKey);
KbWait([],2)   

% block instruction
DrawFormattedText(window,instructionText_cue,'center','center',instructionColor);
Screen('Flip',window);
RestrictKeysForKbCheck(instructionKey);
KbWait([],2)   

%% the bulk of the experiment happens here. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear allData;
allData = cell(numTrials+1,10);
expStartTime = GetSecs();

for ttn = 1:numTrials
    DoTrial_CF;
    
    if (quitNow), break; end
    CheckForBreak;

end

expEndTime = GetSecs();

%% Finished; saving data and cleanup.
if (~quitNow)
    doneText = sprintf('This part is finished!\n \nThanks for participating.');
    DrawFormattedText(window,doneText,'center','center',instructionColor);
    Screen('Flip',window);
    KbWait([],3);
    clear doneText;
end
        
Screen('CloseAll');
if ~exist('Data_trial','var')
    Data_trial = cell2struct(allData(2:end,:), allData(1,:), 2);
end
save(theMatlabFile);
xlswrite(theExcelFile,allData);

disp(['Mean Accuracy: ' num2str(100*mean([Data_trial(:).isCorrect]),'%2.1f%%')]);

catch error
    sca;
    rethrow(error);
end
