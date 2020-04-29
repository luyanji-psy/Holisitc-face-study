%% instruction
if mod(str2double(participantNum), 2)
    sameKey = '1';
    SDkeys = 'SD';
else
    sameKey = '2';
    SDkeys = 'DS';
end
keys = '12';
diffKey = keys(keys ~= sameKey);

instructionKey = KbName('q');
instructionColor = 255;
cue_part = experimentAbbv(15:end-3);

instructionText = sprintf(['Welcome to this experiment.'...
    '\n\n\n'...
    'On each trial, you will see two faces, one after the other.'...
    '\n \n'...
    'There will be a cue for the second face'...
    '\n \n'...
    'Please focus on the whole first face and the cued part of the second face.'...
    '\n \n\n \n'...
    'If the cued part of the second face is the same as the first face,'...
    '\n\n'...
    'please press KEY %s.'...
    '\n\n\n \n'...
    'If the cued part of the second face is different from the first face,'...
    '\n\n'...
    'please press KEY %s.'...
    '\n \n \n \n'...
    'Please respond as quickly and accurately as possible.'],sameKey, diffKey); 

instructionText_cue = sprintf(['In this part, there will be 75 percentage of chances that the %s part of face is cued, '...
    '\n \n'...    
    'and needs to be compared.'...
    '\n \n \n \n'...
    'When you are ready, press Q to start.'],cue_part); 


%% fixation point
fixationColor = [255 255 255];
fixationDiameter = 10;
fixationDuration = 0.5;
fixationRect = CenterRect([0 0 fixationDiameter fixationDiameter],screenRect);

%% other variables
ITInterval = rand*0.2+1; % time after test/study to fixation
blankDuration = 0.2;
%blank2Duration = 10; % time after test face, for collecting responses
studyDuration = .2;
maskDuration = .5;
targetDuration = .2;
testFixationDuration = .75; % how fixation time before each object (sec)

responseTimeLimit = 10;
forceRestPeriod = 1; % session is divided into blocks with a rest period
restMinimumTime = 30; % minimum rest time in seconds.
trialsPerRest = 64;  % have a rest after every 64 trials; 10 blocks for 640 trials

%% Keys
experimenterKey = KbName('F12');
responseKeys = [KbName('1') KbName('2')];
%responseKeys = [KbName('1') KbName('2')...
%                KbName('1!') KbName('2@')];
            
noResponseText = sprintf('Something wrong happened. Press a key.');

%% Face Stims
numFacesPerGender = 20;
faceGenders = 'MF';
Stimuli;
faceGroupName = {'F1','F2','F3','F4','F5','M1','M2','M3','M4','M5'};
faceAlpha = 1;

%% Other Conditions
congruency = 'CI';
alignment = 'AM';
sameDifferent = 'SD';
cuedHalf = 'TB';

%% Mask stimuli
[rgb, ~, alpha] = imread('mask.bmp');
maskDestRect = CenterRect([0 0 size(rgb,2) size(rgb,1)],screenRect);
maskTexture = Screen('MakeTexture',window,cat(3,rgb,alpha));
