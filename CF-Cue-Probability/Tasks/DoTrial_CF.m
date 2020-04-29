%% ITI - do preparation work for the trial.
trialBeginsAt = GetSecs;
WaitSecs(ITInterval);

%% Conditions generation

%topIscued	isCongruent	cuedIsSame	TrialType
%1	1	1	1
%1	1	0	2
%1	0	1	3
%1	0	0	4
%0	1	1	5
%0	1	0	6
%0	0	1	7
%0	0	0	8

trialType = 1 + 4*(1-ed(ttn).topIsCued) + 2*(1-ed(ttn).isCongruent) + (1-ed(ttn).cuedIsSame);
thisFaceSet = mod((ed(ttn).faceIndex + faceSelector(trialType,:)-1),4)+1;

%% draw fixation and blank
Screen('DrawArc',window,fixationColor,fixationRect,0,360);
thisFixBegins = Screen('Flip',window);
WaitSecs(fixationDuration - flipSlack);

%% blank
Screen('Flip',window);
WaitSecs(blankDuration - flipSlack);

%% Study Face
% study face was always aligned
Screen('DrawTexture',window,faces(thisFaceSet(1),ed(ttn).faceGroup).texture,[0 0 200 128] ,faceTopRect,[],[],faceAlpha); %OffsetRect(faceTopRect,100*(1-ed(ttn).isAligned)*(1-ed(ttn).topIsCued),0)
Screen('DrawTexture',window,faces(thisFaceSet(2),ed(ttn).faceGroup).texture,[0 128 200 256] ,faceBottomRect,[],[],faceAlpha); % OffsetRect(faceBottomRect,100*(1-ed(ttn).isAligned)*ed(ttn).topIsCued,0)
Screen('FillRect',window,[255 255 255],lineRect);

Screen('Flip',window);
WaitSecs(studyDuration - flipSlack);

%% mask
Screen('DrawTexture',window,maskTexture,[],maskDestRect);
Screen('Flip',window);
WaitSecs(maskDuration - flipSlack);

%% target face
% aligned condition, target face was aligned
% misaligned condition, the uncued part of the face was shifted
xOffsetRand = (randperm(7,1)-4)*5;    % -15 ~ 15  
yOffsetRand = 0; %(randperm(7,1)-4)*5;

% newRect = OffsetRect(oldRect,x,y). Offset the passed rect matrix by the horizontal (x) and vertical (y) shift given. 
% aligned = 1,top x shift = xOffsetRand, bottom x shift = xOffsetRand; 
% aligned = 0, topIsCued = 1, top x shift = xOffsetRand (the cued part remains the same position), bottom x shift = 100 + xOffsetRand (the uncued part shifted); 
% aligned = 0, topIsCued = 0, top x shift = 100 + xOffsetRand, bottom x shift = xOffsetRand.
Screen('DrawTexture',window,faces(thisFaceSet(3),ed(ttn).faceGroup).texture,[0 0 200 128] ,OffsetRect(faceTopRect,100*(1-ed(ttn).isAligned)*(1-ed(ttn).topIsCued)+xOffsetRand,yOffsetRand),[],[],faceAlpha);
Screen('DrawTexture',window,faces(thisFaceSet(4),ed(ttn).faceGroup).texture,[0 128 200 256] ,OffsetRect(faceBottomRect,100*(1-ed(ttn).isAligned)*ed(ttn).topIsCued+xOffsetRand,yOffsetRand),[],[],faceAlpha);
Screen('FillRect',window,[255 255 255],OffsetRect(lineRect,xOffsetRand,yOffsetRand));

% cue
Screen('FillRect',window,[255 255 255],OffsetRect(cueRect,xOffsetRand,yOffsetRand+(-2*ed(ttn).topIsCued + 1)*cueDistance));
Screen('FillRect',window,[255 255 255],OffsetRect(cueRectL,xOffsetRand,yOffsetRand+(-2*ed(ttn).topIsCued + 1)*(cueDistance+2-((cueSideLength-1)/2+1))));
Screen('FillRect',window,[255 255 255],OffsetRect(cueRectR,xOffsetRand,yOffsetRand+(-2*ed(ttn).topIsCued + 1)*(cueDistance+2-((cueSideLength-1)/2+1))));

Screen('Flip',window);
WaitSecs(targetDuration - flipSlack);

%% response
responseText = sprintf('Please respond.');
DrawFormattedText(window,responseText,'center','center',instructionColor);    
responseScreenBegins = Screen('Flip',window);

%% Keys
RestrictKeysForKbCheck([responseKeys, experimenterKey]);
[pressTime, keyCode, dsecs] = KbWait([],0); %[secs, keyCode, deltaSecs] = KbWait([deviceNumber][, forWhat=0][, untilTime=inf]), for test running, change untilTime to 0.5s

%%quit if F12
if keyCode(experimenterKey)
    quitNow = 1; 
end

%% trial is finished.
trialEndedAt = Screen('Flip',window);
totalTrialDuration = trialEndedAt - trialBeginsAt;

%% postprocessing
if sum(keyCode(responseKeys))==1 % && sum(keyCode2(confidenceKeys))==1
    Resp = SDkeys (find(keyCode(responseKeys)));
    ACC = double( Resp == sameDifferent(2-ed(ttn).cuedIsSame ) );
    reactionTime = pressTime - responseScreenBegins;
        
else
    % wrong key, double key or timeout
    Resp = 'NA';
    ACC = 0;
    reactionTime = 'NA';
    beep;
    DrawFormattedText(window,noResponseText,'center','center',instructionColor);
    Screen('Flip',window);
    RestrictKeysForKbCheck([]);
    KbWait([], 2);
end


%% Clean up
RestrictKeysForKbCheck([]);

dataHeaders = {'Experiment', 'Participant','Age','Gender','Ethnicity',...
    'Trial','Block','Probability','CuedHalf','Congruency', 'Alignment', 'SameDifferent',...
    'FaceGroup','StudyUpper','StudyLower','TargetUpper','TargetLower','FaceIndex',...
    'thisResponse','isCorrect','reactionTime', 'studyDuration', 'targetDuration', 'maskDuration', 'trialEndTime'};
trialData =   { experimentAbbv, participantNum,participantAge, participantGender,participantRace,...
    ttn,ed(ttn).BlockNumber,ed(ttn).Probability, cuedHalf(2-ed(ttn).topIsCued), congruency(2-ed(ttn).isCongruent), alignment(2-ed(ttn).isAligned), sameDifferent(2-ed(ttn).cuedIsSame ) ,...
    faces(thisFaceSet(1),ed(ttn).faceGroup).group, faces(thisFaceSet(1),ed(ttn).faceGroup).filename, faces(thisFaceSet(2),ed(ttn).faceGroup).filename, faces(thisFaceSet(3),ed(ttn).faceGroup).filename, faces(thisFaceSet(4),ed(ttn).faceGroup).filename,ed(ttn).faceIndex,...
    Resp,  ACC, reactionTime, studyDuration, targetDuration, maskDuration, datestr(now,'yyyy-mm-dd-HH:MM:SS')};
if (size(dataHeaders,2) ~= size(trialData,2))
    error('Output malformed: headers and data were not the same size.');
end
allData(1,1:size(dataHeaders,2)) = dataHeaders;
allData(ttn+1,1:size(trialData,2)) = trialData;
