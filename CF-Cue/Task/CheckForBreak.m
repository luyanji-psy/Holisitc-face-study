if (forceRestPeriod && mod(ttn,trialsPerRest)==0 && ttn~=numTrials)

    %% screen display
%     theFigH = figure('Position',[100 100 800 800],'Visible','off');
%     DrawPerformanceChart;
%     figMat = imcapture(theFigH);
%     close(theFigH);    
%     figRect = OffsetRect(CenterRect([0 0 size(figMat,2)*.75 size(figMat,1)*.75],screenRect),0,-50);
%     figTexture = Screen('MakeTexture',window,figMat);
%     Screen('DrawTexture',window, figTexture,[],figRect);
%     
    
    breakText = sprintf(['Please take a rest.\n\n You''re ' num2str(floor(100*ttn/numTrials),'%0.0f') '%% finished.']);
    DrawFormattedText(window,breakText,'center',screenRect(4)-150,instructionColor);
    
    Screen('Flip',window);
    Beeper(1900,.15,.25);
    WaitSecs(restMinimumTime);

    breakText = sprintf('You can now continue with the experiment.\n \nPress any key when ready.');
    DrawFormattedText(window,breakText,'center',screenRect(4)-150,instructionColor);

    Screen('Flip',window);
    Beeper(1280,.15,.25);
    
    RestrictKeysForKbCheck([]);
    [~, keyCodeInt, ~] = KbWait([],3);
    
    clear breakText;
end