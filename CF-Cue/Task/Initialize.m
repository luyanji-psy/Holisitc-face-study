%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; 
commandwindow;
KbName('UnifyKeyNames');
Priority(1);
warning('off','MATLAB:sprintf:InputForPercentSIsNotOfClassChar');
warning('off','MATLAB:fprintf:InputForPercentSIsNotOfClassChar');

Screen('Preference', 'SkipSyncTests', 1);
rand('twister',sum(100*clock));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize and setup the screen
Screen('Preference','TextEncodingLocale','UTF-8');
screens = Screen('Screens');

switch size(screens,2)
    case 3
        whichScreen = 2; 
    case 2
        whichScreen = max(screens);
    case 1
        whichScreen = 0;
    otherwise
        whichScreen = 0;
end
pixelSizes = Screen('PixelSizes', whichScreen);
if max(pixelSizes) < 32
    fprintf('Sorry, I need a screen that supports 32-bit pixelSize.\n');
    return;
end
backgroundColor = 128;
[window,screenRect] = Screen('OpenWindow',whichScreen,backgroundColor,[],max(pixelSizes));
% [offWindow,screenRect2] = Screen('OpenOffscreenwindow', whichScreen, offscreenColor, [],max(pixelSizes));
[screenCenter(1),screenCenter(2)] = RectCenter(screenRect);

screenX = screenRect(3);
screenY = screenRect(4);

HideCursor;

Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
% Screen('BlendFunction', offWindow, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

expectedFrameRate = 60;
frameRate = Screen('NominalFrameRate', window); % the Hz refresh rate
if (frameRate~=expectedFrameRate)
    beep;
    disp(['WARNING... the framerate is not ' num2str(expectedFrameRate) ' Hz; it''s ' num2str(frameRate) ' Hz. This may cause timing issues.']);
end
msPerFrame = Screen('GetFlipInterval',window); % milliseconds per frame
flipSlack = .5*msPerFrame; % needed so that Screen('Flip') can be prepared when the flip occurs.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
