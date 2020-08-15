%% Parameters of fonts used in this exp
textSize = 18;
textFont = 'Helvetica';
textColor = 255; 

Screen('TextSize', window, textSize);
Screen('TextFont', window, textFont);
Screen('TextColor', window, textColor);

%% -------------------------------
excelDir = 'Excel Data\';
filesDir = 'Shared Files\';
excelExtension = '.xlsx';
matExtension = '.mat';
saveDir = 'Matlab Data\';
%backupDir = ['']; % this should be the full path to the dropbox on the main experiment computer

thisDateVector = now;
theDateString = datestr(thisDateVector,'yyyy-mm-dd-HHMM');
theDate8 = str2double(datestr(thisDateVector,'yyyymmdd'));
theDataFilename = [participantNum '_' experimentAbbv '_' num2str(stimRace) '_' theDateString];
theExcelFile = [excelDir theDataFilename excelExtension];
theMatlabFile = [saveDir theDataFilename matExtension];

quitNow = 0;

