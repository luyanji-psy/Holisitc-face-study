%% load face iamges
% facesDir = [pwd '\Stimuli\'];
% faceImageFiles = dir([facesDir '*.png']);
% 
% randomize the order of images
% {theFiles(randperm(numFacesPerGender)).name}'
% numFaceImages = size(faceImageFiles,1);
% if numFaceImages ~= 2*numFacesPerGender
%     warning('Warning: the number of face image files differed from the design expected...');
% end

%% Load stimulus
load(['Stimuli' filesep 'Chinese']);
% load(['Stimuli' filesep stimSet{stimRace} '_Mask']);

%% Make Texture for stimuli
for iFace = 1:length(faces)
    faces(iFace).texture = Screen('MakeTexture', window, cat(3,faces(iFace).matrix,faces(iFace).alpha)); %#ok<SAGROW>
end

% % read images
% faces = struct([]);
% for iImage = 1:numFaceImages
%     clear tempFace
%     tempFace.filename = faceImageFiles(iImage).name;    
%     [tempFace.matrix, ~, tempFace.alpha] = imread([facesDir tempFace.filename]); % load the transparent layer correctly
%     tempFace.texture = Screen('MakeTexture', window, cat(3,tempFace.matrix,tempFace.alpha));
%     tempFace.gender = tempFace.filename(1);
%     tempFace.group = tempFace.filename(1:2);
%     faces = [faces;tempFace];
% end
% clear tempFace

% arrand into 10 (index) x 2 (gender) structure, then randomize assignments
faces = [faces(strcmp({faces.group},'F1')), ...
         faces(strcmp({faces.group},'F2')), ...
         faces(strcmp({faces.group},'F3')), ...
         faces(strcmp({faces.group},'F4')), ...
         faces(strcmp({faces.group},'F5')), ...
         faces(strcmp({faces.group},'M1')), ...
         faces(strcmp({faces.group},'M2')), ...
         faces(strcmp({faces.group},'M3')), ...
         faces(strcmp({faces.group},'M4')), ...
         faces(strcmp({faces.group},'M5'))];
     
for iCol = 1:size(faces,2)
    clear newOrder
    newOrder = randperm( size(faces(:,iCol),1) );
    faces(:,iCol) = faces( newOrder , iCol);
end


% faces = [faces(randperm(length(faces)),1) ,...
%          faces(randperm(length(faces)),2) ,...
%          faces(randperm(length(faces)),3) ,...
%          faces(randperm(length(faces)),4) ,...
%          faces(randperm(length(faces)),5) ,...
%          faces(randperm(length(faces)),6) ,...
%          faces(randperm(length(faces)),7) ,...
%          faces(randperm(length(faces)),8) ,...
%          faces(randperm(length(faces)),9) ,...
%          faces(randperm(length(faces)),10)];
faceRect = [0 0 size(faces(1).matrix,2) size(faces(1).matrix,1)];
faceTopRect = [screenX/2-100 screenY/2-129 screenX/2+100 screenY/2-1];
faceBottomRect = [screenX/2-100 screenY/2+2 screenX/2+100 screenY/2+130];
lineRect = [screenX/2-225 screenY/2-1 screenX/2+225 screenY/2+2];

% cue position
cuePixel = 5;
cueLength = 150;
cueSideLength = 21;
cueDistance = 140;
cueRect = [screenX/2-cueLength screenY/2-(cuePixel-1)/2 screenX/2+cueLength screenY/2+((cuePixel-1)/2+1)];
cueRectL = [screenX/2-cueLength-cuePixel screenY/2-((cueSideLength-1)/2+1) screenX/2-cueLength screenY/2+(cueSideLength-1)/2];
cueRectR = [screenX/2+cueLength screenY/2-(cueSideLength-1)/2 screenX/2+cueLength+3 screenY/2+((cueSideLength-1)/2+1)];



