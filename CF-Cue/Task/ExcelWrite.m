    %%%%%%%%%%%%%%%%%%%%%%%
    %% Excel Write    
    try
        disp(['Creating excel file ', theExcelFile, ' from template...']);
        [status, message] = copyfile(theTemplateFile,theExcelFile);
        if (status~=1), beep, disp(message), end
        [status, message] = fileattrib(theExcelFile,'+w');
        if (status~=1), beep, disp(message), end
        
        % header = {ID; positionID; date; datestr(startTime,13); datestr(endTime,13); etime(endTime,startTime)/60; i; i/(etime(endTime,startTime)/60)};

        disp('Writing data to excel file...');
        [status, message] = xlswrite(theExcelFile,allData,'Data','A1');
        if (status~=1), beep, disp(message.message), end
        
        % [status, message] = xlswrite(theExcelFile,header,'Info','B1');
        disp('Finished!');
    catch err
        rethrow(err);
    end