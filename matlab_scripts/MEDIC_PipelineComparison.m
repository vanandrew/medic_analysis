clc;
clear;
close all;
addpath(('/data/nil-bluearc/GMT/David/david-functions'))
addpath('/data/nil-bluearc/GMT/David/david-functions/fieldtrip-master/external/freesurfer')

DataRoot = '/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2';
OutputRoot = '/net/10.20.145.34/DOSENBACH02/GMT2/Andrew/MEDIC';

% Search for subjects
SearchCommandString = sprintf('ls -1d %s',fullfile(DataRoot,'sub-*'));
[~,SearchResultsString] = system(SearchCommandString);
SearchResultsCell = strsplit(SearchResultsString,'\n');
SearchResultsCell = SearchResultsCell(cellfun(@(x) ~isempty(x),SearchResultsCell))';
[~,SubjectLabels] = cellfun(@(x) fileparts(x),SearchResultsCell,'UniformOutput',false);

% Lambda function for apply with the spotlight analysis
SearchFuncR2 = @(x,y,z) SimpleCorr(x(logical(z)),y(logical(z))).^2;
SearchFuncR = @(x,y,z) SimpleCorr(x(logical(z)),y(logical(z)));

StructuringElementSmall = strel('sphere',2);
StructuringElementMedium = strel('sphere',3);
StructuringElementLarge = strel('sphere',5);

PipelineList = {'MEDIC','TOPUP'};

%%
for SubjectIDX = [1:numel(SubjectLabels)]
    
    ValidSubject = true;
    ThisSubjectLabel = SubjectLabels{SubjectIDX};
    ThisSubjectOutputDir = fullfile(OutputRoot,ThisSubjectLabel);
    ThisSubjectAtlasDir = fullfile(SearchResultsCell{SubjectIDX},'T1','atlas');
    
    % Get list of BOLD directories
    SearchString = sprintf('ls -1d %s',fullfile(SearchResultsCell{SubjectIDX},'ses-*'));
    [~,SessionDirList] = system(SearchString);
    SessionDirList = strsplit(SessionDirList, '\n')'; 
    [~, SessionLabels] = cellfun(@(x) fileparts(x),SessionDirList(1:end-1),'UniformOutput',false);
    SessionLabels = unique(strrep(SessionLabels,'wTOPUP',''));
    
    if ~exist(ThisSubjectOutputDir,'dir');mkdir(ThisSubjectOutputDir);end

    ThisSubjectParcFileName = fullfile(ThisSubjectAtlasDir,sprintf('%s_wmparc_on_MNI152_T1_2mm.nii.gz',ThisSubjectLabel));
    ThisSubjectT1FileNamePattern = fullfile(ThisSubjectAtlasDir,sprintf('%s_T1w*debias*_on_MNI152_T1_2mm.nii.gz',ThisSubjectLabel));
    ThisSubjectT2FileNamePattern = fullfile(ThisSubjectAtlasDir,sprintf('%s_T2w*debias*_on_MNI152_T1_2mm.nii.gz',ThisSubjectLabel));
 
    [~,ThisSubjectT1FileName] = system(sprintf('ls -1 %s',ThisSubjectT1FileNamePattern));
    ThisSubjectT1FileName = strsplit(ThisSubjectT1FileName,'\n');
    ThisSubjectT1FileName = ThisSubjectT1FileName{1};
    
     [~,ThisSubjectT2FileName] = system(sprintf('ls -1 %s',ThisSubjectT2FileNamePattern));
    ThisSubjectT2FileName = strsplit(ThisSubjectT2FileName,'\n');
    ThisSubjectT2FileName = ThisSubjectT2FileName{1};   
    
    ThisSubjectParcData = load_nifti(ThisSubjectParcFileName);
    ThisSubjectT1Data = load_nifti(ThisSubjectT1FileName);
    ThisSubjectT2Data = load_nifti(ThisSubjectT2FileName);
    
    VolDims = size(ThisSubjectT2Data.vol);
    VolElementCount = prod(VolDims);
    
    ThisSubjectParcMask = imdilate(ThisSubjectParcData.vol > 0,StructuringElementSmall);
    % ThisSubjectParcMask = ThisSubjectParcData.vol > 0;
    
    ThisSubjectT1DataArray  = ThisSubjectT1Data.vol(:);
    ThisSubjectT2DataArray  = ThisSubjectT2Data.vol(:);
    
    [ThisSubjectT1GradX,ThisSubjectT1GradY,ThisSubjectT1GradZ] = gradient(ThisSubjectT1Data.vol);
    ThisSubjectT1GradNorm = sqrt(ThisSubjectT1GradX.^2 + ThisSubjectT1GradY.^2 + ThisSubjectT1GradZ.^2);
    
    [ThisSubjectT2GradX,ThisSubjectT2GradY,ThisSubjectT2GradZ] = gradient(ThisSubjectT2Data.vol);
    ThisSubjectT2GradNorm = sqrt(ThisSubjectT2GradX.^2 + ThisSubjectT2GradY.^2 + ThisSubjectT2GradZ.^2);
    
    ThisSubjectT1GradNormDataArray  = ThisSubjectT1GradNorm(ThisSubjectParcMask);
    ThisSubjectT2GradNormDataArray  = ThisSubjectT2GradNorm(ThisSubjectParcMask);
    
    ThisGrayMatterIDX = ((ThisSubjectParcData.vol >= 1000) & (ThisSubjectParcData.vol <= 3000)) | ...
        ismember(ThisSubjectParcData.vol, [47,8,51,52,12,13,49,10,16]);
      
    SessionStructElement = struct('Pipeline',struct('MEDIC',[],'TOPUP',[]));
    SessionCount = numel(SessionLabels);
    ThisSessionStruct =  repmat(SessionStructElement,[SessionCount,1]);
    
    for PipelineIDX = 1:numel(PipelineList)
        
        ThisPipelineString = upper(PipelineList{PipelineIDX});
        
        switch ThisPipelineString
            case 'MEDIC'
                ThisSessionBaseNames = SessionLabels;
            case 'TOPUP'
                ThisSessionBaseNames = strcat(SessionLabels,'wTOPUP');
        end
        
        ThisSubjectDataDir = fullfile(ThisSubjectOutputDir,ThisPipelineString);
        if ~exist(ThisSubjectDataDir,'dir'); mkdir(ThisSubjectDataDir);end        
                  
        for SessionIDX = 1:SessionCount
            
            % Find all the runs for this session
            ThisRunSearchString = sprintf('ls -d %s', fullfile(DataRoot, ThisSubjectLabel,ThisSessionBaseNames{SessionIDX},'bold*'));
            [cmd,ThisRunDirList] = system(ThisRunSearchString);
            ThisRunDirList = strsplit(ThisRunDirList,'\n')';
            ThisRunDirList = ThisRunDirList(1:end-1);
            
            [~, ThisRunLabels] = cellfun(@(x) fileparts(x),ThisRunDirList,'UniformOutput',false);

            % Get the bold numbers from each run label
            BOLDNumberIndices = regexpi(ThisRunLabels,'[0-9]');
            BOLDNumbers = cellfun(@(x,y) str2double(x(y)),ThisRunLabels,BOLDNumberIndices);
            RunCount = numel(BOLDNumbers);
            
            RunDataStruct = struct('T1NMI',zeros(1,RunCount),...
                'T2NMI',zeros(1,RunCount),...
                'T1Corr',zeros(1,RunCount),...
                'T2Corr',zeros(1,RunCount),...
                'T1GradCorr',zeros(1,RunCount),...
                'T2GradCorr',zeros(1,RunCount),...
                'ROC',zeros(3,RunCount),...
                'TSNR',zeros([VolDims,RunCount]),...
                'SD',zeros([VolDims,RunCount]),...
                'T1SpotlightR2',zeros([VolDims,RunCount]),...
                'T2SpotlightR2',zeros([VolDims,RunCount]),...
                'T1SpotlightR',zeros([VolDims,RunCount]),...
                'T2SpotlightR',zeros([VolDims,RunCount])...
                );
    
            % Create the file names for the resting state data in 4DFP
            % format
            
            for RunIDX = 1:numel(ThisRunLabels)
                ThisSourceBasename = sprintf('%s_b%i_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt', ThisSubjectLabel, BOLDNumbers(RunIDX));
                ThisDestBasename = sprintf('%s_%s_b%i_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt', ThisSubjectLabel,SessionLabels{SessionIDX}, BOLDNumbers(RunIDX));
                ThisSourcePrefix = fullfile(ThisRunDirList{RunIDX}, ThisSourceBasename);
                ThisDestPrefix = fullfile(ThisSubjectDataDir, ThisDestBasename);
                
                ThisDataFileName = [ThisDestPrefix,'.nii.gz'];
                
                 % Convert the 4DFP file if it hasn't already been done
                 if ~exist(ThisDataFileName,'file')
                     CommandString = 'niftigz_4dfp -n %s %s';
                     [cmd,out] = system(sprintf(CommandString,ThisSourcePrefix,ThisDestPrefix));
                 end
                
                % Load up the data
                ThisSessionData = load_nifti(ThisDataFileName);
                ThisSessionMeanVolume = mean(ThisSessionData.vol(:,:,:,4:end),4);
                
                ThisSessionValidVoxelIDX = ~all(ThisSessionData.vol == ThisSessionData.vol(:,:,:,1), 4);
                ThisSessionSimilarityMask = ThisSessionValidVoxelIDX & ThisSubjectParcMask;
                
                % Compute the gradient magnitude mask for this session
                [ThisSessionGradX,ThisSessionGradY,ThisSessionGradZ] = gradient(ThisSessionMeanVolume);
                ThisSessionGradNorm = sqrt(ThisSessionGradX.^2 + ThisSessionGradY.^2 + ThisSessionGradZ.^2);
                
                
                % Compute TSNR and plain temporal standard deviation for this data
                %RunDataStruct.TSNR(:,:,:,RunIDX) = std(ThisSessionData.vol(:,:,:,4:end),[],4) ./ ThisSessionMeanVolume;
                %RunDataStruct.SD(:,:,:,RunIDX) = std(ThisSessionData.vol(:,:,:,4:end),[],4);
                
                [~, ~, ~,ThisSubjectT1DataBins] = binbypercentile(ThisSubjectT1DataArray(ThisSessionSimilarityMask),ThisSubjectT1DataArray(ThisSessionSimilarityMask),256,false);
                [~, ~, ~,ThisSubjectT2DataBins] = binbypercentile(ThisSubjectT2DataArray(ThisSessionSimilarityMask),ThisSubjectT2DataArray(ThisSessionSimilarityMask),256,false);
                [~, ~, ~,ThisSessionDataBins] = binbypercentile(ThisSessionMeanVolume(ThisSessionSimilarityMask), ThisSessionMeanVolume(ThisSessionSimilarityMask),256,false);
                
                % Compute mutual information between the mean functional volume
                % and the T1w and T2w images. For this analysis, we conver the
                % functional volumes into 256 color grayscale images before
                % computing mutual information.
                RunDataStruct.T1NMI(RunIDX) = nmi(ThisSubjectT1DataBins,ThisSessionDataBins);
                RunDataStruct.T2NMI(RunIDX) = nmi(ThisSubjectT2DataBins,ThisSessionDataBins);
                
                RunDataStruct.T1Corr(RunIDX) = SimpleCorr(ThisSubjectT1Data.vol(ThisSessionSimilarityMask),ThisSessionMeanVolume(ThisSessionSimilarityMask));
                RunDataStruct.T2Corr(RunIDX) = SimpleCorr(ThisSubjectT2Data.vol(ThisSessionSimilarityMask),ThisSessionMeanVolume(ThisSessionSimilarityMask));
                
                RunDataStruct.T1GradCorr(RunIDX) = SimpleCorr(ThisSubjectT1GradNorm(ThisSessionSimilarityMask),ThisSessionGradNorm(ThisSessionSimilarityMask));
                RunDataStruct.T2GradCorr(RunIDX) = SimpleCorr(ThisSubjectT2GradNorm(ThisSessionSimilarityMask),ThisSessionGradNorm(ThisSessionSimilarityMask));
                
                ThisValidVoxelIDX = imdilate(ThisSubjectParcData.vol > 0,StructuringElementSmall) & ThisSessionValidVoxelIDX;
                
                % Spotlight stuff
                ThisT1SearchLight = ApplySearchLight(StructuringElementMedium.Neighborhood,SearchFuncR2,ThisSubjectT1Data.vol,ThisSessionMeanVolume,ThisValidVoxelIDX);
                ThisT2SearchLight = ApplySearchLight(StructuringElementMedium.Neighborhood,SearchFuncR2,ThisSubjectT2Data.vol,ThisSessionMeanVolume,ThisValidVoxelIDX);
                
                % Save the spotlight data out
                OutputStruct = ThisSubjectParcData;
                OutputStruct.vol = ThisT1SearchLight;
                save_nifti(OutputStruct,fullfile(OutputRoot,'Spotlight',sprintf('%s_%s_%0.2i_%s_t1_r_spotlight.nii.gz',ThisSubjectLabel,SessionLabels{SessionIDX},RunIDX,ThisPipelineString)));
                
                OutputStruct = ThisSubjectParcData;
                OutputStruct.vol = ThisT2SearchLight;
                save_nifti(OutputStruct,fullfile(OutputRoot,'Spotlight',sprintf('%s_%s_%0.2i_%s_t2_r_spotlight.nii.gz',ThisSubjectLabel,SessionLabels{SessionIDX},RunIDX,ThisPipelineString)));
                
                
                RunDataStruct.T1SpotlightR2(:,:,:,RunIDX) = ThisT1SearchLight;
                RunDataStruct.T2SpotlightR2(:,:,:,RunIDX) = ThisT2SearchLight;
                   
                % Compute the ROC values for adjacent gray and non-gray matter
                % voxels
                RunDataStruct.ROC(:,RunIDX) = GetGrayWhiteROC(ThisSessionMeanVolume,ThisSubjectParcData.vol);
                
                
            end
            
            ThisSessionStruct(SessionIDX).Pipeline.(ThisPipelineString) = RunDataStruct;
        
        end

    end


    AllDataStruct(SubjectIDX).SubjectID = ThisSubjectLabel;
    AllDataStruct(SubjectIDX).GrayMatterMask = ThisGrayMatterIDX;
    AllDataStruct(SubjectIDX).ParcData = ThisSubjectParcData.vol;
    AllDataStruct(SubjectIDX).Sessions = ThisSessionStruct;
    
end


 
%% Organize the data into a data table
PipelineList = {'MEDIC','TOPUP'};
SimpleVarList = {'T1Corr', 'T2Corr', 'T1NMI', 'T2NMI', 'T1GradCorr', 'T2GradCorr'};
FullDataTable = [];

for SubjectNumber = 1:numel(AllDataStruct)
    
    ThisSubjectParcData = AllDataStruct(SubjectNumber).ParcData;
    ThisSubjectMask = ThisSubjectParcData ~= 0 ;
    ThisSubjectGrayMatterVoxelIndex = ((ThisSubjectParcData >= 1000) & (ThisSubjectParcData <= 3000)) | ...
    ismember(ThisSubjectParcData, [47,8,51,52,12,13,49,10,16]);
    %ThisSubjectSpotlightMask = ThisSubjectGrayMatterVoxelIndex;
    ThisSubjectSpotlightMask = ThisSubjectMask;

    
    for SessionNumber = 1:numel(AllDataStruct(SubjectNumber).Sessions)
        
        % make sure both pipeline have the same amount of data
        if nnz(AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.MEDIC.T1NMI) == nnz(AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.TOPUP.T1NMI)
            
            ThisSessionData = AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.MEDIC;
            ThisRunCount = numel(ThisSessionData.T1Corr);
            ThisRunTable = table();
            
            for RunNumber =  1:ThisRunCount
            
                ThisRunTable{RunNumber,'Subject'} = ThisSubjectLabel;
                ThisRunTable{RunNumber,'Session'} = SessionLabels{SessionNumber};
                ThisRunTable{RunNumber,'Run'} = RunNumber;
                
                for PipelineIDX = 1:numel(PipelineList)
                    
                    ThisPipelineString = PipelineList{PipelineIDX};
                    
                    for VarIDX = 1:numel(SimpleVarList)

                        
                        ThisVar = SimpleVarList{VarIDX};
                        ThisRunTable{RunNumber,strcat(ThisPipelineString,ThisVar)} = AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).(ThisVar)(RunNumber);
                        
                        % Do the spotlight analysis
                        DilMask = imdilate(ThisSubjectMask,strel('sphere',1));
                        ThisT1Spotlight = squeeze( AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).T1SpotlightR2(:,:,:,RunNumber));
                        ThisRunTable{RunNumber,strcat(ThisPipelineString,'T1SpotlightR2')} = mean(ThisT1Spotlight(ThisSubjectSpotlightMask),'omitnan');
                        
                        ThisT2Spotlight = squeeze( AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).T2SpotlightR2(:,:,:,RunNumber));
                        ThisRunTable{RunNumber,strcat(ThisPipelineString,'T2SpotlightR2')} = mean(ThisT2Spotlight(ThisSubjectSpotlightMask),'omitnan');
                        
                        %ThisTSNR = 1./ squeeze( AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).TSNR(:,:,:,RunNumber));
                        %ThisSD = squeeze( AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).SD(:,:,:,RunNumber));

                        %ThisValidIDX = ThisSubjectSpotlightMask & isfinite(ThisTSNR) & (ThisTSNR ~= 0);
                        
                        %ThisRunTable{RunNumber,strcat(ThisPipelineString,'TSNR')} = mean(ThisTSNR(ThisValidIDX),'omitnan');
                        %ThisRunTable{RunNumber,strcat(ThisPipelineString,'SD')} = mean(ThisSD(ThisValidIDX),'omitnan');
                        
                        
                        
                    end
                    
                    ThisRunTable{RunNumber,strcat(ThisPipelineString,'ROCGW')} = AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).ROC(1,RunNumber);
                    ThisRunTable{RunNumber,strcat(ThisPipelineString,'ROCIE')} = AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).ROC(2,RunNumber);
                    ThisRunTable{RunNumber,strcat(ThisPipelineString,'ROCVW')} = AllDataStruct(SubjectNumber).Sessions(SessionNumber).Pipeline.(ThisPipelineString).ROC(3,RunNumber);
                    
                    
                end
                
                
                
            end
            disp([SubjectNumber,SessionNumber,RunNumber])
            FullDataTable = cat(1,FullDataTable,ThisRunTable);
        end
    end
    
end

%%
VarList = {'TSNR','T1SpotlightR2','T2SpotlightR2'};

for VarIDX = 1:numel(VarList)
    
    ThisVar = VarList{VarIDX};
    ThisMEDICDataCell = {};
    ThisTOPUPDataCell = {};
    for i = 1:numel(AllDataStruct)
        for j = 1:numel(AllDataStruct(i).Sessions)
            ThisMEDICDataCell{numel(ThisMEDICDataCell)+1} = AllDataStruct(i).Sessions(j).Pipeline.MEDIC.(ThisVar);
            ThisTOPUPDataCell{numel(ThisTOPUPDataCell)+1} = AllDataStruct(i).Sessions(j).Pipeline.TOPUP.(ThisVar);
        end
    end
    
    
end
