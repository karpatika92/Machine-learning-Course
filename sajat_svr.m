%SUPPORT VECTOR REGRESSION
%Data intake
clear all
Data=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B2:AK1445');%adatsyopkodás
lnprice=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'A2:A1445');
TestData=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B1446:AK1807');%adats
%Setting Base parameters
cv=4;
%generating indices
indices=crossvalind('Kfold',1444,cv);
%Standardizing Data
[TrainInput,ps] = mapminmax(Data',0,1); %0-1 tartományra transzformáljuk a training set input adatokat 
[TrainTarget,ts] = mapminmax(lnprice',0,1); %0-1 tartományra transzformáljuk a training set target adatokat 
TestInput2 = mapminmax('apply',TestData',ps); %0-1 tartományra transzformáljuk a training set input adatokat 
 
TestInput = mapminmax('apply',TestData',ps); % [0,1] tartományra transzformáljuk az adatokat
valid_pred_SVM=zeros(1444,1);
best_rmse_valid_svm=999;
%metaparaméterek értékei
epsilons=[0.0001 0.001 0.01 0.05];
%cs=[2^(-5) 2^(-3) 2^(-1) 2^(1) 2^(3) 2^(5) 2^(7) 2^(9) 2^(11) 2^(13)];
%gammas=[2^(-15) 2^(-13) 2^(-11) 2^(-9) 2^(-7) 2^(-5) 2^(-3) 2^(-1) 2^(1) 2^(3)]; 
cs=[2^(5) 2^(7) 2^(9) 2^(11) 2^(13)];
gammas=[2^(-9) 2^(-7) 2^(-5) 2^(-3) 2^(-1)]; 
ds=[2 3 4];
%%
%különböz? paraméterek ciklusai
%gamma
for m=1:5
    gamma=gammas(m);
    for ck=1:5
        k=cs(ck);
for e=1:4
        eps=epsilons(e);
    SVM_pred=zeros(1444,cv); %clear pred matix

%Cross-validation-cycle begins
for c=1:cv

% Train SVR Model
clear model
TSet=find(indices~=c); %Trainset
%TSet=TSet2(1:T);
VSet=find(indices==c); %Valid Set
%VSet=VSet2(1:362);
cmd=['-s ',num2str(3),' -t ',num2str(1),' -e ',num2str(eps),' -h ',num2str(0),' -c ',num2str(k),' -g ',num2str(gamma), ' -d ', num2str(3)]; %meta parameters
model= svmtrain( TrainTarget(indices~=c)',TrainInput(:,indices~=c)', cmd); %train SVM
y_svm_transform_valid=svmpredict(TrainTarget(indices==c)',TrainInput(:,indices==c)',model); %predict in validation set
SVM_pred(VSet,c)=mapminmax('reverse',y_svm_transform_valid,ts); %transform data back 

y_svm_transform_train(c,1:numel(TSet))=svmpredict(TrainTarget(indices~=c)',TrainInput(:,indices~=c)',model); %predict train set
SVM_pred(TSet,c)=mapminmax('reverse',y_svm_transform_train(c,1:numel(TSet)),ts); %transform data back
valid_pred_SVM(VSet)=mapminmax('reverse',y_svm_transform_valid,ts); %validation vector

end

y_svm_train_mean=mean(SVM_pred,2);
y_svm_valid_mean=valid_pred_SVM;
%RMSE kiszámítása
rmseTrainSVM(m,e,ck)=sqrt(sum((lnprice(indices~=3)-y_svm_train_mean(indices~=3)).^2/sum(indices~=3))); %rmse a train halmazon
rmseValidationSVM(m,e,ck)=sqrt(sum((lnprice(indices~=3)-y_svm_valid_mean(indices~=3)).^2)/sum(indices~=3)); %rmse a validation halmazon
%írja felül ha jobb
    if(best_rmse_valid_svm>rmseValidationSVM(m,e,ck)),
        %save all parameters and results as best
        best_rmse_valid_svm=rmseValidationSVM(m,e,ck);
        best_rmse_train_svm=rmseTrainSVM(m,e,ck);
        y_svm_valid_best=y_svm_valid_mean;
        best_cmd=cmd;
        best_eps=e;
        best_model=m;
        best_c=ck;
        best_Test=svmpredict((1:362)',TestInput2',model);
    end
end
end

end
best_rmse_valid_svm
%%
%linear regression
mdl = fitlm(Data(TSet,:),lnprice(TSet)); %regressziós modell becslése
DataTrainR = predict(mdl,Data(TSet,:)); %a becsült regressziós modellel elõrejelzünk a train adatokra
DataValidationR = predict(mdl,Data(VSet,:)); %a becsült regressziós modellel elõrejelzünk a validációs adatokra
DataTestR = predict(mdl,TestData); %a becsült regressziós modellel elõrejelzünk a teszt adatokra
%%
rmseTrainR=sqrt(sum(((lnprice(indices~=c))-DataTrainR).^2)/sum(indices~=c)); %rmse a train halmazon
rmseValidationR=sqrt(sum((lnprice(indices==c)-DataValidationR).^2)/sum(indices==c)); %rmse a validation halmazon

%%
%Best Model Estimation

output=mapminmax('reverse',best_Test,ts);
output2=exp(output);


%%
figure(1)
y=exp(lnprice(indices==c)'); %validációs halmaz lakásainak eredeti árai
ySVM=exp(y_svm_valid_best(indices==c))'; %validációs halmaz lakásainak becsült árai neuraális hálóval
yR=exp(DataValidationR)'; %validációs halmaz lakásainak becsült árai regresszióval
scatter(ySVM, y, 30, 'r', 'fill'); %scatter plottaé ábrázoljuk az adatokat 
axis([0 100 0 100]); %beállítjuk a tengelyeket 
set(gca,'YTick',0:20:100) %beállítjuk az y tengely felosztását
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az y tengely megjelenítését
set(gca,'XTick',0:20:100) %beállítjuk az x tengely felosztását
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az x tengely megjelenítését
xlabel('Lakásárak becslése SVMmel és regresszióval'); %az x tengelynek feliratot adnunk
ylabel('Lakásárak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal ábrázoljuk az adatokat
hleg1 = legend([strcat('SVM MSE: ', {''}, num2str(best_rmse_valid_svm),''), strcat('Regresszió MSE: ', {''}, num2str(rmseValidationR),'')]);
set(hleg1,'Location','NorthWest') %beállítjuk, hogy látszódjon a felirat hogy melyik pontok melyik becsléshez tartoznak
title('ValidationSet')  
%%
figure(2)
y=exp(lnprice(indices~=c)'); %validációs halmaz lakásainak eredeti árai
ySVM=exp(y_svm_valid_best(indices~=c))'; %validációs halmaz lakásainak becsült árai neuraális hálóval
yR=exp(DataValidationR)'; %validációs halmaz lakásainak becsült árai regresszióval
scatter(ySVM, y, 30, 'r', 'fill'); %scatter plottaé ábrázoljuk az adatokat 
axis([0 100 0 100]); %beállítjuk a tengelyeket 
set(gca,'YTick',0:20:100) %beállítjuk az y tengely felosztását
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az y tengely megjelenítését
set(gca,'XTick',0:20:100) %beállítjuk az x tengely felosztását
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az x tengely megjelenítését
xlabel('Lakásárak becslése SVMmel és regresszióval'); %az x tengelynek feliratot adnunk
ylabel('Lakásárak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal ábrázoljuk az adatokat
hleg1 = legend([strcat('SVM MSE: ', {''}, num2str(best_rmse_train_svm),''), strcat('Regresszió MSE: ', {''}, num2str(rmseTrainR),'')]);
set(hleg1,'Location','NorthWest') %beállítjuk, hogy látszódjon a felirat hogy melyik pontok melyik becsléshez tartoznak
title('TrainSet')  