%% MLP Lak�s�rak
clear
Data=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B2:AK1445');%adatsyopkod�s
lnprice=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'A2:A1445');
TestData=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B1446:AK1807');%adatsyopkod�s
%%
neuron=[5 6 7 8 9];     %ez a param�ter adja meg hogy a rejtett r�tegben l�v� neuronok sz�m�t 
repeatN = 5;    %mivel az �leken a s�lyok random m�don ad�dnak az optimali�zl�s eset�n ezzel a param�terrel tudjuk be�ll�tani, hogy h�nyszor futtatjuk le a h�l�zatot (�tlagol�ssal kapjuk majd meg a h�l�zat eredm�nyess�g�t)
TrainSet = 1:1082; %tanul� halmaz
ValidationSet = 1083:1444; %valid�ci�s halmaz
%KnownSet=1:1444;
TestSet = 1445:1806; %test halmat
Variables = 10; %v�ltoz�k
cv=2; %cross validation ciklus
indices=crossvalind('Kfold',1444,6);

for l=1:cv
%Data=circshift(Data,500);
%lnprice=circshift(lnprice,500);
[TrainInput,ps] = mapminmax(Data'); %0-1 tartom�nyra transzform�ljuk a training set input adatokat 
[TrainTarget,ts] = mapminmax(lnprice'); %0-1 tartom�nyra transzform�ljuk a training set target adatokat  
TestInput = mapminmax('apply',TestData',ps); % [0,1] tartom�nyra transzform�ljuk az adatokat

for j=1:numel(neuron)
    for k=1:repeatN
        
inputs = TrainInput; %inputk�nt megadjuk a transzform�lt adatokat 
targets = TrainTarget; %targetk�nt megadjuk a transzform�lt adatokat 

setdemorandstream(491218380+k) %be�ll�tjuk a randomsz�m gener�tort hogy k�s�bb vissza tudjuk n�zni az eredm�nyeket 
hiddenLayerSize = neuron(j);  %megadjuk a rejett r�tegben l�v� neuronok sz�m�t
net = fitnet(hiddenLayerSize); 

net.divideFcn = 'divideind'; %felosztjuk az adatokat tanul� �s valid�k� r�szre
net.divideParam.trainInd = TrainSet;  %tanul� r�sz
net.divideParam.valInd   = ValidationSet; %valid�l� r�sz
net.inputs{1}.processFcns = {}; %be�ll�tjuk hogy m�r ne transzform�lja a h�l�zat az adatainkat hiszen mi m�r megtett�k 
net.outputs{2}.processFcns = {}; %be�ll�tjuk hogy m�r ne transzform�lja a h�l�zat az adatainkat hiszen mi m�r megtett�k 
net.trainFcn = 'trainlm';   %a Levengerg-Marquardt tanul� algoritmust haszn�ljuk
net.performFcn = 'mse';   %mse alapj�n optimaliz�l az algoritmus
net.trainParam.max_fail=6;
net.trainParam.mu=0.01;

[net,tr] = train(net,inputs(:,indices~=l),targets(indices~=l)); %lefuttatjuk a h�l�zatot 
TrainOutputs = net(inputs(:,indices~=l)); %el�rejelz�nk a h�l�zattal a trainhalmayra
ValidationOutputs=net(inputs(:,indices==l));
TestOutputs = net(TestInput); %el�rejelz�nk a h�l�zattal

DataTrainNN((j-1)*repeatN+k,:)=mapminmax('reverse',TrainOutputs,ts); %elmentj�k a m�r visszatranszform�lt outputadatokat
DataValidationNN((j-1)*repeatN+k,:)=mapminmax('reverse',ValidationOutputs,ts); %elmentj�k a m�r visszatranszform�lt outputadatokat
DataTestNN((j-1)*repeatN+k,:)=mapminmax('reverse',TestOutputs,ts); %elmentj�k a m�r visszatranszform�lt outputadatokat

    end
DataMeanTrainNN(j,:)=mean(DataTrainNN(((j-1)*repeatN+1):(j*repeatN),:)); %�tlagoljuk az egyes eredm�nyeket amelyek azonos rejtett r�tegben l�v� neuronsz�mhoz kapcsol�dnak
DataMeanValidationNN(j,:)=mean(DataValidationNN(((j-1)*repeatN+1):(j*repeatN),:)); %�tlagoljuk az egyes eredm�nyeket amelyek azonos rejtett r�tegben l�v� neuronsz�mhoz kapcsol�dnak
DataMeanTestNN(j,:)=mean(DataTestNN(((j-1)*repeatN+1):(j*repeatN),:)); %�tlagoljuk az egyes eredm�nyeket amelyek azonos rejtett r�tegben l�v� neuronsz�mhoz kapcsol�dnak

%rmseTestNN(j,1)=sqrt(sum((lnprice(TestSet)'-DataMeanTestNN(j,:)).^2))/numel(TestSet); %rmse a test halmazon

CrossvalTrainNN(l,j,:)=DataMeanTrainNN(j,:);%lementj�k ay l. corssvalidation ciklus eredmenyet
CrossvalTestNN(l,j,:)=DataMeanTestNN(j,:);%lementj�k ay l. corssvalidation ciklus eredmenyet
CrossvalValidationNN(l,j,:)=DataMeanValidationNN(j,:);%lementj�k ay l. corssvalidation ciklus eredmenyet

end
end
%%
for j=1:numel(neuron)
CrossvalMeanTrainNN(j,:)=mean(CrossvalTrainNN(:,j,:),1);%ki�tlagoljuk a k�l�nb�y? CV ciklusokat
CrossvalMeanValidationNN(j,:)=mean(CrossvalValidationNN(:,j,:),1);%ki�tlagoljuk a k�l�nb�y? CV ciklusokat
CrossvalMeanTestNN(j,:)=mean(CrossvalTestNN(:,j,:),1);%ki�tlagoljuk a k�l�nb�y? CV ciklusokat
rmseTrainNN(j,1)=sqrt(sum((lnprice(indices~=l)'-CrossvalMeanTrainNN(j,:)).^2))/1083; %rmse a train halmazon
rmseValidationNN(j,1)=sqrt(sum((lnprice(indices==l)'-CrossvalMeanValidationNN(j,:)).^2))/361; %rmse a valid�ci�s halmazon
end

%%
z=sort(rmseValidationNN); %sorbarakjuk az rmse �rt�keket
optimalneuron=find(rmseValidationNN==z(1,1));   %kiv�lsztjuk melyikhez tartozik a legkisebb rmse, ez lesz az optim�lis h�l�zat

%% regression
mdl = fitlm(Data(1:ValidationSet(end),1:Variables),lnprice(1:ValidationSet(end),:)); %regresszi�s modell becsl�se

DataTrainR = predict(mdl,Data(TrainSet,1:Variables)); %a becs�lt regresszi�s modellel el�rejelz�nk a train adatokra
DataValidationR = predict(mdl,Data(ValidationSet,1:Variables)); %a becs�lt regresszi�s modellel el�rejelz�nk a valid�ci�s adatokra
DataTestR = predict(mdl,TestData(:,1:Variables)); %a becs�lt regresszi�s modellel el�rejelz�nk a teszt adatokra

rmseTrainR=sqrt(sum((lnprice(TrainSet)-DataTrainR).^2))/numel(TrainSet); %rmse a train halmazon
rmseValidationR=sqrt(sum((lnprice(ValidationSet)-DataValidationR).^2))/numel(ValidationSet); %rmse a valid�ci�s halmazon
%rmseTestR=sqrt(sum((lnprice(TestSet)-DataTestR).^2))/numel(TestSet); %rmse a test halmazon

%% figures
figure(1)
y=exp(lnprice(1:1083)'); %valid�ci�s halmaz lak�sainak eredeti �rai
yNN=exp(CrossvalMeanTrainNN(optimalneuron,:)); %valid�ci�s halmaz lak�sainak becs�lt �rai neura�lis h�l�val
yR=exp(DataTrainR); %valid�ci�s halmaz lak�sainak becs�lt �rai regresszi�val
scatter(yNN, y, 30, 'r', 'fill'); %scatter plotta� �br�zoljuk az adatokat 
axis([0 100 0 100]); %be�ll�tjuk a tengelyeket 
set(gca,'YTick',0:20:100) %be�ll�tjuk az y tengely feloszt�s�t
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az y tengely megjelen�t�s�t
set(gca,'XTick',0:20:100) %be�ll�tjuk az x tengely feloszt�s�t
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az x tengely megjelen�t�s�t
xlabel('Lak�s�rak becsl�se neur�lis h�l�val �s regresszi�val'); %az x tengelynek feliratot adnunk
ylabel('Lak�s�rak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal �br�zoljuk az adatokat
hleg1 = legend([strcat('Neur�lis h�l� MSE: ', {''}, num2str(rmseTrainNN(optimalneuron)),''), strcat('Regresszi� MSE: ', {''}, num2str(rmseTrainR),'')]);
set(hleg1,'Location','NorthWest') %be�ll�tjuk, hogy l�tsz�djon a felirat hogy melyik pontok melyik becsl�shez tartoznak
title('TrainSet')  

figure(2)
y=exp(lnprice(1083:1444)'); %valid�ci�s halmaz lak�sainak eredeti �rai
yNN=exp(DataMeanValidationNN(optimalneuron,:)); %valid�ci�s halmaz lak�sainak becs�lt �rai neura�lis h�l�val
yR=exp(DataValidationR); %valid�ci�s halmaz lak�sainak becs�lt �rai regresszi�val
scatter(yNN, y, 30, 'r', 'fill'); %scatter plotta� �br�zoljuk az adatokat 
axis([0 100 0 100]); %be�ll�tjuk a tengelyeket 
set(gca,'YTick',0:20:100) %be�ll�tjuk az y tengely feloszt�s�t
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az y tengely megjelen�t�s�t
set(gca,'XTick',0:20:100) %be�ll�tjuk az x tengely feloszt�s�t
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az x tengely megjelen�t�s�t
xlabel('Lak�s�rak becsl�se neur�lis h�l�val �s regresszi�val'); %az x tengelynek feliratot adnunk
ylabel('Lak�s�rak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal �br�zoljuk az adatokat
hleg1 = legend([strcat('Neur�lis h�l� MSE: ', {''}, num2str(rmseValidationNN(optimalneuron)),''), strcat('Regresszi� MSE: ', {''}, num2str(rmseValidationR),'')]);
set(hleg1,'Location','NorthWest') %be�ll�tjuk, hogy l�tsz�djon a felirat hogy melyik pontok melyik becsl�shez tartoznak
title('ValidationSet')  

%%
figure(3)
y=exp(lnprice(1084:1444)'); %valid�ci�s halmaz lak�sainak eredeti �rai
yNN=exp(CrossvalMeanTestNN(optimalneuron,:)); %valid�ci�s halmaz lak�sainak becs�lt �rai neura�lis h�l�val
yR=exp(DataTestR); %valid�ci�s halmaz lak�sainak becs�lt �rai regresszi�val
scatter(yNN, y, 30, 'r', 'fill'); %scatter plotta� �br�zoljuk az adatokat 
axis([0 100 0 100]); %be�ll�tjuk a tengelyeket 
set(gca,'YTick',0:20:100) %be�ll�tjuk az y tengely feloszt�s�t
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az y tengely megjelen�t�s�t
set(gca,'XTick',0:20:100) %be�ll�tjuk az x tengely feloszt�s�t
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %be�ll�tjuk az x tengely megjelen�t�s�t
xlabel('Lak�s�rak becsl�se neur�lis h�l�val �s regresszi�val'); %az x tengelynek feliratot adnunk
ylabel('Lak�s�rak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal �br�zoljuk az adatokat
hleg1 = legend([strcat('Neur�lis h�l� MSE: ', {''}, num2str(rmseTestNN(optimalneuron)),''), strcat('Regresszi� MSE: ', {''}, num2str(rmseTestR),'')]);
set(hleg1,'Location','NorthWest') %be�ll�tjuk, hogy l�tsz�djon a felirat hogy melyik pontok melyik becsl�shez tartoznak
title('TestSet')  
%%
y = exp(CrossvalMeanTestNN(optimalneuron,:)');
fej={'sorszam' 'price'};
%sorsz = DataTest(:,1)
sorsz=1:362;
sorsz=sorsz';
%outshit = [[0;sorsz],[0;transpose(y)]]
out=horzcat(y,sorsz);
out2=[fej;num2cell(out)];
xlswrite('looooooool.csv',out2)