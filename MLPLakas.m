%% MLP Lakásárak
clear
Data=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B2:AK1445');%adatsyopkodás
lnprice=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'A2:A1445');
TestData=xlsread('C:\Users\karpatika\Documents\RAJK\MACHINELEARNING\3_IngatlanRajk.xls',1,'B1446:AK1807');%adatsyopkodás
%%
neuron=[5 6 7 8 9];     %ez a paraméter adja meg hogy a rejtett rétegben lévõ neuronok számát 
repeatN = 5;    %mivel az éleken a súlyok random módon adódnak az optimaliázlás esetén ezzel a paraméterrel tudjuk beállítani, hogy hányszor futtatjuk le a hálózatot (átlagolással kapjuk majd meg a hálózat eredményességét)
TrainSet = 1:1082; %tanuló halmaz
ValidationSet = 1083:1444; %validációs halmaz
%KnownSet=1:1444;
TestSet = 1445:1806; %test halmat
Variables = 10; %változók
cv=2; %cross validation ciklus
indices=crossvalind('Kfold',1444,6);

for l=1:cv
%Data=circshift(Data,500);
%lnprice=circshift(lnprice,500);
[TrainInput,ps] = mapminmax(Data'); %0-1 tartományra transzformáljuk a training set input adatokat 
[TrainTarget,ts] = mapminmax(lnprice'); %0-1 tartományra transzformáljuk a training set target adatokat  
TestInput = mapminmax('apply',TestData',ps); % [0,1] tartományra transzformáljuk az adatokat

for j=1:numel(neuron)
    for k=1:repeatN
        
inputs = TrainInput; %inputként megadjuk a transzformált adatokat 
targets = TrainTarget; %targetként megadjuk a transzformált adatokat 

setdemorandstream(491218380+k) %beállítjuk a randomszám generátort hogy késõbb vissza tudjuk nézni az eredményeket 
hiddenLayerSize = neuron(j);  %megadjuk a rejett rétegben lévõ neuronok számát
net = fitnet(hiddenLayerSize); 

net.divideFcn = 'divideind'; %felosztjuk az adatokat tanuló és validákó részre
net.divideParam.trainInd = TrainSet;  %tanuló rész
net.divideParam.valInd   = ValidationSet; %validáló rész
net.inputs{1}.processFcns = {}; %beállítjuk hogy már ne transzformálja a hálózat az adatainkat hiszen mi már megtettük 
net.outputs{2}.processFcns = {}; %beállítjuk hogy már ne transzformálja a hálózat az adatainkat hiszen mi már megtettük 
net.trainFcn = 'trainlm';   %a Levengerg-Marquardt tanuló algoritmust használjuk
net.performFcn = 'mse';   %mse alapján optimalizál az algoritmus
net.trainParam.max_fail=6;
net.trainParam.mu=0.01;

[net,tr] = train(net,inputs(:,indices~=l),targets(indices~=l)); %lefuttatjuk a hálózatot 
TrainOutputs = net(inputs(:,indices~=l)); %elõrejelzünk a hálózattal a trainhalmayra
ValidationOutputs=net(inputs(:,indices==l));
TestOutputs = net(TestInput); %elõrejelzünk a hálózattal

DataTrainNN((j-1)*repeatN+k,:)=mapminmax('reverse',TrainOutputs,ts); %elmentjük a már visszatranszformált outputadatokat
DataValidationNN((j-1)*repeatN+k,:)=mapminmax('reverse',ValidationOutputs,ts); %elmentjük a már visszatranszformált outputadatokat
DataTestNN((j-1)*repeatN+k,:)=mapminmax('reverse',TestOutputs,ts); %elmentjük a már visszatranszformált outputadatokat

    end
DataMeanTrainNN(j,:)=mean(DataTrainNN(((j-1)*repeatN+1):(j*repeatN),:)); %átlagoljuk az egyes eredményeket amelyek azonos rejtett rétegben lévõ neuronszámhoz kapcsolódnak
DataMeanValidationNN(j,:)=mean(DataValidationNN(((j-1)*repeatN+1):(j*repeatN),:)); %átlagoljuk az egyes eredményeket amelyek azonos rejtett rétegben lévõ neuronszámhoz kapcsolódnak
DataMeanTestNN(j,:)=mean(DataTestNN(((j-1)*repeatN+1):(j*repeatN),:)); %átlagoljuk az egyes eredményeket amelyek azonos rejtett rétegben lévõ neuronszámhoz kapcsolódnak

%rmseTestNN(j,1)=sqrt(sum((lnprice(TestSet)'-DataMeanTestNN(j,:)).^2))/numel(TestSet); %rmse a test halmazon

CrossvalTrainNN(l,j,:)=DataMeanTrainNN(j,:);%lementjük ay l. corssvalidation ciklus eredmenyet
CrossvalTestNN(l,j,:)=DataMeanTestNN(j,:);%lementjük ay l. corssvalidation ciklus eredmenyet
CrossvalValidationNN(l,j,:)=DataMeanValidationNN(j,:);%lementjük ay l. corssvalidation ciklus eredmenyet

end
end
%%
for j=1:numel(neuron)
CrossvalMeanTrainNN(j,:)=mean(CrossvalTrainNN(:,j,:),1);%kiátlagoljuk a különböy? CV ciklusokat
CrossvalMeanValidationNN(j,:)=mean(CrossvalValidationNN(:,j,:),1);%kiátlagoljuk a különböy? CV ciklusokat
CrossvalMeanTestNN(j,:)=mean(CrossvalTestNN(:,j,:),1);%kiátlagoljuk a különböy? CV ciklusokat
rmseTrainNN(j,1)=sqrt(sum((lnprice(indices~=l)'-CrossvalMeanTrainNN(j,:)).^2))/1083; %rmse a train halmazon
rmseValidationNN(j,1)=sqrt(sum((lnprice(indices==l)'-CrossvalMeanValidationNN(j,:)).^2))/361; %rmse a validációs halmazon
end

%%
z=sort(rmseValidationNN); %sorbarakjuk az rmse értékeket
optimalneuron=find(rmseValidationNN==z(1,1));   %kiválsztjuk melyikhez tartozik a legkisebb rmse, ez lesz az optimális hálózat

%% regression
mdl = fitlm(Data(1:ValidationSet(end),1:Variables),lnprice(1:ValidationSet(end),:)); %regressziós modell becslése

DataTrainR = predict(mdl,Data(TrainSet,1:Variables)); %a becsült regressziós modellel elõrejelzünk a train adatokra
DataValidationR = predict(mdl,Data(ValidationSet,1:Variables)); %a becsült regressziós modellel elõrejelzünk a validációs adatokra
DataTestR = predict(mdl,TestData(:,1:Variables)); %a becsült regressziós modellel elõrejelzünk a teszt adatokra

rmseTrainR=sqrt(sum((lnprice(TrainSet)-DataTrainR).^2))/numel(TrainSet); %rmse a train halmazon
rmseValidationR=sqrt(sum((lnprice(ValidationSet)-DataValidationR).^2))/numel(ValidationSet); %rmse a validációs halmazon
%rmseTestR=sqrt(sum((lnprice(TestSet)-DataTestR).^2))/numel(TestSet); %rmse a test halmazon

%% figures
figure(1)
y=exp(lnprice(1:1083)'); %validációs halmaz lakásainak eredeti árai
yNN=exp(CrossvalMeanTrainNN(optimalneuron,:)); %validációs halmaz lakásainak becsült árai neuraális hálóval
yR=exp(DataTrainR); %validációs halmaz lakásainak becsült árai regresszióval
scatter(yNN, y, 30, 'r', 'fill'); %scatter plottaé ábrázoljuk az adatokat 
axis([0 100 0 100]); %beállítjuk a tengelyeket 
set(gca,'YTick',0:20:100) %beállítjuk az y tengely felosztását
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az y tengely megjelenítését
set(gca,'XTick',0:20:100) %beállítjuk az x tengely felosztását
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az x tengely megjelenítését
xlabel('Lakásárak becslése neurális hálóval és regresszióval'); %az x tengelynek feliratot adnunk
ylabel('Lakásárak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal ábrázoljuk az adatokat
hleg1 = legend([strcat('Neurális háló MSE: ', {''}, num2str(rmseTrainNN(optimalneuron)),''), strcat('Regresszió MSE: ', {''}, num2str(rmseTrainR),'')]);
set(hleg1,'Location','NorthWest') %beállítjuk, hogy látszódjon a felirat hogy melyik pontok melyik becsléshez tartoznak
title('TrainSet')  

figure(2)
y=exp(lnprice(1083:1444)'); %validációs halmaz lakásainak eredeti árai
yNN=exp(DataMeanValidationNN(optimalneuron,:)); %validációs halmaz lakásainak becsült árai neuraális hálóval
yR=exp(DataValidationR); %validációs halmaz lakásainak becsült árai regresszióval
scatter(yNN, y, 30, 'r', 'fill'); %scatter plottaé ábrázoljuk az adatokat 
axis([0 100 0 100]); %beállítjuk a tengelyeket 
set(gca,'YTick',0:20:100) %beállítjuk az y tengely felosztását
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az y tengely megjelenítését
set(gca,'XTick',0:20:100) %beállítjuk az x tengely felosztását
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az x tengely megjelenítését
xlabel('Lakásárak becslése neurális hálóval és regresszióval'); %az x tengelynek feliratot adnunk
ylabel('Lakásárak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal ábrázoljuk az adatokat
hleg1 = legend([strcat('Neurális háló MSE: ', {''}, num2str(rmseValidationNN(optimalneuron)),''), strcat('Regresszió MSE: ', {''}, num2str(rmseValidationR),'')]);
set(hleg1,'Location','NorthWest') %beállítjuk, hogy látszódjon a felirat hogy melyik pontok melyik becsléshez tartoznak
title('ValidationSet')  

%%
figure(3)
y=exp(lnprice(1084:1444)'); %validációs halmaz lakásainak eredeti árai
yNN=exp(CrossvalMeanTestNN(optimalneuron,:)); %validációs halmaz lakásainak becsült árai neuraális hálóval
yR=exp(DataTestR); %validációs halmaz lakásainak becsült árai regresszióval
scatter(yNN, y, 30, 'r', 'fill'); %scatter plottaé ábrázoljuk az adatokat 
axis([0 100 0 100]); %beállítjuk a tengelyeket 
set(gca,'YTick',0:20:100) %beállítjuk az y tengely felosztását
set(gca,'YTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az y tengely megjelenítését
set(gca,'XTick',0:20:100) %beállítjuk az x tengely felosztását
set(gca,'XTickLabel',{'0','20','40','60','80','100'}) %beállítjuk az x tengely megjelenítését
xlabel('Lakásárak becslése neurális hálóval és regresszióval'); %az x tengelynek feliratot adnunk
ylabel('Lakásárak'); %az y tengelynek feliratot adnunk

grid on
hold on
scatter(yR, y, 30, 'g', 'fill'); %scatter plottal ábrázoljuk az adatokat
hleg1 = legend([strcat('Neurális háló MSE: ', {''}, num2str(rmseTestNN(optimalneuron)),''), strcat('Regresszió MSE: ', {''}, num2str(rmseTestR),'')]);
set(hleg1,'Location','NorthWest') %beállítjuk, hogy látszódjon a felirat hogy melyik pontok melyik becsléshez tartoznak
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