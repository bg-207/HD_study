glm = fitglm([file name],'DiseaseStage ~ 1 + V50WordsinBN','distribution','binomial','link','logit')
Y = GLMwordsinnoiseSNRnegative1premanifestvscontrols24042020{:,'DiseaseStage'};
cvfolds = crossvalind('KFold',42,5);
cp = classperf(Y);

for i = 1:5
    testIdx = (cvfolds == i);
    trainIdx = ~testIdx;
    XTrain = GLMwordsinnoiseSNRnegative1premanifestvscontrols24042020(trainIdx, :);
    XTest = GLMwordsinnoiseSNRnegative1premanifestvscontrols24042020(testIdx, :);
    b = GLMwordsinnoiseSNRnegative1premanifestvscontrols24042020{:,'DiseaseStage'};
    yTrain = b(trainIdx);
    yTest = b(testIdx);
    class = fitglm(XTrain,'DiseaseStage ~ 1 + V50WordsinBN','distribution','binomial','link','logit')
    output = predict(class,XTest)
    T= array2table(output)
    a = XTest(:,'DiseaseStage');
    classifierOutput = round(predict(class,XTest))
    classperf(cp,classifierOutput,testIdx);
    Outputtotal{i} = T;
    Inputtotal{i} = a;
end

x = [Inputtotal,Outputtotal];
X2 = [x{1,1};x{1,2};x{1,3};x{1,4};x{1,5}];
Y2 = [x{1,6};x{1,7};x{1,8};x{1,9};x{1,10}];
X2array = table2array(X2);
Y2array = table2array(Y2);
X = X2array;
Y = Y2array;
[X,Y,T,AUC,OPTROCPT] = perfcurve(X,Y,'1');
    AUC
    plot(X,Y)
    hold on
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC for Classification by Logistic Regression')
opt = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)))
youdens = [Y-X(:)]
[val, idx] = max(youdens);
youdensindex = T(([idx]))
    plot(X([idx]),Y([idx]), 'bo')
    plot(OPTROCPT(1),OPTROCPT(2),'ro')
    hold off
Y3 = round(Y2array >= youdensindex)

fig = figure;
[m,order]= confusionmat(X2array,Y3)
B = categorical(order,[0 1],{'Controls' 'Pre manifest'}); %naming the titles in the confusionmatrix
categories(B)
cm = confusionchart(m,B,'RowSummary','row-normalized','ColumnSummary','column-normalized');
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1;
fig.Position = fig_Position;
cm.Normalization = 'row-normalized'; 
cm.Normalization = 'absolute'; 
cm.Title = 'Discriminating Controls and Pre manifest using a 5 minute speech-in-noise task'
ci=coefCI(glm)