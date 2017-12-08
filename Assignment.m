%Loading Data From File
A = load('breast-cancer-wisconsin.data');
InputData=A(:,2:10);
TargetData=A(:,11);

%Pre-processing___ replacing 0 with mean value of 6th column
meanValue=mean(InputData(:,6));
disp(meanValue);
InputData(InputData==0)=round(meanValue);
TargetData(TargetData==2)=-1;
TargetData(TargetData==4)=1;

 Results=zeros(10,3);
 percent=70;% percentage of training input data
for neurons=[10 20 30 40 50]
    disp(percent);
    sizeOfInputData=round((percent/100)*699);
    trainInputData=InputData(1:sizeOfInputData,:);
    trainTargetData=TargetData(1:sizeOfInputData,:);
    
    testInputData=InputData(sizeOfInputData+1:699,:);
    testTargetData=TargetData(sizeOfInputData+1:699,:);


    net = newff(trainInputData',trainTargetData',neurons, {'tansig' 'tansig'}, 'trainr', 'learngd', 'mse');
    net.trainParam.goal = 0.01;
    net.trainParam.epochs = 100;
    net.trainParam.max_fail=10;
    net.trainParam.lr=0.01;
    trainedNet = train(net, trainInputData',trainTargetData');

    output=trainedNet(testInputData');
   
    error=0;
    
    for i=1:size(output,2)
    
        if((output(1,i)<0 &&testTargetData(i,1)~=-1)||(output(1,i)>=0 &&testTargetData(i,1)~=1))
            output(2,i)=testTargetData(i,1);
            error=error+1;
        end;
    
    end;
    disp(((size(testTargetData)-error)/size(testTargetData))*100);
    Results(percent/10,1)=percent;
    Results(percent/10,2)=100-percent;
    Results(percent/10,3)=((size(testTargetData)-error)/size(testTargetData))*100;
end

