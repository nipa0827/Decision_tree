  # Load the party package. It will automatically load other
# dependent packages.
#library(party)

data <- read.csv('ad.csv')
summary(data)


data$Puchased <- as.character(data$Purchased)
data$Purchased[data$Purchased==1] <- 'Yes'
data$Purchased[data$Purchased==0] <- 'No'
data$Purchased <- as.factor(data$Purchased)
#table(data$Purchased)

library(caret)

p=0.70
n.train = as.integer(nrow(data)*p)
indx = sample(1:nrow(data), n.train)
TrainData = data[indx,]
TestData = data[-indx,]

library(party)

png(file="decision_tree.png")
Fit.ctree <- ctree(Purchased ~ User.ID + Gender + Age + EstimatedSalary, data=TrainData )

plot(Fit.ctree)
dev.off()

pred=predict(Fit.ctree,newdata=TestData) # get predicted values for testing data
table(TestData$Purchased,pred) # tabulate actuall and predected class values


k=10 # number of folds
folds=sample(rep_len(1:k,nrow(data))) # generate random fold indices
table(folds)

Acc=c() # define accuracy vector
for(i in 1:k){
  Fit.ctree=ctree(Purchased ~ User.ID + Gender + Age + EstimatedSalary,data=data[folds!=i,]) # fit model on all folds except fold i
  pred=predict(Fit.ctree,newdata=data[folds==i,])  # predict class for fold i
  Acc[i]=sum(pred==data$Purchased[folds==i])/length(data$Purchased[folds==i]) # accuracy for fold i
}

print(mean(Acc))
