#############################################################################################################################################
#############################################################################################################################################
############################      FACTORS INFLUENCING CLIENT SUBSCRIPTION                                   #################################
############################                               TO ANNUAL TERM DEPOSIT                           #################################
############################                                USING DATA MINING TECHNIQUE                     #################################
#############################################################################################################################################
#############################################################################################################################################



####################################################Attribute Information: ##################################################################
#############################################################################################################################################  

#####################################             Dependent variables: ######################################################################

###                                                bank client data:                                    ###
###                                                                                                     ###


##########  1 - age (numeric) (young age 0-30, middle age 30-50 , senior age >50)
##########  2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
##########  3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
##########  4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
##########  5 - default: has credit in default? (categorical: 'no','yes','unknown')
##########  6 - housing: has housing loan? (categorical: 'no','yes','unknown')
##########  7 - loan: has personal loan? (categorical: 'no','yes','unknown')



###                                related with the last contact of the current campaign:                 ###
###                                                                                                       ###


##########  8 - contact: contact communication type (categorical: 'cellular','telephone') 
##########  9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
##########  10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
##########  11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.




###                                other attributes:                                                      ###
###                                                                                                       ###   



##########  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
##########  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
##########  14 - previous: number of contacts performed before this campaign and for this client (numeric)
##########  15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')


###                          social and economic context attributes                                       ###
###                                                                                                       ###


##########  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
##########  17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
##########  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
##########  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
##########  20 - nr.employed: number of employees - quarterly indicator (numeric)

##########                      Output variable (desired target):                                         ###
##########                                                                                                ###
 

########## 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')


##################################### Set the working directory  ###########################################################################

setwd("C:/Users/chira/OneDrive/Desktop/ADM REPEAT/CSV/bankfull") 

#####################################  Library Assignment        ###########################################################################

library('dplyr')
library('ROCR')
library('Metrics')
library('caret')
library('randomForest')
library('ggplot2')
library('ggthemes')
library('Boruta')
library('tidyr')
library('lazyeval')
library('VIM')
library('ebmc')
library('e1071')
library(corrplot)
library(caTools)
library(C50)




################################################# Import BANK data into R    ###########################################################

bank <- read.csv("Bank.csv", stringsAsFactors = T , na.strings=c("unknown")) 
dim(bank)
head(bank)




######################################## Dealing with Attributes          ##################################################################
######################################### as given in description         ##################################################################


bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$contact <- as.factor(bank$contact)
bank$month <- as.factor(bank$month)
bank$day_of_week <- as.factor(bank$day_of_week)
bank$poutcome <- as.factor(bank$job)
bank$y <- as.factor(bank$y)
bank$ï..age <- as.numeric(bank$ï..age)
bank$duration <- as.numeric(bank$duration)
bank$campaign <- as.factor(bank$campaign)
bank$pdays <- as.numeric(bank$pdays)
bank$previous <- as.factor(bank$previous)
bank$emp.var.rate <- as.numeric(bank$emp.var.rate)
bank$cons.price.idx <- as.numeric(bank$cons.price.idx)
bank$cons.conf.idx <- as.numeric(bank$cons.conf.idx)
bank$euribor3m <- as.numeric(bank$euribor3m)
bank$nr.employed <- as.numeric(bank$nr.employed)
summary(bank)
################################################# Missing data analysis
aggr_plot <- aggr(bank, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of Missing values","Pattern of Missings"))


##### Removing variables due to high missing values  #######################

bankdat <- bank[,-4]


###### Missing Values Imputation using Mice function ###########################################
library(mice)
md.pattern(bankdat)

bankdata <- mice(bankdat,m=5,maxit=50,meth='pmm',seed=500)
summary(bandata)


################################################ Collinearity check     ############################################

bankdata %>%
  filter(y == "yes") %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot::corrplot()

################################################ Checking the distribution             ##############################


bankdata %>% 
  select_if(is.numeric) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")


cdplot(y ~ ï..age , data = bankdata, main = "TM")
cdplot(y ~ duration , data = bankdata, main = "TM")



##########################FEATURE ENGINEERING ###########################################################################################
##################################################### feature engineering for duration ###################################################

bankdata$duration <- cut(bankdata$duration,
                         breaks = c(0,300,1000,max(bankdata$duration)),
                         labels=c("low", "medium", "high"))

bankdata$duration <- factor(bankdata$duration)


###################################################### feature engineering for variation rate 
bankdata$emp.var.rate <- cut(bankdata$emp.var.rate,
                             breaks = c(min(bankdata$emp.var.rate),0,max(bankdata$emp.var.rate)),
                             labels=c("low", "high"))


bankdata$emp.var.rate <- factor(bankdata$emp.var.rate)

############################################### feature engineering euribor variable

bankdata$euribor3m <- cut(bankdata$euribor3m,
                          breaks = c(0,1.5,3,max(bankdata$euribor3m)),
                          labels=c("low", "medium" ," high"))

bankdata$euribor3m <- factor(bankdata$euribor3m)

#################################################### feature engineering on cons price index ################
bankdata$cons.price.idx <- cut(bankdata$cons.price.idx,
                               breaks = c(92,92.5,93,max(bankdata$cons.price.idx)),
                               labels=c("less_change", "medium_change", "high_change"))


################################################# feature engineering on cons confidence index ################
bankdata$cons.conf.idx <- cut(bankdata$cons.conf.idx,
                              breaks = c(min(bankdata$cons.conf.idx),-41.80,max(bankdata$cons.conf.idx)),
                              labels=c("less_likely", "more_likely"))


summary(bankdata)



#################################################   Feature selection              #########################################################

###################    1.)   Feature selection using Random Forest

model_rf<-randomForest(y ~ ., data = bankdata)
importance    <- importance(model_rf)
order(importance(model_rf))


##################     2.)   Feature selection using Boruta 


boruta_output <- Boruta(y ~ ., data=bankdata, doTrace=2)  # perform Boruta search Number of casualities
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance") 




#######################Sampling Data ################################################


spl=sample.split(bankdata$y,SplitRatio=0.8)
train=subset(bankdata, spl==TRUE)
test=subset(bankdata, spl==FALSE)




################################### Modelling         ############################################
################################### c5.0 Model        ############################################


##C5.0 hold out

bank_model <- C5.0(y ~.,data=train,trials = 20) 

##C5.0 cross validation

bank_model <- C5.0(y ~.,data=train, control = C5.0Control(winnow = FALSE))

##K-fold cross validation

#Sample
x <- sample(40000,6000)
bankdata <- bankdata[x,] 

control <- trainControl(method="repeatedcv", number=5, repeats=10) #5 x 10-fold cv
metric <- "Kappa" 
train(y~., data=bankdata, method="C5.0", metric=metric, trControl=control)


plot(bank_model)

bank_pred <- predict(bank_model,test)
library(gmodels)
library(caret)
CrossTable(test$y, bank_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)

confusionMatrix(bank_pred, test$y, positive = "Yes")





################################### Naive Bayes Model ############################################
################################### Naive Bayes Model laplace = 0#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm

#### Checking Effect of call duration on the outcome y #################################
summary(bankdata)
bankdur <-  bankdata[,-2]

bankdata$duration <- NULL


################################### Naive Bayes Model laplace = 0#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankdata.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankdata.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



#########################################################################################
##########################################################################################
#### Checking Effect of call duration on the outcome y #################################
summary(bankdata)
bankdur <-  bankdata[,-10]


spl=sample.split(bankdur$y,SplitRatio=0.8)
train=subset(bankdur, spl==TRUE)
test=subset(bankdur, spl==FALSE)

################################### Naive Bayes Model laplace = 0#################################

bankdur.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankdur.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankdur.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankdur.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankdur.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankdur.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


#########################################################################################
##########################################################################################
#### Checking Effect of confidence index on the outcome y #################################


bankconf <-  bankdata[,-17]
summary(bankconf)

spl=sample.split(bankconf$y,SplitRatio=0.8)
train=subset(bankconf, spl==TRUE)
test=subset(bankconf, spl==FALSE)

################################### Naive Bayes Model laplace = 0#################################

bankconf.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankconf.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankconf.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankconf.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankconf.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankconf.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm




#########################################################################################
##########################################################################################
#### Checking Effect of pdays on the outcome y #################################


bankpd <-  bankdata[,-12]
summary(bankpd)

spl=sample.split(bankpd$y,SplitRatio=0.8)
train=subset(bankpd, spl==TRUE)
test=subset(bankpd, spl==FALSE)

################################### Naive Bayes Model laplace = 0#################################

bankpd.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankpd.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankpd.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankpd.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankpd.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankpd.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



#########################################################################################
##########################################################################################
#### Checking Effect of previous contact on the outcome y #################################


bankpr <-  bankdata[,-13]
summary(bankpr)

spl=sample.split(bankpr$y,SplitRatio=0.8)
train=subset(bankpr, spl==TRUE)
test=subset(bankpr, spl==FALSE)

################################### Naive Bayes Model laplace = 0#################################

bankpr.nb <- naiveBayes(y ~ ., data = train,laplace = 0)

nb_train_predict <- predict(bankpr.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm



################################### Naive Bayes Model laplace = 1#################################

bankpr.nb <- naiveBayes(y ~ ., data = train,laplace = 1)

nb_train_predict <- predict(bankpr.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm


################################### Naive Bayes Model laplace = 2#################################

bankpr.nb <- naiveBayes(y ~ ., data = train,laplace = 2)

nb_train_predict <- predict(bankpr.nb, test[ , names(test) != "y"])

################################### Accuracy check  ###################################
cfm <- confusionMatrix(nb_train_predict, test$y)
cfm

