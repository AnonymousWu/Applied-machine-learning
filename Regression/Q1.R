install.packages('leaps')
library(leaps)

mydata = read.table("brunhild.txt",header=TRUE)
Hours = mydata[,c(1)]
Sulfate = mydata[,c(2)]

#-----(a) Regression line in log-log coordinate -----
logHours = log(Hours)
logSulfate = log(Sulfate)

# scatter plot of the data and regression line
plot(logHours, logSulfate, main="log-log Plot", 
     xlab="logHours", ylab="logSulfate ", pch=19)
abline(lm.logsulfate)

# build regression line in log-log coordinate
lm.logsulfate <-lm(logSulfate~logHours)
summary(lm.logsulfate,cor=F)
plot(lm.logsulfate)

#-----(b) Regression curve in the original coordinate-----

# scatter plot of the data and regression line
plot(Hours, Sulfate, main="Original Coordinate Plot", 
     xlab="Hours", ylab="Sulfate ", pch=19)
abline(lm.sulfate)

# build regression line in original coordinate
lm.sulfate <-lm(Sulfate~Hours)
summary(lm.sulfate,cor=F)
plot(lm.sulfate)
