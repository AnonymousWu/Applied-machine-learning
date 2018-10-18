install.packages('leaps')
library(leaps)

mydata = read.table("physical.txt",header=TRUE)

Mass = mydata[,c(1)]
Fore = mydata[,c(2)]
Bicep	= mydata[,c(3)]
Chest	= mydata[,c(4)]
Neck = mydata[,c(5)]
Shoulder = mydata[,c(6)]	
Waist	= mydata[,c(7)]
Height = mydata[,c(8)]	
Calf = mydata[,c(9)]	
Thigh = mydata[,c(10)]	
Head = mydata[,c(11)]

lm.mass <-lm(Mass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head)
summary(lm.mass,cor=F)
plot(lm.mass)

# plot the residual against the fitted values 
#in the original coordinates
cubeRootMass = Mass^(1/3)
lm.mass <-lm(cubeRootMass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head)
summary(lm.mass,cor=F)
plot(lm.mass)

# plot the residual against the fitted values 
#in the cube root coordinates.
crFore = Fore^(1/3)
crBicep	= Bicep^(1/3)
crChest	= Chest^(1/3)
crNeck = Neck^(1/3)
crShoulder = Shoulder^(1/3)
crWaist	= Waist^(1/3)
crHeight = Height^(1/3)
crCalf = Calf^(1/3)	
crThigh = Thigh^(1/3)	
crHead = Head^(1/3)
lm.crmass <-lm(cubeRootMass~crFore+crBicep+crChest+crNeck+crShoulder+crWaist+crHeight+crCalf+crThigh+crHead)
summary(lm.crmass,cor=F)
plot(lm.crmass)
