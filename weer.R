2
library(ggplot2)
library(dplyr)


#dataset 1
weather<- read.csv('uurgeg.csv')
weather <- weather[78921:87704,]
weather <- weather[,-1]
names(weather)[names(weather) == 'X'] <- 'date'

#dataset 2
weat2022<- read.csv('uurgeg2022.csv')
weat2022 <- weat2022[33:15152,]
weat2022<- weat2022[,-1]
names(weat2022)[names(weat2022) == 'X'] <- 'date'

#combining both datasets
weatdata<-rbind(weather, weat2022)
weatdata<- as_data_frame(weatdata)

#
weatdata <- weatdata[7907:23887,]

#changing the date to date format
weatdata$date<- as.Date(weatdata$date, format = '%Y %m %d')

names(weatdata)[names(weatdata) == 'X.1'] <- 'time'
weatdata$time<- as.POSIXlt(weatdata$time, format = '%H')
weatdata$hour<- hour(weatdata$time )

weatdata$date<- paste(weatdata$date, weatdata$hour)
weatdata<- weatdata[-2]

#weatdata$time <- format(strptime(formatC(weatdata$time, width = 2, format = 'd', flag = '0') , format = "%H"),format = "%H")


weatdata<- weatdata[-c(6039:7006),]
weatdata<- weatdata[-c(7530:7544),]
weatdata<- weatdata[-c(12635:12817),]

write_csv(weatdata, 'weerbewerkt.csv')


