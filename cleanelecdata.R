library(dplyr)
library(dygraphs)
library(feasts)
library(forecast)
library(ggplot2)
library(ggExtra) 
library(highcharter) 
library(janitor)
library(lubridate)
library(plotly)
library(tidyverse)
library(tidyr) 
library(tsibble)
library(tsibbledata)
library(TSstudio)
library(UKgrid)
#library(virids)
library(xts)
library(zoo)
library(tseries)
library(pracma)
library(imputeTS)
library(timetk)

#Reading the data
normdata<- read.csv('elec_quantity_nt_orig_CurrentElectricityQuantity_5yrhours.csv', header = FALSE)
eleclow<- read.csv('elec_quantity_lt_orig_CurrentElectricityQuantity_5yrhours.csv', header = FALSE)

#Changing the epoch time to human-readable time and date
normdata$date<- as.POSIXct(normdata$V1, origin = '1970-01-01 19:00:00', tz= 'UTC')
eleclow$date<- as.POSIXct(eleclow$V1, origin = '1970-01-01 19:00:00', tz= 'UTC')
eleclow$date<- ymd_hms(eleclow$date)


#Changing the name of electricity consumption column
normdata$usage<- as.numeric(normdata$V2)
eleclow$usage<- as.numeric(eleclow$V2)

#For loop to substract the rows from each other - Normal tarif
normdata$usages <- NA

for(i in 1:(nrow(normdata) - 1)) {
  normdata$usages[i] = normdata$usage[i+1] - normdata$usage[i]
}

#For loop to substract the rows from each other - Low tarif
eleclow$usages <- NA

for(i in 1:(nrow(eleclow) - 1)) {
  eleclow$usages[i] = eleclow$usage[i+1] - eleclow$usage[i]
}


#Deleting column V1, V2
normdata<- normdata[-1]
normdata<- normdata[-1]
eleclow<- eleclow[-1]
eleclow<- eleclow[-1]
normdata<- normdata[-2]
eleclow<- eleclow[-2]

#Joining the normal and low tarif
elecdata <-full_join(eleclow, normdata, by = 'date') 

#Adding up the low and normal tarive in one column
elecdata$usage <- elecdata$usages.x + elecdata$usages.y
#-------------------------------------------------------------------------------------------------------------
#Creating new column with day and night time, based on Toon app and after deleting the time again
elecdata$time<- format(as.POSIXct(elecdata$date),
                       format = '%H:%M:%S')

elecdata$daynight <- ifelse(elecdata$time >= '22:00:00' | elecdata$time <= '05:00:00', 0,1)

#Deleting the time column again
elecdata <- elecdata[-5]
#-------------------------------------------------------------------------------------------------------------
#~
#-------------------------------------------------------------------------------------------------------------

#EXPLORATARY DATA ANALYSIS
head(elecdata)
summary(elecdata)
dim(elecdata)

#Checking which columns are missing data
list_na <- colnames(elecdata)[ apply(elecdata, 2, anyNA) ]
list_na
#"usages.x" "usages.y" "usage" 

#Missing part change to 0
elecdata[,colnames(elecdata) %in% list_na]
#Delete last row
elecdata<- elecdata[-15948,]

write.csv2(elecdata, 'elecdataG.csv')
#elec_tbl_ts$day <- date(elec_tbl_ts$date)
#elec_tbl_ts$time <- as.integer(substr(elec_tbl_ts$date,12,13))
