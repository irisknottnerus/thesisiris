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
gasdata<- read.csv('gas_quantity_CurrentGasQuantity_5yrhours.csv', header = FALSE)

#Changing the epoch time to human-readable time and date
gasdata$date<- as.POSIXct(gasdata$V1, origin = '1970-01-01 19:00:00', tz= 'UTC')
gasdata$date<- ymd_hms(gasdata$date)


#Changing the name of electricity consumption column
gasdata$usage<- as.numeric(gasdata$V2)

#For loop to substract the rows from each other - Low tarif
gasdata$usages <- NA

for(i in 1:(nrow(gasdata) - 1)) {
  gasdata$usages[i] = gasdata$usage[i+1] - gasdata$usage[i]
}

gasdata$usage<- gasdata$usages
#Deleting column V1, V2
gasdata<- gasdata[-1]
gasdata<- gasdata[-1]
gasdata<- gasdata[-3]
#-------------------------------------------------------------------------------------------------------------
#Creating new column with day and night time, based on Toon app and after deleting the time again
gasdata$time<- format(as.POSIXct(gasdata$date),
                       format = '%H:%M:%S')

gasdata$daynight <- ifelse(gasdata$time >= '22:00:00' | gasdata$time <= '05:00:00', 0,1)

#Deleting the time column again
gasdata <- gasdata[-3]
#-------------------------------------------------------------------------------------------------------------
#~
#-------------------------------------------------------------------------------------------------------------

#EXPLORATARY DATA ANALYSIS
head(gasdata)
summary(gasdata)
dim(gasdata)

#Checking which columns are missing data
list_na <- colnames(gasdata)[ apply(gasdata, 2, anyNA) ]
list_na

#Missing part change to 0
hi<-gasdata[,colnames(gasdata) %in% list_na]
#Delete last row
gasdata<- gasdata[-15319,]

#elec_tbl_ts$day <- date(elec_tbl_ts$date)
#elec_tbl_ts$time <- as.integer(substr(elec_tbl_ts$date,12,13))
#-----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#Building a timeseries table
gastbl<- gasdata%>%
  select(date, usage)

gastbl %>% as_tsibble(index = date) -> gas_tbl_ts
print(gas_tbl_ts)
class(gastbl)
class(gas_tbl_ts)

#Still some missing values
missinggass<-scan_gaps(gas_tbl_ts)
#34 random rows missing
#Prepare missing dates 
gas_tbl_ts<- gas_tbl_ts %>% 
  group_by_key() %>% 
  fill_gaps()

moving1<- rollmean(gas_tbl_ts$usage, k = 200)
head(moving1)
df = data.frame(moving1)


#filling NA's with moving average
gas_tbl_ts$usage<- na_ma(gas_tbl_ts$usage, k = 2, weighting = 'exponential', maxgap = 10)
#simple and linear nog proberen 
gas_tbl_ts$usage[c(7007)] <- moving1[7007]
gas_tbl_ts$usage[c(7007)] <- 38
#gas_tbl_ts$usage[c(8497)] <- moving1[8497]
#??
gas_tbl_ts$usage[c(8498:8511)] <- 0
gas_tbl_ts$daynight[c(8498:8507)] <- 1
gas_tbl_ts$daynight[c(8508:8511)] <- 0
gas_tbl_ts$usage[c(6375:7003)] <- 0
gas_tbl_ts$daynight[c(6375:7003)] <- 0

#-----------------------------------------------------------------------------------------
#Simple plots
boxplot(gas_tbl_ts$date)
#plot
ggplot(data = gas_tbl_ts, aes(x = date, y = usage)) +
  geom_line() +
  geom_smooth(se = FALSE) +
  labs(x = "Monthly Data", y = "Energy ")+
  scale_x_datetime(date_breaks = "month", date_labels = "%y-%m")
#All years
gas_tbl_ts %>% gg_season(usage, labels = "both")
#Only 2020
g1<- gas_tbl_ts%>%
  filter(year(date) == 2020)%>%
  gg_season(usage)
g2<- gas_tbl_ts%>%
  filter(year(date) == 2021)%>%
  gg_season(usage)
g3<- gas_tbl_ts%>%
  filter(year(date) == 2022)%>%
  gg_season(usage)

gridExtra::grid.arrange(g1, g2, g3)

#Interactive Graph
p <- gas_tbl_ts %>%
  ggplot( aes(x=date, y=usage)) +
  geom_area(fill="#69b3a2", alpha=0.5) +
  geom_line(color="#69b3a2") +
  xlab("Date per hour")
ylab("Energy usage") 
p <- ggplotly(p)
p

(time_plot <- ggplot(gas_tbl_ts, aes(x = date, y = usage)) +
    geom_line() +
    scale_x_datetime(date_labels = "%m", date_breaks = "2 month") +
    theme_classic())

gas_tbl_ts<- gas_tbl_ts[-c(6039:7006),]
gas_tbl_ts<- gas_tbl_ts[-c(7530:7544),]
gas_tbl_ts$daynight[c(7194:7199)] <- 1
gas_tbl_ts<- gas_tbl_ts[-c(12635:12817),]
gas_tbl_ts$usage[c(8371:8384)]<- gas_tbl_ts$usage[c(8371:8384)]/2
gas_tbl_ts$usage[3625]<- gas_tbl_ts$usage[3625]/2
#strategy to answer Q///metho is how which method
#average moving  smoothing for visualization
#https://en.wikipedia.org/wiki/Moving_average
#Trend #https://rc2e.com/timeseriesanalysis
#---------------------------------------------------------------------------------------

#gas_tbl_ts<- gas_tbl_ts[-1,]
x<- matrix(gas_tbl_ts$usage)
x2 <- ymd_hms(gas_tbl_ts$date)
tsding<- xts(x, order.by = x2)
test1<- ts(x2,x, frequency = 24)
#gas_tbl_ts$date<- ymd_hms(gas_tbl_ts$date)
#times_gas = ts(gas_tbl_ts$usage, frequency = 1440)

tsding1<-ts(tsding, frequency = 24)

p1 <- autoplot(tsding1) +
  ylab("usage") + xlab("days")

p2 <- autoplot(window(tsding1, 1, end=22)) +
  ylab("usage") + xlab("weeks")

gridExtra::grid.arrange(p1,p2)


tsding1 %>% tail(24*7*4) %>% 
  decompose() %>% 
  autoplot()

msts_cons<-tsding1 %>% msts(seasonal.periods = c(24, 24*7))
msts_cons  %>% head(  24 *7 *4 ) %>% mstl() %>% autoplot()    
#https://rpubs.com/AlgoritmaAcademy/multiseasonality

#tsding1 %>%  stlf() %>%
#autoplot() + xlab("Week")
moving<- rollmean(tsding1, k = 200)
head(moving)
df = data.frame(moving)
autoplot(moving)


hampts<-hampel(tsding1, 2, 7)
hampts$ind
#HAMPEL OUTLIER , 5 3 IMPLACE: TRUE
#omad <- hampel(tsding1, k=20, t0 = 5)

ggAcf(tsding1)
aelec <- window(tsding1, 40, 420)
autoplot(aelec) + xlab("Year") + ylab("GWh")
acf(aelec)

hii<- ts(gas_tbl_ts['usage'], start = c(1), frequency = 24)
str(hii)
hii

autoplot(hii, facet=TRUE) 
max(hii)










