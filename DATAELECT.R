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
elecdata<- read.csv2('elecdataG.csv', header = TRUE)
elecdata<- elecdata[-1]
elecdata$date<- ymd_hms(elecdata$date)

#---------------------------------------------------------------------------------------
#Building a timeseries table
elec_tbl<- elecdata%>%
  select(date, usage)

elec_tbl %>% as_tsibble(index = date) -> elec_tbl_ts
print(elec_tbl_ts)
class(elec_tbl)
class(elec_tbl_ts)

#Still some missing values
missing<-scan_gaps(elec_tbl_ts)
#34 random rows missing
#Prepare missing dates 
elec_tbl_ts<- elec_tbl_ts %>% 
  group_by_key() %>% 
  fill_gaps()

moving2<- rollmean(elec_tbl_ts$usage, k = 200)
head(moving2)
df2 = data.frame(moving2)


#filling NA's with moving average
elec_tbl_ts$usage<- na_ma(elec_tbl_ts$usage, k = 2, weighting = 'exponential', maxgap = 10)
#simple and linear nog proberen 
elec_tbl_ts$usage[c(7006:7007)] <- elec_tbl_ts$usage[c(7006:7007)]/2
##elec_tbl_ts$usage[c(8497)] <- moving2[8497]
#??
elec_tbl_ts$usage[c(8497:8511)] <- 0
elec_tbl_ts$daynight[c(8497:8507)] <- 1
elec_tbl_ts$daynight[c(8508:8511)] <- 0
elec_tbl_ts$usage[c(6375:7005)] <- 0


#-----------------------------------------------------------------------------------------
#Simple plots
boxplot(elec_tbl_ts$date)
#plot
ggplot(data = elec_tbl_ts, aes(x = date, y = usage)) +
  geom_line() +
  geom_smooth(se = FALSE) +
  labs(x = "Monthly Data", y = "Energy ")+
  scale_x_datetime(date_breaks = "month", date_labels = "%y-%m")
#All years
elec_tbl_ts %>% gg_season(usage, labels = "both")
#Only 2020
g1<- elec_tbl_ts%>%
  filter(year(date) == 2020)%>%
  gg_season(usage)
g2<- elec_tbl_ts%>%
  filter(year(date) == 2021)%>%
  gg_season(usage)
g3<- elec_tbl_ts%>%
  filter(year(date) == 2022)%>%
  gg_season(usage)

gridExtra::grid.arrange(g1, g2, g3)

#Interactive Graph
p <- elec_tbl_ts %>%
  ggplot( aes(x=date, y=usage)) +
  geom_area(fill="#69b3a2", alpha=0.5) +
  geom_line(color="#69b3a2") +
  xlab("Date per hour")
  ylab("Energy usage") 
p <- ggplotly(p)
p

(time_plot <- ggplot(elec_tbl_ts, aes(x = date, y = usage)) +
    geom_line() +
    scale_x_datetime(date_labels = "%m", date_breaks = "2 month") +
    theme_classic())

elec_tbl_ts<- elec_tbl_ts[-c(6039:7006),]
elec_tbl_ts<- elec_tbl_ts[-c(7530:7544),]
elec_tbl_ts$daynight[c(7194:7199)] <- 1
elec_tbl_ts<- elec_tbl_ts[-c(12635:12817),]
elec_tbl_ts$usage[c(8371:8384)]<- elec_tbl_ts$usage[c(8371:8384)]/2
elec_tbl_ts$usage[3625]<- elec_tbl_ts$usage[3625]/2
#strategy to answer Q///metho is how which method
#average moving  smoothing for visualization
#https://en.wikipedia.org/wiki/Moving_average
#Trend #https://rc2e.com/timeseriesanalysis
#---------------------------------------------------------------------------------------

#elec_tbl_ts<- elec_tbl_ts[-1,]
x<- matrix(elec_tbl_ts$usage)
x2 <- ymd_hms(elec_tbl_ts$date)
tsding2<- xts(x, order.by = x2)
test1<- ts(x2,x, frequency = 24)
#elec_tbl_ts$date<- ymd_hms(elec_tbl_ts$date)
#times_elec = ts(elec_tbl_ts$usage, frequency = 1440)

tsding2<-ts(tsding2, frequency = 24)

p1 <- autoplot(tsding2) +
  ylab("usage") + xlab("days")

p2 <- autoplot(window(tsding2, 1, end=22)) +
  ylab("usage") + xlab("weeks")

gridExtra::grid.arrange(p1,p2)


tsding2 %>% tail(24*7*4) %>% 
  decompose() %>% 
  autoplot()

msts_cons<-tsding2 %>% msts(seasonal.periods = c(24, 24*7))
msts_cons  %>% head(  24 *7 *4 ) %>% mstl() %>% autoplot()    
#https://rpubs.com/AlgoritmaAcademy/multiseasonality

#tsding2 %>%  stlf() %>%
#autoplot() + xlab("Week")
moving<- rollmean(tsding2, k = 200)
head(moving)
df = data.frame(moving)
autoplot(moving)


hampts<-hampel(tsding2, 5, 10)
hampts$ind
#HAMPEL OUTLIER , 5 3 IMPLACE: TRUE
#omad <- hampel(tsding2, k=20, t0 = 5)

ggAcf(tsding2)
aelec <- window(tsding2, 40, 420)
autoplot(aelec) + xlab("Year") + ylab("GWh")
ggAcf(aelec)

elec_tbl_ts %>% gg_season(usage, labels = "both")


ggsubseriesplot(tsding2)

hi<- stl(test1, s.window = "per")
?stl
plot(hi)

dframe <- cbind(elec_tbl_ts, Monthly = elec_tbl_ts$usage,
                DailyAverage = elec_tbl_ts$usage/monthdays(elec_tbl_ts$usage))
autoplot(dframe, facet=TRUE) 


