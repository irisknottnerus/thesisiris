

gas_tbl_ts$usage<- gas_tbl_ts$usage /1000
gas_tbl_ts$usage<- gas_tbl_ts$usage * 10.55
elec_tbl_ts$usage<- elec_tbl_ts$usage/1000

alldata<- right_join(elec_tbl_ts, gas_tbl_ts, by = 'date')
alldata$usage.x<- alldata$usage.x + alldata$usage.y 
alldata$usage<- alldata$usage.x
alldata$daynight <- alldata$daynight.x
alldata<- alldata[-2] 

write.csv(alldata, 'datalos.csv')
write_csv(alldata, "alldataday.csv")
alldata<- read_csv("alldataday.csv")
alldata$date<- ymd_hms(alldata$date)
#DELETING DAYNIGHT FOR NOW
alldata<- alldata[-2]
alldata<- alldata[-2]
alldata<- alldata[-3]

write_csv(alldata, "alldata1.csv")
#-------------------------------------------------------------------------------------------------

maplot <- alldata2 %>%
  select(date, srate = usage) %>%
  mutate(srate_ma01 = rollmean(srate, k = 5, fill = NA, align = 'right'),
         srate_ma02 = rollmean(srate, k = 25, fill = NA, align = 'right'),
         srate_ma03 = rollmean(srate, k = 100, fill = NA, align = 'right'),
         srate_ma05 = rollmean(srate, k = 500, fill = NA, align = 'right'),
         srate_ma10 = rollmean(srate, k = 2000, fill = NA, align = 'right'))

maplot %>%
  gather(metric, value, srate:srate_ma10) %>%
  ggplot(aes(date, value, color = metric)) +
  geom_line()

savings_tma <- alldata2 %>%
  select(date, srate = usage) %>%
  mutate(srate_tma = rollmean(srate, k = 20, fill = NA, align = "right"))

savings_tma %>%
  gather(metric, value, -date) %>%
  ggplot(aes(date, value, color = metric)) +
  geom_line()
 #https://uc-r.github.io/ts_moving_averages

#-------------------------------------------------------------------------------------------------

#or 
x<- matrix(alldata$usage)
x2 <- ymd_hms(alldata$date)
tsall<- xts(x, order.by = x2)
tsall1<-ts(tsall, deltat = 1/24)
#or
#alldata2<- alldata[878:14816,]
ts2<- ts(alldata2$usage, start = 1, frequency = 24)
print(ts2)
plot(tsall1)
  


ts2 %>% tail(24*7*4) %>% 
  decompose() %>% 
  autoplot()

msts_cons<-ts2 %>% msts(seasonal.periods = c(24, 24*7))
msts_cons  %>% head(  24 *7 *4 ) %>% mstl() %>% autoplot()    
#https://rpubs.com/AlgoritmaAcademy/multiseasonality

moving<- rollmean(tsall1, k = 24)
head(moving)
df = data.frame(moving)
autoplot(moving)

#hampts<-hampel(tsall1, 2, 7)
#hampts$ind #HAMPEL OUTLIER , 5 3 IMPLACE: TRUE #omad <- hampel(tsding1, k=20, t0 = 5)

ggAcf(ts2)

acf(ts2)
cycle(ts2)
summary(ts2)

tsall1%>%
  mstl()%>%
  autoplot()
plot(decompose(ts2))
ts.plot(diff(ts2))


write_csv(alldata, "alldata.csv")
