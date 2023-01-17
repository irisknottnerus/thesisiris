



energ<- read.csv('alldata1.csv')
weather<- read.csv('weerbewerkt.csv')

nieuw<- cbind(energ, weather)
nieuw<- nieuw[-3]
nieuw<- nieuw[-25]
nieuw<- nieuw[-25]
nieuw<- nieuw[-25]
nieuw<- nieuw[-8]
nieuw<- nieuw[-8]



write_csv(nieuw, "elecweer.csv")

citation("xts")
citation()

