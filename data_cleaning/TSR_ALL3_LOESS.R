load(url('https://www.dropbox.com/s/ud32tbptyvjsnp4/data.R?dl=1'))
install.packages("Gmisc")
library(Gmisc)
library(ggplot2)
library(dplyr)

##### TSR_ALL3_score.csv
setwd("C:/Users/Jacky C/PycharmProjects/tsr_ml/data_cleaning")
file_path <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3_score.csv")
TSR_ALL3 <- read.csv(file_path)

### the function needs to be checked 
#amiright <- function(df, var1, var2){
  #df_1 <- dplyr::select(df, var1, var2)
  #df_1$number <- 1
  #df_1 <- df_1 %>%
  #  group_by(df_1[1], df_1[2]) %>% 
  #  summarise(number = sum(number))
  #lw <- loess(var2 ~ var1,data=df)
  #j <- order(df$var1)
  
  #plot <- ggplot()+
  #  geom_point(data = df_1, aes(x =var1, y = var2, size = number))+
  #  geom_line(aes(x =df$var1[j], y = lw$fitted[j]),col="red", lwd=2)+
  #  ylim(min(var2), max(var2))
#}

#amiright(TSR_ALL3, "discharged_mrs", "bi_total")

# discharged_mrs VERSUS bi_total
TSR_ALL3_selected_1 <- TSR_ALL3[c("discharged_mrs", "bi_total")]
TSR_ALL3_selected_1$number <- 1
TSR_ALL3_selected_1 <- TSR_ALL3_selected_1 %>%
  group_by(discharged_mrs, bi_total) %>% 
  summarise(number = sum(number))
lw1 <- loess(bi_total ~ discharged_mrs,data=TSR_ALL3)
j <- order(TSR_ALL3$discharged_mrs)

#plot(bi_total ~ discharged_mrs, data=TSR_ALL3_selected_1,pch=19,cex=number/400)
#lines(TSR_ALL3$discharged_mrs[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_1, aes(x =discharged_mrs, y = bi_total, size = number))+
  geom_line(aes(x =TSR_ALL3$discharged_mrs[j], y = lw1$fitted[j]),col="red", lwd=2)+
  ylim(0, 100)



# discharged_mrs VERSUS nihss_total
TSR_ALL3_selected_2 <- TSR_ALL3[c("discharged_mrs", "nihss_total")]
TSR_ALL3_selected_2$number <- 1
TSR_ALL3_selected_2 <- TSR_ALL3_selected_2 %>%
  group_by(discharged_mrs, nihss_total) %>% 
  summarise(number = sum(number))
lw2 <- loess(nihss_total ~ discharged_mrs,data=TSR_ALL3)
j <- order(TSR_ALL3$discharged_mrs)

#plot(nihss_total ~ discharged_mrs, data=TSR_ALL3_selected_2,pch=19,cex=number/400)
#lines(TSR_ALL3$discharged_mrs[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_2, aes(x =discharged_mrs, y = nihss_total, size = number))+
  geom_line(aes(x =TSR_ALL3$discharged_mrs[j], y = lw2$fitted[j]),col="red", lwd=2)+
  ylim(0, 42)



# bi_total VERSUS nihss_total
TSR_ALL3_selected_3 <- TSR_ALL3[c("bi_total", "nihss_total")]
TSR_ALL3_selected_3$number <- 1
TSR_ALL3_selected_3 <- TSR_ALL3_selected_3 %>%
  group_by(bi_total, nihss_total) %>% 
  summarise(number = sum(number))
lw3 <- loess(nihss_total ~ bi_total,data=TSR_ALL3)
j <- order(TSR_ALL3$bi_total)

#plot(nihss_total ~ bi_total, data=TSR_ALL3_selected_3,pch=19,cex=number/400)
#lines(TSR_ALL3$bi_total[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_3, aes(x =bi_total, y = nihss_total, size = number))+
  geom_line(aes(x =TSR_ALL3$bi_total[j], y = lw3$fitted[j]),col="red", lwd=2)+
  ylim(0, 42)+
  xlim(0, 100)


##### TSR_ALL3_score_cleaned.csv
file_path <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3_score_cleaned.csv")
TSR_ALL3 <- read.csv(file_path)

# discharged_mrs VERSUS bi_total
TSR_ALL3_selected_1 <- TSR_ALL3[c("discharged_mrs", "bi_total")]
TSR_ALL3_selected_1$number <- 1
TSR_ALL3_selected_1 <- TSR_ALL3_selected_1 %>%
  group_by(discharged_mrs, bi_total) %>% 
  summarise(number = sum(number))
lw1 <- loess(bi_total ~ discharged_mrs,data=TSR_ALL3)
j <- order(TSR_ALL3$discharged_mrs)

#plot(bi_total ~ discharged_mrs, data=TSR_ALL3_selected_1,pch=19,cex=number/400)
#lines(TSR_ALL3$discharged_mrs[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_1, aes(x =discharged_mrs, y = bi_total, size = number))+
  geom_line(aes(x =TSR_ALL3$discharged_mrs[j], y = lw1$fitted[j]),col="red", lwd=2)+
  ylim(0, 100)



# discharged_mrs VERSUS nihss_total
TSR_ALL3_selected_2 <- TSR_ALL3[c("discharged_mrs", "nihss_total")]
TSR_ALL3_selected_2$number <- 1
TSR_ALL3_selected_2 <- TSR_ALL3_selected_2 %>%
  group_by(discharged_mrs, nihss_total) %>% 
  summarise(number = sum(number))
lw2 <- loess(nihss_total ~ discharged_mrs,data=TSR_ALL3)
j <- order(TSR_ALL3$discharged_mrs)

#plot(nihss_total ~ discharged_mrs, data=TSR_ALL3_selected_2,pch=19,cex=number/400)
#lines(TSR_ALL3$discharged_mrs[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_2, aes(x =discharged_mrs, y = nihss_total, size = number))+
  geom_line(aes(x =TSR_ALL3$discharged_mrs[j], y = lw2$fitted[j]),col="red", lwd=2)+
  ylim(0, 42)



# bi_total VERSUS nihss_total
TSR_ALL3_selected_3 <- TSR_ALL3[c("bi_total", "nihss_total")]
TSR_ALL3_selected_3$number <- 1
TSR_ALL3_selected_3 <- TSR_ALL3_selected_3 %>%
  group_by(bi_total, nihss_total) %>% 
  summarise(number = sum(number))
lw3 <- loess(nihss_total ~ bi_total,data=TSR_ALL3)
j <- order(TSR_ALL3$bi_total)

#plot(nihss_total ~ bi_total, data=TSR_ALL3_selected_3,pch=19,cex=number/400)
#lines(TSR_ALL3$bi_total[j],lw1$fitted[j],col="red",lwd=3)

ggplot()+
  geom_point(data = TSR_ALL3_selected_3, aes(x =bi_total, y = nihss_total, size = number))+
  geom_line(aes(x =TSR_ALL3$bi_total[j], y = lw3$fitted[j]),col="red", lwd=2)+
  ylim(0, 42)+
  xlim(0, 100)
