library(lubridate)
library(ggplot2)
interval <- interval(ymd("2017-09-27"), ymd("2017-11-19"))
as.period(interval, unit = "days")
as.numeric(interval, unit = "days")
data_frame <- read.csv("Weight.csv",header = FALSE)
days <- seq(by = 1, from = 1, to = length(data_frame$V1))
data_frame$index <- days
names(data_frame) <- c("Date", "Weight", "Index")
regression <- lm(Weight ~ Index)

attach(data_frame)

ggplot(data_frame, aes(x = Index, y = Weight)) + geom_point(color = "#3333CC") + ylim(c(175,200)) +
  labs(title = "Weight in Pounds vs. Day of My Diet", x = "Day of Diet", y = "Weight in Pounds") + 
  geom_text(aes(label=Weight), hjust = 0, vjust = -0.5, size = 3) + geom_smooth(method = "lm", color = "blue", se = FALSE, size = 0.25)
a

