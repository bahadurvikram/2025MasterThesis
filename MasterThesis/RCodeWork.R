# Install necessary packages if you haven't already
if(!require(quantmod)){install.packages("quantmod")}
if(!require(data.table)){install.packages("data.table")}
if(!require(dplyr)){install.packages("dplyr")} # For data manipulation
if(!require(lubridate)){install.packages("lubridate")} # For date handling

library(quantmod)
library(data.table)
library(dplyr)
library(lubridate)
rm(list = ls())

download_stock_data <- function(symbol, start_date, end_date) {
  tryCatch({
    data <- getSymbols(symbol, from = start_date, to = end_date, auto.assign = FALSE)
    if (is.null(data) || nrow(data) == 0) {
      print(paste("No data found for symbol", symbol, "within the specified date range."))
      return(NULL)
    }
    return(data)
  }, error = function(e) {
    print(paste("Error downloading data for", symbol, ":", e$message))
    return(NULL)
  })
}


## Get present location
LOC_CODE = dirname(rstudioapi::getSourceEditorContext()$path)

print(LOC_CODE)
## Set it as current working directory
setwd(LOC_CODE)


symbol1 <- "ASIANPAINT.NS" #"HDFCNIFETF.NS" # Example: SBI Nifty 50 ETF
symbol2 <- "BHARTIARTL.NS" #"NIFTY1.NS" # Example: Nippon India ETF Nifty 50 BeES
formation_start <- "2022-01-01"
formation_end <- "2022-12-31"
trading_start <- "2023-01-01"
trading_end <- "2023-06-30"

#pair_trading <- function(symbol1, symbol2, formation_start, formation_end, trading_start, trading_end) {
  
  # 1. Download Data
  stock1_data <- download_stock_data(symbol1, formation_start, trading_end)
  stock2_data <- download_stock_data(symbol2, formation_start, trading_end)
  
  if (is.null(stock1_data) || is.null(stock2_data)) {
    return(NULL) # Exit if data download fails
  }
  
  # Convert to data frames for easier manipulation
  stock1_df <- as.data.frame(stock1_data) %>% mutate(Date = ymd(rownames(.)))
  stock2_df <- as.data.frame(stock2_data) %>% mutate(Date = ymd(rownames(.)))
  
  # 2. Calculate Cumulative Returns (Formation Period)
  stock1_df_formation <- stock1_df %>% filter(Date >= ymd(formation_start) & Date <= ymd(formation_end))
  stock1_df_formation$Return <- (Cl(stock1_data[stock1_df_formation$Date]) - lag(Cl(stock1_data[stock1_df_formation$Date]))) / lag(Cl(stock1_data[stock1_df_formation$Date]))
  stock1_df_formation$Return[is.na(stock1_df_formation$Return)] <- 0  # Handle first day
  stock1_df_formation$Cumulative_Return <- cumprod(1 + stock1_df_formation$Return)
  
  
  stock2_df_formation <- stock2_df %>% filter(Date >= ymd(formation_start) & Date <= ymd(formation_end))
  stock2_df_formation$Return <- (Cl(stock2_data[stock2_df_formation$Date]) - lag(Cl(stock2_data[stock2_df_formation$Date]))) / lag(Cl(stock2_data[stock2_df_formation$Date]))
  stock2_df_formation$Return[is.na(stock2_df_formation$Return)] <- 0  # Handle first day
  stock2_df_formation$Cumulative_Return <- cumprod(1 + stock2_df_formation$Return)
  
  
  # 3. Normalize Prices (Formation Period)
  #stock1_df_formation <- stock1_df_formation %>% mutate(Normalized_Price = Cumulative_Return / Cumulative_Return[1])
  #stock2_df_formation <- stock2_df_formation %>% mutate(Normalized_Price = Cumulative_Return / Cumulative_Return[1])
  
  
  stock1_df_formation$Normalized_Price <- stock1_df_formation$Cumulative_Return
  stock2_df_formation$Normalized_Price <- stock2_df_formation$Cumulative_Return
  
  
  par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))  # This sets the margins
  
  plot(stock1_df_formation$Normalized_Price, col="black")
  #plot(stock2_df_formation$Normalized_Price)
  lines(stock2_df_formation$Normalized_Price,col="blue")
  
  legend("topright", legend=c("HDFCNIFETF.NS-HDFC", "NIFTY1.NS-KOTAK"), 
         col=c("black", "blue"), lty=1, lwd=2)
  
  
  
  par(mfrow = c(1, 1))  # This resets the margins
  
  # there should be one more step as sum of square of difference SSD of normalized prices
  # it should be used to find the candidate pairs with minimum SSD
  # 4. Calculate Spread and Historical Standard Deviation (Formation Period)
  spread_formation <- stock1_df_formation$Normalized_Price - stock2_df_formation$Normalized_Price
  historical_sd <- sd(spread_formation, na.rm=TRUE)
  
  # 5. Trading Period
  stock1_df_trading <- stock1_df %>% filter(Date >= ymd(trading_start) & Date <= ymd(trading_end))
  stock1_df_trading$Return <- (Cl(stock1_data[stock1_df_trading$Date]) - lag(Cl(stock1_data[stock1_df_trading$Date]))) / lag(Cl(stock1_data[stock1_df_trading$Date]))
  stock1_df_trading$Return[is.na(stock1_df_trading$Return)] <- 0  # Handle first day
  stock1_df_trading$Cumulative_Return <- cumprod(1 + stock1_df_trading$Return)
  
  
  stock2_df_trading <- stock2_df %>% filter(Date >= ymd(trading_start) & Date <= ymd(trading_end))
  stock2_df_trading$Return <- (Cl(stock2_data[stock2_df_trading$Date]) - lag(Cl(stock2_data[stock2_df_trading$Date]))) / lag(Cl(stock2_data[stock2_df_trading$Date]))
  stock2_df_trading$Return[is.na(stock2_df_trading$Return)] <- 0  # Handle first day
  stock2_df_trading$Cumulative_Return <- cumprod(1 + stock2_df_trading$Return)
  
  stock1_df_trading$Normalized_Price <- stock1_df_trading$Cumulative_Return
  stock2_df_trading$Normalized_Price <- stock2_df_trading$Cumulative_Return
  #stock1_df_trading <- stock1_df_trading %>% mutate(Normalized_Price = Cumulative_Return / Cumulative_Return[1])
  #stock2_df_trading <- stock2_df_trading %>% mutate(Normalized_Price = Cumulative_Return / Cumulative_Return[1])
  
  trading_decisions <- data.frame(Date = stock1_df_trading$Date, Action = character(nrow(stock1_df_trading)), Return=numeric(nrow(stock1_df_trading)), stringsAsFactors = FALSE)
  
  #daily_strategy_return <- numeric(nrow(stock1_df_trading)) # Initialize daily return vector
  
  position_open <- FALSE # Flag to track if a position is open
  open_price_stock1 <- NA
  open_price_stock2 <- NA
  open_date <- NA
  position_type <- "" # Store the type of open position
  for (i in 2:nrow(stock1_df_trading)) {
    spread <- stock1_df_trading$Normalized_Price[i] - stock2_df_trading$Normalized_Price[i]
    print(paste("spread = ", spread, " position_open=", position_open, " Date=", stock1_df_trading$Date[i],
                " position_type=",position_type))
    # Open Trade
    if (!position_open && abs(spread) > 2 * historical_sd) {
      if (spread > 0) {
        trading_decisions$Action[i] <- "Short Stock 1, Long Stock 2"
        position_type <- "Short Stock 1, Long Stock 2" # Store position type
      } else {
        trading_decisions$Action[i] <- "Long Stock 1, Short Stock 2"
        position_type <- "Long Stock 1, Short Stock 2" # Store position type
      }
      open_price_stock1 <- as.numeric(Cl(stock1_data[stock1_df_trading$Date[i]]))
      open_price_stock2 <- as.numeric(Cl(stock2_data[stock2_df_trading$Date[i]]))
      position_open <- TRUE
      open_date <- stock1_df_trading$Date[i]
    }
    
    
    # Close Trade (Spread Crossing or End of Trading Period)
    if (position_open && (abs(spread) < historical_sd / 2 || i == nrow(stock1_df_trading))) { # Close if spread gets close to zero or last day
      trading_decisions$Action[i] <- "Close Position"
      close_price_stock1 <- as.numeric(Cl(stock1_data[stock1_df_trading$Date[i]]))
      close_price_stock2 <- as.numeric(Cl(stock2_data[stock2_df_trading$Date[i]]))
      
      # Calculate daily strategy return
      return_val <- NA
      if (position_type == "Short Stock 1, Long Stock 2") { # Use stored position type
        return_val <- ((open_price_stock1 - close_price_stock1)/open_price_stock1) + ((close_price_stock2 - open_price_stock2)/open_price_stock2)
      } else if (position_type == "Long Stock 1, Short Stock 2") { # Use stored position type
        return_val <- ((close_price_stock1 - open_price_stock1)/open_price_stock1) + ((open_price_stock2 - close_price_stock2)/open_price_stock2)
      }
      print(paste("close_price_stock1=", close_price_stock1," open_price_stock1=", open_price_stock1))
      print(paste("close_price_stock2=", close_price_stock2," open_price_stock2=", open_price_stock2))
      print((open_price_stock1 - close_price_stock1)/open_price_stock1)
      trading_decisions$Return[i] <- return_val
      #daily_strategy_return <- append(daily_strategy_return, return_val)
      
      position_open <- FALSE # Reset position flag
      open_price_stock1 <- NA
      open_price_stock2 <- NA
      open_date <- NA
      position_type <- "" # Reset position type  <--- This is crucial!
    } else if (position_open){ # If trade is open but not closed
      #daily_strategy_return <- append(daily_strategy_return, 0) # Append 0 to the vector
      trading_decisions$Return[i] <- 0.00
    }
  }
  
  lines(trading_decisions$Action,col="blue")
  
  
#  return(list(stock1 = stock1_df, stock2 = stock2_df, trading_decisions = trading_decisions, historical_sd = historical_sd))
#}


# Example usage (replace with your desired symbols and dates):


#results <- pair_trading(symbol1, symbol2, formation_start, formation_end, trading_start, trading_end)

results

if (!is.null(results)) {
  print(results$trading_decisions)
  print(paste("Historical Standard Deviation:", results$historical_sd))
  # You can further analyze the results (e.g., calculate returns, etc.)
}

plot.new()
plot.window(xlim=c(0,1), ylim=c(1,10))
abline(a=5, b=.45)
abline(a=0, b=.45)
points(.4,5)
points(.4,6)
points(.4,6, pch=8)


axis(1)
axis(2)

title(main="The Overall Title")
title(xlab="An x-axis label")
title(ylab="A y-axis label")

box()
