# Install necessary packages if you haven't already
if(!require(quantmod)){install.packages("quantmod")}
if(!require(tidyverse)){install.packages("tidyverse")} # For data manipulation and plotting (optional)

library(quantmod)
library(tidyverse) # Optional

download_stock_data <- function(symbol, start_date, end_date) {
  #' Downloads historical stock data from Yahoo Finance using quantmod.
  #'
  #' Args:
  #'   symbol (str): The stock symbol (e.g., "AAPL", "MSFT", "NIFTYBEES.NS"). For Indian stocks, ensure you include the appropriate suffix (e.g., ".NS" for NSE).
  #'   start_date (str): The start date for the data (YYYY-MM-DD).
  #'   end_date (str): The end date for the data (YYYY-MM-DD).
  #'
  #' Returns:
  #'   xts/zoo object: An xts/zoo object containing the historical stock data, or NULL if an error occurs.
  #'   Prints an error message if the download fails.
  
  tryCatch({
    data <- getSymbols(symbol, from = start_date, to = end_date, auto.assign = FALSE) # auto.assign = FALSE is crucial
    if (is.null(data) || nrow(data) == 0) {  # Check for empty data
      print(paste("No data found for symbol", symbol, "within the specified date range."))
      return(NULL)
    }
    return(data)
  }, error = function(e) {
    print(paste("Error downloading data for", symbol, ":", e$message))
    return(NULL)
  })
}


# Example usage:
data <- getSymbols("NIFTYBEES.NS", from = "2020-01-01", to = "2024-12-31", auto.assign = FALSE) #
symbol <- "NIFTYBEES.NS"  # Example: Nippon India ETF Nifty 50 BeES
start_date <- "2023-01-01"
end_date <- "2023-12-31"

df <- download_stock_data(symbol, start_date, end_date)

if (!is.null(df)) {
  print(head(df)) # Print the first few rows
  # Now you can work with the time series object 'df'
  
  # Convert to a data frame for easier manipulation (optional)
  df_as_df <- as.data.frame(df)
  # Save to a CSV file:
  write.csv(df_as_df, paste0(symbol, "_data.csv"))
  print(paste("Data saved to", paste0(symbol, "_data.csv")))
  
  # Accessing specific columns:
  #print(df[, "Close"]) # Access the 'Close' price column
  #print(df["2023-03-15"]) # Access a specific date (if available)
  
  # Plotting (requires tidyverse)
  #df_as_df %>%  # Convert to dataframe first
  #  ggplot(aes(x = index(df), y = Close)) + # index() gets the dates
  #  geom_line() +
  #  labs(title = paste("Closing Price of", symbol), x = "Date", y = "Price")
  
}


# Example for a US stock:
symbol_us <- "AAPL"
df_us <- download_stock_data(symbol_us, start_date, end_date)

if (!is.null(df_us)) {
  print(head(df_us))
  df_us_as_df <- as.data.frame(df_us)
  write.csv(df_us_as_df, paste0(symbol_us, "_data.csv"))
  print(paste("Data saved to", paste0(symbol_us, "_data.csv")))
}


# Important points to consider:

# 1. NSE Symbol Format: Just like in Python, for stocks listed on the National Stock Exchange of India (NSE), it's crucial to include the ".NS" suffix (e.g., "NIFTYBEES.NS", "SBINIFTY.NS").

# 2. Error Handling: The `tryCatch` block is essential for R.

# 3. Empty Data Check: The code checks for `NULL` or an empty object.

# 4. Saving to CSV: The code saves the data to a CSV file.

# 5. Time Series Objects (xts/zoo): `quantmod` returns time series objects (xts or zoo). These are specifically designed for financial data and have useful functions for time-based analysis.  You can often work with them directly, but the example shows how to convert to a regular data frame using `as.data.frame()` if needed.

# 6. Accessing Data:  Accessing columns and rows in time series objects is slightly different than in data frames. The example shows how.

# 7. Plotting: The commented-out code shows a basic example of plotting using `ggplot2` (from the `tidyverse` package). If you want to plot, uncomment this section and make sure you have `tidyverse` installed. The x-axis is accessed using `index(df)` which gives the date from the time series object.

# 8. Installing Packages: Make sure to install the necessary packages: `install.packages(c("quantmod", "tidyverse"))`

# 9. `auto.assign = FALSE`: It is *very* important to set  `auto.assign = FALSE` in `getSymbols`.  The default behavior of `getSymbols` is to automatically assign the downloaded data to a variable with the symbol name in your global environment. This can be very messy and lead to unexpected behavior.  Setting it to `FALSE` prevents this and allows you to explicitly assign the data to a variable.