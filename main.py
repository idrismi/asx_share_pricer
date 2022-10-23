import numpy as np
import pandas as pd
from quickfs import QuickFS
import yfinance as yf
from datetime import datetime as dt
from regressions import ExponentialRegression
import time


API_KEY = "2fbcd1d2ee927e374fc34c8f164673d7f5d8cf51"
INVESTMENT_RETURN = 0.15


class InvestmentSummary:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = QuickFS(api_key)
        self.tickers = self.client.get_supported_companies(country="AU", exchange="ASX")

    def run(self):
        index = 0
        temp_companies = ['fmg']
        temp_companies = [ticker + ':AU' for ticker in temp_companies]

        for ticker in temp_companies:# self.tickers[30:31]:  # TODO Delete [n1:n2]
            # ticker = 'CBA:AU' # TODO:DELETE
            
            try:
                company = Company(self.client, ticker)
            except:
                print(f"{ticker} data not available.")
                continue

            ls = [company.ticker, company.financials]
            for e in ls:
                print(e)

            if company.summary is not np.nan:
                for k, v in company.summary.items():
                    print(k + ": " + str(v))
            print("\n" * 2)

        """TODO: UNCOMMENT at the end
            # Save financials_df data to csv.
            company.financials.to_csv(self.ticker.split(':')[0] + '.csv')

            # Add summary_df data to summary_df and save to csv.
            # TODO: if summary file exists header = True else  False
            pd.DataFrame(company.summary, index=[index]).to_csv('summary_data.csv', header=True, mode='a') # TODO: Change header=header
            index += 1

            # Wait till next day if not enough quota. 
            remaining_quota = self.client.get_usage()['quota']['remaining']
            if remaining_quota < 10:
                sleep_till_reset(self.client)
        """


class Company:
    """Provides summary of a company's financial info"""
  
    features = ["roic", "book_value", "eps_basic", "revenue"]  # TODO:  Add cashflow

    def __init__(self, client, ticker):
        data = client.get_data_full(symbol=ticker)
        # TODO: Check for financials is None
        financials = pd.DataFrame(data["financials"]["annual"])
        self.ticker = ticker
        # self.metadata = data['metadata']
        # Check if there's enough data for processing.
        if len(financials) < 2:
            self.financials = np.nan
            self.summary = np.nan
        else:
            self.financials = self.update_financials(financials)
            self.summary = self.produce_summary(self.financials)

    def add_pe_ratio(self, df):
        '''Calculates pe ratio where each year's price is taken from the earliest date of the month "month"'''
        start_year = df["year"].min()
        end_year = df["year"].max()
        yf_ticker = self.ticker.split(":")[0] + ".AX"  # E.g. 'cba.ax'
        price_df = yearly_price(yf_ticker, start_year, end_year)
        df = pd.merge(df, price_df, on="year", how="left")
        df["pe_ratio"] = round(df["price"] / df["eps_basic"], 2)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def produce_summary(self, df):
        """Returns a dictionary of financial summary"""
        def roic_stable(ds):
            pc = ds.pct_change()
            pc = pc.replace([np.inf, -np.inf], np.nan)
            print(pc.min())
            return not pc.min() < -0.25

        start_year = self.financials["year"].min()
        end_year = self.financials["year"].max()
        years = end_year - start_year + 1
        summary = {"ticker": self.ticker, "years": years}
        feats = self.features.copy()
        feats.remove("roic")

        # Growth rate and growth stable
        if len(df) < 5:
            summary["roic_mean"] = np.nan  # calculate mean for ROIC, not growthrate
            summary["roic_stable"] = False
            for feat in feats:
                summary[f"growth_{feat}"] = np.nan
                summary[f"stable_{feat}"] = False
        else:
            summary["roic_mean"] = round(
                df["roic"].mean(), 4
            )  # calculate mean for ROIC, not growthrate
            summary["roic_stable"] = roic_stable(df["roic"])
            for feat in feats:
                er = ExponentialRegression(df["year"], df[feat])
                summary[f"growth_{feat}"] = round(er.growth_rate(), 4)
                summary[f"stable_{feat}"] = er.r_squared > 0.75
                summary[f"r_squared_{feat}"] = round(er.r_squared, 4)  # TODO: DELETE

        # Add PE ratio
        summary["pe_ratio"] = round(df["pe_ratio"].mean(), 4)

        # Add sticker and margin of safety (mos) price
        eps = df["eps_basic"].iloc[-1]
        summary["sticker_price"] = round(
            get_sticker_price(eps, summary["growth_book_value"], summary["pe_ratio"]), 4
        )
        summary["mos_price"] = round(summary["sticker_price"] / 2, 4)

        # TODO: Add to summary debt_is_reasonable
        # longterm_debt = financials_df['lt_debt'].iloc[-1]
        # freecashflow =
        # summary_dict['debt_is_reasonable'] = debt_is_reasonable(longterm_debt, freecashflow)
        return summary

    def update_financials(self, df):
        """
        Clean financials dataframe
        """

        df["year"] = df["period_end_date"].str[:4]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"] > df["year"].max() - 10]  # Data for most recent 10 years.
        df = self.add_pe_ratio(df)
        return df[["year", "price", "pe_ratio"] + self.features]


def debt_is_reasonable(long_term_debt, free_cashflow):
    return long_term_debt / free_cashflow < 3.5


def get_growth_rate(pv, fv, t, rounding=4, ignore_negative_pv=False):
    if ignore_negative_pv and pv <= 0:
        return np.nan
    return round((fv / pv) ** (1 / t) - 1, rounding)


def get_sticker_price(current_eps, growth_rate, future_pe):
    fv_eps = future_value(current_eps, growth_rate)
    fv_price = fv_eps * future_pe
    return round(present_value(fv_price, INVESTMENT_RETURN), 4)


def future_value(pv, r, t=10):
    return pv * (1 + r) ** t


def present_value(fv, r, t=10):
    return fv / (1 + r) ** t


def sleep_till_reset(client):
    reset_time = client.get_usage()["quota"]["resets"]
    reset_time = dt.strptime(reset_time, "Y%-%m-%dT%H:%M:%SZ")
    print(f"Quota for day reached. Script will continue at {reset_time}.")
    time_till_reset = reset_time - dt.now()
    time.sleep(time_till_reset.seconds + 1)


def yearly_price(ticker, start_year, end_year, month=7):
    df = yf.Ticker(ticker).history(period="max")
    if len(df) == 0:  # Make sure there's data.
        return pd.DataFrame(
            {
                "year": [year for year in range(start_year, end_year + 1)],
                "price": np.nan,
            }
        )
    # Clean df
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df = df.rename(columns={"Close": "price"})
    df = df[
        (df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)
    ]  # Years filter
    df = df[df["Date"].dt.month == month]  # Month filter

    # Initiailise blank dataframe price_df
    price_df = pd.DataFrame()

    # Append row in df, for earliest day of month of each year, to price_df
    for yr in range(start_year, end_year + 1):
        temp_df = df[df["Date"].dt.year == yr]
        temp_df = temp_df[
            temp_df["Date"].dt.day == temp_df["Date"].dt.day.min()
        ]  # Earliest date of month.
        price_df = price_df.append(temp_df, ignore_index=True)

    # Clean price_Df
    price_df["year"] = price_df["Date"].dt.year
    price_df.drop("Date", axis=1, inplace=True)

    return price_df


if __name__ == "__main__":
    main = InvestmentSummary(API_KEY)
    main.run()
