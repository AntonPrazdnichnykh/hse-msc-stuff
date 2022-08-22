import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Database:
    def __init__(self, db_path):
        db = sqlite3.connect(db_path)
        chart_data = pd.read_sql_query("SELECT * from chart_data", db)
        chart_data.drop(columns='id', inplace=True)
        trading_session = pd.read_sql_query("SELECT * from trading_session", db)
        trading_session['session_id'] = trading_session['id']
        trading_session.drop(columns='id', inplace=True)

        merged = pd.merge(chart_data, trading_session, on='session_id', validate='many_to_one')
        merged['datetime'] = pd.to_datetime(merged[['date', 'time']].apply(lambda x: x[0] + '/' + x[1], axis=1),
                                            format="%Y-%m-%d/%H:%M:%S")

        self.df = merged.drop(columns=['date', 'time'])

    def get_timeseries(self, trading_type, platform_id, normalize=None):

        assert platform_id in [1, 2], "Wrong platform id"

        if trading_type == 'daily':
            length = 30
        elif trading_type == 'monthly':
            length = 60
        else:
            raise NotImplementedError()

        out_df = self.df[(self.df.trading_type == trading_type) & (self.df.platform_id == platform_id)].sort_values(
            by='datetime', ignore_index=True
        )

        columns_ts = ['session_id'] + [f"min_{i}" for i in range(length)]
        price_ts = pd.DataFrame(columns=columns_ts)
        volume_ts = pd.DataFrame(columns=columns_ts)
        prev_session_last_price = 0
        for k, v in out_df.groupby('session_id').groups.items():
            prices = [None] * length
            volumes = [0] * length
            cur_session = out_df.iloc[v, :]
            h_mins = cur_session.datetime.apply(lambda x: (x.hour, x.minute))
            prev_hour = cur_session.datetime.iloc[0].hour
            for hm, indices in h_mins.groupby(h_mins).groups.items():
                hour, minute = hm
                prices_cur = out_df.price[indices]
                volumes_cur = out_df.lot_size[indices]
                volume_sum = volumes_cur.sum()
                mean_price = (prices_cur * volumes_cur / volume_sum).sum()
                prices[min(length - 1, minute)] = mean_price
                volumes[min(length - 1, minute)] = volume_sum
                if hour > prev_hour:
                    if prices[-1] is None:
                        prices[-1] = 0
                    if volumes[-1] is None:
                        volumes[-1] = 0
                    prices[-1] = (prices[-1] * volumes[-1] + mean_price * volume_sum) / (volumes[-1] + volume_sum)
                    volumes[-1] += volume_sum


            cur_price = prev_session_last_price
            for idx, ts in enumerate(prices):
                if ts is None:
                    prices[idx] = cur_price
                else:
                    cur_price = ts
            prev_session_last_price = cur_price
            price_values = [k] + prices
            volume_values = [k] + volumes
            price_ts = price_ts.append(dict(zip(columns_ts, price_values)), ignore_index=True)
            volume_ts = volume_ts.append(dict(zip(columns_ts, volume_values)), ignore_index=True)

        if normalize is not None:
            price_ts.iloc[:, 1:] = norm_timeseries(volume_ts.iloc[:, 1:], price_ts.iloc[:, 1:], how=normalize)

        price_ts.session_id = price_ts.session_id.astype(np.int32)

        return price_ts, volume_ts


def norm_timeseries(size_df, price_df, how="uniform", eps=1e-8):
    if how == "size-wise":
        distr = size_df.div(size_df.sum(axis=1), axis=0)
        first_moment = (price_df * distr).sum(axis=1)
        second_moment = (price_df ** 2 * distr).sum(axis=1)

        var = second_moment - first_moment ** 2
        var[var.abs() < eps] = 1  # means that time series is constant, set it equal to 0

        std = np.sqrt(var)
    elif how == "uniform":
        first_moment = price_df.mean(axis=1)
        std = price_df.std(axis=1)
    else:
        raise NotImplementedError()

    return price_df.sub(first_moment, axis=0).div(std, axis=0)


def plot_timeseries(time_series, labels, size=(20, 3)):
    # plt.figure(figsize=size)
    labels_unique = np.unique(labels)
    fig, axs = plt.subplots(1, len(labels_unique), figsize=size)
    for idx, cls in enumerate(labels_unique):
        cur_ts = time_series[labels == cls]
        mean_ts = cur_ts.mean(0)
        colors = cm.rainbow(np.linspace(0, 1., cur_ts.shape[0]))
        for ts, color in zip(cur_ts, colors):
            axs[idx].plot(ts, color=color)
        axs[idx].plot(mean_ts, 'k+')
        axs[idx].set_title(f"Cluster {cls}")
        axs[idx].set_xlabel("minute")
        axs[idx].set_ylabel("normed price")

    plt.show()


if __name__ == "__main__":
    db = Database("trade_info.sqlite3")
    prices, volumes = db.get_timeseries(trading_type='daily', platform_id=1, normalize=True)
    print(prices.head())
    print(volumes.head())
