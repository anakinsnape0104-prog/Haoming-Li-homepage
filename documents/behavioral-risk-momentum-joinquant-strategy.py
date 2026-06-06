# JoinQuant HS300 LightGBM multi-factor strategy, TOP18 and 2% stop-loss variant
# Copy this file into the JoinQuant strategy editor.

try:
    from jqdata import *
except ImportError:
    # Allows local syntax checks outside JoinQuant.
    pass

try:
    from jqfactor import get_factor_values as jq_get_factor_values
except Exception:
    jq_get_factor_values = None

import datetime
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    lgb = None

MODEL_CACHE = None


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_order_cost(
        OrderCost(
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            min_commission=5
        ),
        type='stock'
    )

    g.index_code = '000300.XSHG'
    g.factor_list = [
        'PSY', 'VR', 'DAVOL5', 'VROC12', 'ARBR',
        'Variance20', 'sharpe_ratio_60', 'Skewness60',
        'ROC20', 'PLRC24', 'fifty_two_week_close_rank'
    ]

    g.label_horizon = 20
    g.train_days = 120
    g.factor_batch_days = 30
    g.retrain_gap = 5
    g.top_n = 18
    g.stop_loss = 0.02
    g.min_list_days = 120
    g.use_excess_return_label = True

    g.model_ready = False
    g.days_since_train = g.retrain_gap
    g.last_train_date = None
    g.last_scores = None
    g.last_score_date = None
    g.last_use_ml = False

    run_daily(trade, time='9:30', reference_security=g.index_code)
    run_daily(after_market_close, time='15:30', reference_security=g.index_code)

    log.info('HS300 LightGBM TOP18 stop2 strategy initialized. lightgbm available: %s' % (lgb is not None))
    log.info('jqfactor.get_factor_values available: %s' % (jq_get_factor_values is not None))


def trade(context):
    signal_date = get_signal_date(context)
    current_data = get_current_data()

    universe = get_hs300_universe(signal_date, current_data)
    log.info('Trade date=%s signal_date=%s filtered_universe=%d' % (
        context.current_dt.date(), signal_date, len(universe)
    ))
    if len(universe) < g.top_n:
        log.info('Not enough tradable HS300 stocks: %d' % len(universe))
        return

    sell_stop_loss_positions(context, current_data)

    scores, use_ml = get_scores(context, universe, signal_date)
    if scores is None or len(scores) == 0:
        log.info('No valid score generated on %s' % signal_date)
        return
    log.info('Score generated. scored_stocks=%d use_ml=%s' % (len(scores), use_ml))

    risk_mult = compute_index_risk_multiplier(signal_date)
    g.last_use_ml = use_ml
    rebalance_portfolio(context, scores, risk_mult, current_data)

    g.last_scores = scores
    g.last_score_date = signal_date


def get_signal_date(context):
    if hasattr(context, 'previous_date') and context.previous_date is not None:
        return context.previous_date

    trade_days = get_trade_days(end_date=context.current_dt.date(), count=2)
    if len(trade_days) >= 2:
        return trade_days[-2]
    return context.current_dt.date()


def get_hs300_universe(signal_date, current_data):
    stocks = get_index_stocks(g.index_code, date=signal_date)
    filtered = []

    for stock in stocks:
        try:
            cd = current_data[stock]
            info = get_security_info(stock)
        except Exception:
            continue

        if cd.paused:
            continue
        if cd.is_st:
            continue
        if 'ST' in cd.name or '*' in cd.name or '退' in cd.name:
            continue
        if info is None or info.start_date is None:
            continue
        if (signal_date - info.start_date).days < g.min_list_days:
            continue

        filtered.append(stock)

    log.info('HS300 universe raw=%d filtered=%d' % (len(stocks), len(filtered)))
    return filtered


def safe_current_price(stock, current_data):
    try:
        cd = current_data[stock]
        for field in ['last_price', 'day_open']:
            if hasattr(cd, field):
                price = getattr(cd, field)
                if price is not None and np.isfinite(price) and price > 0:
                    return float(price)
        return np.nan
    except Exception:
        return np.nan


def sell_stop_loss_positions(context, current_data):
    for stock, position in list(context.portfolio.positions.items()):
        if position.total_amount <= 0:
            continue
        if position.closeable_amount <= 0:
            continue
        try:
            cd = current_data[stock]
        except Exception:
            continue
        if cd.paused:
            continue

        price = safe_current_price(stock, current_data)
        cost = float(position.avg_cost)
        if not np.isfinite(price) or not np.isfinite(cost) or cost <= 0:
            continue

        if price <= cost * (1.0 - g.stop_loss):
            if is_low_limit(stock, current_data):
                log.info('Stop-loss triggered but low-limit blocks sell: %s' % stock)
                continue
            log.info('Stop-loss sell: %s price=%.3f cost=%.3f' % (stock, price, cost))
            order_target(stock, 0)


def get_scores(context, stocks, signal_date):
    global MODEL_CACHE
    use_ml = False

    if should_retrain_model():
        model = train_lightgbm_model(stocks, signal_date)
        if model is not None:
            MODEL_CACHE = model
            g.model_ready = True
            g.last_train_date = signal_date
            g.days_since_train = 0
            log.info('LightGBM model trained on %s' % signal_date)
        else:
            MODEL_CACHE = None
            g.model_ready = False
            g.days_since_train = g.retrain_gap
            log.info('LightGBM unavailable or training failed; fallback scoring will be used.')
    else:
        g.days_since_train += 1

    x_today = get_factor_matrix(stocks, signal_date)
    if x_today is None or x_today.empty:
        log.info('Today factor matrix unavailable; use price-only fallback scoring.')
        price_scores = price_only_scores(stocks, signal_date)
        if price_scores is None or len(price_scores) == 0:
            return None, False
        return price_scores.sort_values(ascending=False), False

    if MODEL_CACHE is not None:
        try:
            preds = MODEL_CACHE.predict(x_today.values)
            preds = pd.Series(preds, index=x_today.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if not np.allclose(preds.values, preds.values[0]):
                use_ml = True
                ml_scores = zscore_series(preds)
                fallback = fallback_scores(x_today)
                scores = 0.85 * ml_scores + 0.15 * fallback
                return scores.sort_values(ascending=False), use_ml
        except Exception as exc:
            log.info('LightGBM prediction failed: %s' % exc)

    scores = fallback_scores(x_today)
    return scores.sort_values(ascending=False), False


def should_retrain_model():
    if lgb is None:
        return False
    if not g.model_ready or MODEL_CACHE is None:
        return True
    return g.days_since_train >= g.retrain_gap


def train_lightgbm_model(stocks, signal_date):
    if lgb is None:
        return None

    panel = build_training_panel(stocks, signal_date)
    if panel is None:
        return None

    x_train, y_train = panel
    if len(y_train) < 600:
        log.info('Training samples too few: %d' % len(y_train))
        return None

    try:
        y_train = np.clip(y_train, -0.30, 0.30)
        dtrain = lgb.Dataset(x_train, label=y_train)
        params = dict(
            objective='regression',
            metric='l2',
            learning_rate=0.04,
            num_leaves=63,
            max_depth=6,
            min_data_in_leaf=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.90,
            lambda_l2=2.0,
            seed=42,
            verbose=-1
        )
        try:
            model = lgb.train(params, dtrain, num_boost_round=80, verbose_eval=False)
        except TypeError:
            model = lgb.train(params, dtrain, num_boost_round=80)
        return model
    except Exception as exc:
        log.info('LightGBM training failed: %s' % exc)
        return None


def build_training_panel(stocks, signal_date):
    need_days = g.train_days + g.label_horizon + 1
    trade_days = list(get_trade_days(end_date=signal_date, count=need_days))
    if len(trade_days) < need_days:
        return None

    feature_dates = trade_days[:-g.label_horizon]
    target_dates = trade_days[g.label_horizon:]
    feature_dates = feature_dates[-g.train_days:]
    target_dates = target_dates[-g.train_days:]

    close_panel = get_close_panel(stocks, trade_days[0], trade_days[-1])
    if close_panel is None or close_panel.empty:
        return None

    bench_close = get_index_close_series(g.index_code, trade_days[0], trade_days[-1])
    if g.use_excess_return_label and (bench_close is None or bench_close.empty):
        return None

    factor_hist = get_factor_history(stocks, feature_dates)
    if factor_hist is None:
        return None

    x_list = []
    y_list = []

    for feature_date, target_date in zip(feature_dates, target_dates):
        if feature_date not in close_panel.index or target_date not in close_panel.index:
            continue

        x_df = factor_matrix_from_history(factor_hist, stocks, feature_date)
        if x_df is None or x_df.empty:
            continue

        start_price = close_panel.loc[feature_date].reindex(x_df.index)
        end_price = close_panel.loc[target_date].reindex(x_df.index)
        y = end_price / start_price - 1.0

        if g.use_excess_return_label:
            if feature_date not in bench_close.index or target_date not in bench_close.index:
                continue
            bench_ret = bench_close.loc[target_date] / bench_close.loc[feature_date] - 1.0
            y = y - bench_ret

        valid = np.isfinite(x_df.sum(axis=1).values) & np.isfinite(y.values)
        if valid.sum() == 0:
            continue

        x_list.append(x_df.values[valid])
        y_list.append(y.values[valid])

    if len(x_list) == 0:
        return None

    x_train = np.vstack(x_list)
    y_train = np.concatenate(y_list)
    return x_train, y_train


def get_factor_history(stocks, feature_dates):
    if len(feature_dates) == 0:
        return None

    merged = None
    start = 0
    while start < len(feature_dates):
        chunk_dates = feature_dates[start:start + g.factor_batch_days]
        start += g.factor_batch_days

        try:
            chunk = call_get_factor_values(
                stocks,
                factors=g.factor_list,
                end_date=chunk_dates[-1],
                count=len(chunk_dates)
            )
        except Exception as exc:
            log.info('get_factor_values history chunk failed: %s' % exc)
            return None

        if merged is None:
            merged = chunk
        else:
            for factor in g.factor_list:
                if factor not in merged or factor not in chunk:
                    return None
                merged[factor] = pd.concat([merged[factor], chunk[factor]], axis=0)

    return merged


def call_get_factor_values(stocks, factors, end_date, count):
    if jq_get_factor_values is not None:
        return jq_get_factor_values(stocks, factors=factors, end_date=end_date, count=count)

    func = globals().get('get_factor_values')
    if func is None:
        raise RuntimeError('get_factor_values is unavailable. In JoinQuant, add: from jqfactor import get_factor_values')

    try:
        return func(stocks, factors=factors, end_date=end_date, count=count)
    except TypeError:
        return func(securities=stocks, factors=factors, end_date=end_date, count=count)


def get_factor_matrix(stocks, signal_date):
    try:
        factor_data = call_get_factor_values(
            stocks,
            factors=g.factor_list,
            end_date=signal_date,
            count=1
        )
    except Exception as exc:
        log.info('get_factor_values today failed: %s' % exc)
        return None

    return factor_matrix_from_history(factor_data, stocks, signal_date)


def factor_matrix_from_history(factor_data, stocks, target_date):
    data = {}
    for factor in g.factor_list:
        if factor not in factor_data:
            return None
        df = factor_data[factor]
        if df is None or len(df) == 0:
            return None

        row = pick_factor_row(df, target_date)
        if row is None:
            return None
        data[factor] = row.reindex(stocks)

    x = pd.DataFrame(data, index=stocks)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(fill_and_zscore, axis=0)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x


def pick_factor_row(df, target_date):
    tmp = df.copy()
    tmp.index = pd.to_datetime(tmp.index).date

    if target_date in tmp.index:
        return tmp.loc[target_date]

    earlier_dates = [d for d in tmp.index if d <= target_date]
    if len(earlier_dates) == 0:
        return None
    return tmp.loc[earlier_dates[-1]]


def fill_and_zscore(series):
    s = pd.to_numeric(series, errors='coerce')
    med = s.median()
    if not np.isfinite(med):
        med = 0.0
    s = s.fillna(med)

    mean = s.mean()
    std = s.std()
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return (s - mean) / std


def fallback_scores(x_df):
    weights = {
        'ROC20': 0.25,
        'PLRC24': 0.20,
        'fifty_two_week_close_rank': 0.10,
        'PSY': 0.10,
        'VR': 0.08,
        'DAVOL5': 0.08,
        'VROC12': 0.05,
        'ARBR': 0.04,
        'sharpe_ratio_60': 0.15,
        'Skewness60': 0.03,
        'Variance20': -0.08
    }

    score = pd.Series(0.0, index=x_df.index)
    for factor, weight in weights.items():
        if factor in x_df.columns:
            score = score + weight * x_df[factor]
    return zscore_series(score)


def price_only_scores(stocks, signal_date):
    close = get_close_panel_by_count(stocks, signal_date, 121)
    if close is None or close.empty or len(close) < 61:
        return None

    last = close.iloc[-1]
    mom20 = last / close.iloc[-21] - 1.0
    mom60 = last / close.iloc[-61] - 1.0

    if len(close) >= 121:
        mom120 = last / close.iloc[-121] - 1.0
    else:
        mom120 = pd.Series(0.0, index=close.columns)

    score = (
        0.50 * zscore_series(mom20) +
        0.35 * zscore_series(mom60) +
        0.15 * zscore_series(mom120)
    )
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return score.reindex(stocks).fillna(0.0)


def get_close_panel_by_count(stocks, end_date, count):
    try:
        df = get_price(
            stocks,
            end_date=end_date,
            count=count,
            frequency='daily',
            fields=['close'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
    except Exception as exc:
        log.info('get_price close by count failed: %s' % exc)
        return None

    if df is None or len(df) == 0:
        return None

    if 'time' not in df.columns or 'code' not in df.columns:
        return None

    df = df.copy()
    df['date'] = pd.to_datetime(df['time']).dt.date
    close = df.pivot(index='date', columns='code', values='close')
    close = close.reindex(columns=stocks)
    close = close.replace([np.inf, -np.inf], np.nan)
    close = close.fillna(method='ffill').fillna(method='bfill')
    return close


def zscore_series(series):
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = s.mean()
    std = s.std()
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return (s - mean) / std


def get_close_panel(stocks, start_date, end_date):
    try:
        df = get_price(
            stocks,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['close'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
    except Exception as exc:
        log.info('get_price close panel failed: %s' % exc)
        return None

    if df is None or len(df) == 0:
        return None

    if 'time' not in df.columns or 'code' not in df.columns:
        return None

    df = df.copy()
    df['date'] = pd.to_datetime(df['time']).dt.date
    close = df.pivot(index='date', columns='code', values='close')
    close = close.reindex(columns=stocks)
    close = close.replace([np.inf, -np.inf], np.nan)
    close = close.fillna(method='ffill').fillna(method='bfill')
    return close


def get_index_close_series(index_code, start_date, end_date):
    try:
        df = get_price(
            index_code,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['close'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
    except Exception as exc:
        log.info('get_price index failed: %s' % exc)
        return None

    if df is None or len(df) == 0:
        return None

    if 'time' in df.columns:
        idx = pd.to_datetime(df['time']).dt.date
        return pd.Series(df['close'].values, index=idx)

    out = df['close'].copy()
    out.index = pd.to_datetime(out.index).date
    return out


def compute_index_risk_multiplier(signal_date):
    try:
        df = get_price(
            g.index_code,
            end_date=signal_date,
            count=121,
            frequency='daily',
            fields=['close'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
    except Exception as exc:
        log.info('Risk multiplier price fetch failed: %s' % exc)
        return 1.0

    if df is None or len(df) < 121:
        return 1.0

    close = np.asarray(df['close'], dtype='float64')
    idx_now = close[-1]
    ma60 = np.nanmean(close[-60:])
    ma120 = np.nanmean(close[-120:])
    mom20 = idx_now / close[-21] - 1.0

    if idx_now >= ma60:
        return 1.0
    if idx_now >= ma120:
        return 0.85
    if mom20 < 0:
        return 0.45
    return 0.65


def rebalance_portfolio(context, scores, risk_mult, current_data):
    positions = context.portfolio.positions

    profitable_holds = []
    for stock, position in list(positions.items()):
        if position.total_amount <= 0:
            continue
        if stock not in scores.index:
            continue
        price = safe_current_price(stock, current_data)
        cost = float(position.avg_cost)
        if np.isfinite(price) and np.isfinite(cost) and cost > 0 and price >= cost:
            profitable_holds.append(stock)

    target = []
    for stock in profitable_holds:
        if stock not in target:
            target.append(stock)

    for stock in scores.index:
        if len(target) >= g.top_n:
            break
        if stock in target:
            continue
        if not can_buy(stock, current_data):
            continue
        target.append(stock)

    if len(target) == 0:
        log.info('No target stocks; clear sellable positions.')
        clear_positions_not_in_target(context, set(), current_data)
        return

    target_set = set(target)
    clear_positions_not_in_target(context, target_set, current_data)

    weight = max(min(risk_mult, 1.0), 0.0) / float(len(target))
    for stock in target:
        if stock in positions and positions[stock].total_amount > 0:
            if current_data[stock].paused:
                continue
            order_target_percent_compat(context, stock, weight)
        else:
            if can_buy(stock, current_data):
                order_target_percent_compat(context, stock, weight)

    log.info(
        'Rebalance done. targets=%d risk_mult=%.2f use_ml=%s'
        % (len(target), risk_mult, g.last_use_ml)
    )


def clear_positions_not_in_target(context, target_set, current_data):
    for stock, position in list(context.portfolio.positions.items()):
        if position.total_amount <= 0:
            continue
        if stock in target_set:
            continue
        if position.closeable_amount <= 0:
            continue
        if stock in current_data and current_data[stock].paused:
            continue
        if stock in current_data and is_low_limit(stock, current_data):
            continue
        order_target(stock, 0)


def order_target_percent_compat(context, stock, weight):
    target_value = context.portfolio.total_value * max(float(weight), 0.0)
    return order_target_value(stock, target_value)


def can_buy(stock, current_data):
    try:
        cd = current_data[stock]
    except Exception:
        return False
    if cd.paused:
        return False
    if cd.is_st:
        return False
    price = safe_current_price(stock, current_data)
    if np.isfinite(price) and price > 0 and price >= cd.high_limit * 0.995:
        return False
    return True


def is_low_limit(stock, current_data):
    try:
        cd = current_data[stock]
        price = safe_current_price(stock, current_data)
        if not np.isfinite(price):
            return False
        return price <= cd.low_limit * 1.005
    except Exception:
        return False


def after_market_close(context):
    log.info(
        'After close. score_date=%s model_date=%s use_ml=%s'
        % (g.last_score_date, g.last_train_date, g.last_use_ml)
    )
    trades = get_trades()
    for trade_id in trades:
        log.info('Trade: %s' % trades[trade_id])
