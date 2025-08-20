import polars as pl
import numpy as np

# from haversine import haversine
from scipy.stats import gaussian_kde
from polars.datatypes import Struct, List
from .utils import timer, TRAIN_VAL_SIZE, TRAIN_ALL_SIZE, DATA_DIR
from .feature_specs import get_all_feature_lists


def dur_to_min(col):
    """More efficient duration to minutes converter"""
    # Extract days and time parts in one pass
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = (
        pl.when(col.str.contains(r"^\d+\."))
        .then(col.str.replace(r"^\d+\.", ""))
        .otherwise(col)
    )
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)


def kde_mode(arr):
    arr = arr[~np.isnan(arr)]
    kde = gaussian_kde(arr)
    xs = np.linspace(np.min(arr), np.max(arr), 1000)
    ys = kde(xs)
    mode = xs[np.argmax(ys)]
    return mode


def get_rank_bin_expr(rank_col: str, bin_edges: list[int], bin_col: str):
    expr = pl.when(pl.col(rank_col) <= bin_edges[0]).then(1)
    for i in range(1, len(bin_edges)):
        expr = expr.when(pl.col(rank_col) <= bin_edges[i]).then(i + 1)
    expr = expr.otherwise(len(bin_edges) + 1).cast(pl.Int8).alias(bin_col)
    return expr


###########################################################
#                     Feature Engineering                 #
###########################################################


@timer
def initial_transformations(df, FULL):
    # Precompute marketing carrier columns check
    mc_cols = [
        f"legs{l}_segments{s}_marketingCarrier_code" for l in (0, 1) for s in range(4)
    ]
    mc_exists = [col for col in mc_cols if col in df.columns]

    piece_to_kg = 20
    baggage_cols = [
        f"legs{l}_segments{s}_baggageAllowance" for l in (0, 1) for s in range(2)
    ]

    df = df.with_columns(
        [
            # Price features
            (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
            (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
            pl.col("totalPrice").log1p().alias("log_price"),
            pl.col("taxes").log1p().alias("log_taxes"),
            (pl.col("taxes") == 0).cast(pl.Int8).alias("is_zero_tax"),
            # Duration features
            (
                pl.col("legs0_duration").fill_null(0)
                + pl.col("legs1_duration").fill_null(0)
            ).alias("total_duration"),
            pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / pl.col("legs1_duration"))
            .otherwise(0)
            .alias("duration_ratio"),
            (pl.col("legs0_duration").rank("dense").over("ranker_id") == 1)
            .cast(pl.Int32)
            .alias("is_shortest_duration"),
            # Trip type
            (
                pl.col("legs1_duration").is_null()
                | (pl.col("legs1_duration") == 0)
                | pl.col("legs1_segments0_departureFrom_airport_iata").is_null()
            )
            .cast(pl.Int32)
            .alias("is_one_way"),
            # Total segments count
            (
                pl.sum_horizontal(
                    pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists
                )
                if mc_exists
                else pl.lit(0)
            ).alias("l0_seg"),
            # FF features
            (
                pl.col("frequentFlyer").fill_null("").str.count_matches("/")
                + (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)
            ).alias("n_ff_programs"),
            (pl.col("frequentFlyer").fill_null("") != "")
            .cast(pl.Int32)
            .alias("has_ff_program"),
            pl.col("corporateTariffCode")
            .is_not_null()
            .cast(pl.Int32)
            .alias("has_corporate_tariff"),
            pl.when(
                pl.col("corporateTariffCode").is_not_null()
                & (pl.col("pricingInfo_isAccessTP") == 1.0)
            )
            .then(1)
            .otherwise(0)
            .alias("corporate_policy_compliant"),
            pl.when(
                pl.col("corporateTariffCode").is_not_null() & (pl.col("isVip") == True)
            )
            .then(1)
            .otherwise(0)
            .alias("corporate_vip_flag"),
            # Baggage & fees
            (
                (
                    pl.col("miniRules0_monetaryAmount")
                    / pl.col("totalPrice").cast(pl.Float64)
                )
            ).alias("fee_ratio_rule0"),
            (
                (
                    pl.col("miniRules1_monetaryAmount")
                    / pl.col("totalPrice").cast(pl.Float64)
                )
            ).alias("fee_ratio_rule1"),
            (
                (pl.col("miniRules0_monetaryAmount") == 0)
                & (pl.col("miniRules0_statusInfos") == 1)
            )
            .cast(pl.Int8)
            .alias("free_cancel"),
            (
                (pl.col("miniRules1_monetaryAmount") == 0)
                & (pl.col("miniRules1_statusInfos") == 1)
            )
            .cast(pl.Int8)
            .alias("free_exchange"),
            # Routes & carriers
            pl.col("searchRoute")
            .is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"])
            .cast(pl.Int32)
            .alias("is_popular_route"),
            pl.col("searchRoute")
            .str.contains("MOW|LED")
            .cast(pl.Int32)
            .alias("contains_capitials"),
            pl.col("legs0_segments0_flightNumber")
            .is_in(["208"])
            .cast(pl.Int32)
            .alias("is_popular_flight"),
            (
                pl.col("legs0_segments0_marketingCarrier_code")
                != pl.col("legs0_segments0_operatingCarrier_code")
            )
            .cast(pl.Int32)
            .alias("is_codeshare_leg0_seg0"),
            (
                pl.col("legs1_segments0_marketingCarrier_code")
                != pl.col("legs1_segments0_operatingCarrier_code")
            )
            .cast(pl.Int32)
            .alias("is_codeshare_leg1_seg0"),
            # # Seats
            # (pl.col("legs0_segments0_seatsAvailable") == 9).alias("legs0_seat_enough"),
            # (pl.col("legs1_segments0_seatsAvailable") == 9).alias("legs1_seat_enough"),
        ]
    )

    # Price / duration
    df = df.with_columns(
        (pl.col("totalPrice") / pl.col("total_duration")).alias("price_duration_rat")
    )

    # Fill missing values using hand-craft rules
    df = df.with_columns(
        [
            pl.col("fee_ratio_rule0")
            .is_null()
            .cast(pl.Int8)
            .alias("fee_ratio_rule0_is_missing"),
            pl.col("fee_ratio_rule1")
            .is_null()
            .cast(pl.Int8)
            .alias("fee_ratio_rule1_is_missing"),
            # pl.col("pricingInfo_isAccessTP").fill_null(-1).cast(pl.Int32),
            pl.col("legs0_segments0_seatsAvailable")
            .fill_null(strategy="mean")
            .alias("legs0_segments0_seatsAvailable"),
            pl.col("miniRules0_statusInfos").fill_null(-1).cast(pl.Int32),
            pl.col("miniRules1_statusInfos").fill_null(-1).cast(pl.Int32),
        ]
    )
    # df = df.drop("legs0_segments0_seatsAvailable")
    return df


@timer
def build_segment_features(df):
    # Segment counts - more efficient
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_flightNumber" for s in range(4)]
        if seg_cols:
            seg_exprs.append(
                pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                .cast(pl.Int32)
                .alias(f"n_segments_leg{leg}")
            )
        else:
            seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

    # First create segment counts
    df = df.with_columns(seg_exprs)

    # Then use them for derived features
    df = df.with_columns(
        [
            # (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias(
            #     "total_segments"
            # ),
            (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
            pl.when(pl.col("is_one_way") == 1)
            .then(0)
            .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32))
            .alias("is_direct_leg1"),
            (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id"))
            .cast(pl.Int32)
            .alias("is_min_segments"),
            (
                pl.col("n_segments_leg0")
                == pl.col("n_segments_leg0").min().over("ranker_id")
            )
            .cast(pl.Int32)
            .alias("is_min_segments_leg0"),
            pl.when(pl.col("is_one_way") == 1)
            .then(0)
            .otherwise(
                (
                    pl.col("n_segments_leg1")
                    == pl.col("n_segments_leg1").min().over("ranker_id")
                ).cast(pl.Int32)
            )
            .alias("is_min_segments_leg1"),
        ]
    )

    # Shortest direct
    direct_shortest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg(pl.col("total_duration").min().alias("min_direct"))
    )

    df = (
        df.join(direct_shortest, on="ranker_id", how="left")
        .with_columns(
            (
                (pl.col("is_direct_leg0") == 1)
                & (pl.col("total_duration") == pl.col("min_direct"))
            )
            .cast(pl.Int32)
            .fill_null(0)
            .alias("is_direct_shortest")
        )
        .drop("min_direct")
    )

    return df


@timer
def build_time_features(df: pl.DataFrame) -> pl.DataFrame:
    time_cols = [
        "legs0_departureAt",
        "legs0_arrivalAt",
        "legs1_departureAt",
        "legs1_arrivalAt",
    ]

    # Conver time format to datetime
    dt_cols = [
        pl.col(col).str.to_datetime(strict=False).alias(f"{col}_dt")
        for col in time_cols
    ]
    df = df.with_columns(dt_cols)

    # Original time features
    time_exprs = []
    for col in time_cols:
        dt_col = f"{col}_dt"
        h = pl.col(dt_col).dt.hour().fill_null(-1)
        m = pl.col(dt_col).dt.minute().fill_null(0)
        # minute_of_day = (h * 60 + m).alias(f"{col}_minute_of_day")

        # 早高峰标记 (7, 8, 9)
        peak_morning = ((h >= 7) & (h <= 9)).cast(pl.Int32).alias(f"{col}_peak_morning")
        # 晚高峰标记 (19, 20, 21)
        peak_evening = (
            ((h >= 19) & (h <= 21)).cast(pl.Int32).alias(f"{col}_peak_evening")
        )

        time_exprs.extend(
            [
                h.alias(f"{col}_hour"),
                # minute_of_day,
                pl.col(dt_col).dt.weekday().fill_null(0).alias(f"{col}_weekday"),
                ((h >= 6) & (h < 21)).cast(pl.Int32).alias(f"{col}_business_time"),
                peak_morning,
                peak_evening,
                pl.when(h == -1)
                .then(-1)
                .when(h < 6)
                .then(0)
                .when(h < 7)
                .then(1)
                .when(h < 12)
                .then(2)
                .when(h < 19)
                .then(3)
                .when(h < 22)
                .then(4)
                .otherwise(5)
                .alias(f"{col}_time_bin"),
            ]
        )
    df = df.with_columns(time_exprs)

    # Combo features for sorting model (bin, red-eye, business-friendly)
    combo_exprs = []
    for leg in ["legs0", "legs1"]:
        dep_bin = f"{leg}_departureAt_time_bin"
        arr_bin = f"{leg}_arrivalAt_time_bin"
        dep_hour = f"{leg}_departureAt_hour"
        arr_hour = f"{leg}_arrivalAt_hour"

        # bin combination feature
        combo_exprs.append(
            (pl.col(dep_bin).cast(pl.Utf8) + "_" + pl.col(arr_bin).cast(pl.Utf8)).alias(
                f"{leg}_dep_arr_bin_combo"
            )
        )

        # business-friendly: both dep & arr in 6–10 or 17–21
        combo_exprs.append(
            (
                ((pl.col(dep_hour) >= 6) & (pl.col(dep_hour) < 21))
                & ((pl.col(arr_hour) >= 6) & (pl.col(arr_hour) < 21))
            )
            .cast(pl.Int8)
            .alias(f"{leg}_is_business_friendly")
        )

        # red-eye flight: dep or arr hour < 6
        combo_exprs.append(
            ((pl.col(dep_hour) < 6) | (pl.col(arr_hour) < 6))
            .cast(pl.Int8)
            .alias(f"{leg}_is_red_eye")
        )

    df = df.with_columns(combo_exprs)

    # Stay duration
    df = df.with_columns(
        (
            (pl.col("legs1_departureAt_dt") - pl.col("legs0_arrivalAt_dt"))
            .dt.total_microseconds()
            .cast(pl.Float64)
            / 1e6
            / 3600.0
        ).alias("stay_duration_hours")
    )

    df = df.with_columns(
        pl.when(pl.col("stay_duration_hours") < 0)
        .then(0)
        .otherwise(pl.col("stay_duration_hours"))
        .alias("stay_duration_hours")
    )

    # Bins of stay duration
    stay_exprs = [
        pl.col("stay_duration_hours").log1p().alias("stay_duration_hours_log"),
        pl.when(
            (pl.col("stay_duration_hours").is_not_null())
            & (pl.col("stay_duration_hours") > 0)
            & (pl.col("stay_duration_hours") < 36)
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("is_short_trip"),
        pl.when(pl.col("stay_duration_hours") == 0)
        .then(-1)
        .when(pl.col("stay_duration_hours") < 16)
        .then(0)
        .when(pl.col("stay_duration_hours") < 38)
        .then(1)
        .when(pl.col("stay_duration_hours") < 64)
        .then(2)
        .when(pl.col("stay_duration_hours") < 88)
        .then(3)
        .when(pl.col("stay_duration_hours") < 112)
        .then(4)
        .otherwise(5)
        .alias("stay_duration_bin"),
    ]
    time_exprs.extend(stay_exprs)

    # Interval between requestDate and boarding time
    booking_exprs = [
        (
            (pl.col("legs0_departureAt_dt") - pl.col("requestDate"))
            .dt.total_microseconds()
            .cast(pl.Float64)
            / 1e6
            / 3600
        )
        .fill_null(0)
        .pipe(lambda s: pl.when(s < 0).then(0).otherwise(s))
        .alias("hours_to_departure"),
        (
            (
                (pl.col("legs0_departureAt_dt") - pl.col("requestDate"))
                .dt.total_microseconds()
                .cast(pl.Float64)
                / 1e6
                < 48 * 3600
            )
        )
        .cast(pl.Int8)
        .alias("is_last_minute_booking"),
    ]
    time_exprs.extend(booking_exprs)
    df = df.with_columns(time_exprs)

    # # request hour
    # df = df.with_columns(pl.col("requestDate").dt.hour().alias("req_hour"))

    # df = df.with_columns(
    #     pl.when(pl.col("req_hour").is_between(0, 5))
    #     .then(0)
    #     .when(pl.col("req_hour").is_between(6, 8))
    #     .then(1)
    #     .when(pl.col("req_hour").is_between(9, 14))
    #     .then(2)
    #     .when(pl.col("req_hour").is_between(15, 17))
    #     .then(3)
    #     .otherwise(4)
    #     .alias("req_hour_bin")
    # )

    # overnight, short flight
    for leg in ["legs0", "legs1"]:
        df = df.with_columns(
            (
                (
                    (
                        pl.col(f"{leg}_arrivalAt_dt").dt.date()
                        != pl.col(f"{leg}_departureAt_dt").dt.date()
                    )
                )
                & (pl.col(f"{leg}_duration") > 6 * 60)
            )
            .cast(pl.Int8)
            .alias(f"{leg}_is_overnight_flight")
        )

        # bin
        df = df.with_columns(
            pl.when(pl.col(f"{leg}_duration") <= 60)
            .then(0)
            .when(pl.col(f"{leg}_duration") <= 105)
            .then(1)
            .when(pl.col(f"{leg}_duration") <= 360)
            .then(2)
            .when(pl.col(f"{leg}_duration") <= 600)
            .then(3)
            .otherwise(4)
            .alias(f"{leg}_duration_bin")
        )

    df = df.drop([f"{col}_dt" for col in time_cols] + ["requestDate"])

    return df


@timer
def build_rank_features(df):
    # First apply the columns that will be used for ranking
    # Price and duration basic ranks
    rank_exprs = []
    for col, alias in [
        ("totalPrice", "price"),
        ("total_duration", "duration"),
        ("legs0_duration", "leg0_duration"),
        ("legs1_duration", "leg1_duration"),
    ]:
        min_col = pl.col(col).min().over("ranker_id")
        max_col = pl.col(col).max().over("ranker_id")
        rank_exprs.extend(
            [
                ((pl.col(col) - min_col) / (max_col - min_col + 1e-9)).alias(
                    f"{alias}_quantile_rank"
                ),
            ]
        )

    df = df.with_columns(rank_exprs)

    # Interaction between ranks
    # NOTE: ratio is unstable
    eps = 1e-6
    df = df.with_columns(
        [
            (pl.col("price_quantile_rank") * pl.col("duration_quantile_rank")).alias(
                "rank_interaction_mul"
            ),
            (
                (
                    (pl.col("price_quantile_rank") + eps)
                    / (pl.col("duration_quantile_rank") + eps)
                ).clip(0.01, 100)
            ).alias("rank_interaction_ratio"),
            (pl.col("price_quantile_rank") + pl.col("duration_quantile_rank")).alias(
                "rank_interaction_sum"
            ),
            (pl.col("price_quantile_rank") - pl.col("duration_quantile_rank")).alias(
                "rank_interaction_sub"
            ),
            (
                pl.col("leg0_duration_quantile_rank")
                * pl.col("leg1_duration_quantile_rank")
            ).alias("leg_dur_interaction_mul"),
            (
                (pl.col("leg0_duration_quantile_rank") + eps)
                / (pl.col("leg1_duration_quantile_rank") + eps)
            )
            .clip(0.01, 100)
            .alias("leg_dur_interaction_ratio"),
            (
                pl.col("leg0_duration_quantile_rank")
                - pl.col("leg1_duration_quantile_rank")
            ).alias("leg_dur_interaction_sub"),
        ]
    )

    return df


@timer
def build_price_features(df: pl.DataFrame) -> pl.DataFrame:
    # ==== 1. 计算组内统计量（median, std, min, mean, max）====
    price_stats = df.group_by("ranker_id").agg(
        [
            pl.col("totalPrice").median().alias("grp_price_median"),
            pl.col("totalPrice").std().alias("grp_price_std"),
            pl.col("totalPrice").min().alias("grp_price_min"),
            pl.col("totalPrice").mean().alias("grp_price_mean"),
            pl.col("totalPrice").max().alias("grp_price_max"),
            pl.col("totalPrice").quantile(0.25, "nearest").alias("grp_price_q25"),
            pl.col("totalPrice").quantile(0.75, "nearest").alias("grp_price_q75"),
        ]
    )

    price_stats = price_stats.with_columns(
        [
            (pl.col("grp_price_q75") - pl.col("grp_price_q25")).alias("grp_price_iqr"),
        ]
    )

    df = df.join(price_stats, on="ranker_id", how="left")

    # ==== 2. 构造价格相关特征 ====
    price_exprs = [
        # 是否是 ranker_id 内 top3 最便宜
        (
            (pl.col("totalPrice").rank("dense").over("ranker_id") == 1)
            .cast(pl.Int8)
            .alias("is_top3_cheapest")
        ),
        # z-score（相对中位数）
        (
            (pl.col("totalPrice") - pl.col("grp_price_median"))
            / (pl.col("grp_price_std") + 1)
        )
        .alias("price_zscore_from_median")
        .fill_null(0),
        # 相对最小价格差比值
        (
            (pl.col("totalPrice") - pl.col("grp_price_min"))
            / (pl.col("grp_price_min") + 1)
        ).alias("price_relative_to_min"),
        # 是否便宜于均价、是否比均价+std还贵（outlier）
        (pl.col("totalPrice") < pl.col("grp_price_q25"))
        .cast(pl.Int8)
        .alias("is_cheaper_than_avg"),
        (
            pl.col("totalPrice")
            > (
                pl.col("grp_price_q75")
                + 1.5 * (pl.col("grp_price_q75") - pl.col("grp_price_q25"))
            )
        )
        .cast(pl.Int8)
        .alias("is_expensive_outlier"),
    ]

    df = df.with_columns(price_exprs)

    # ==== 3. 构造直达航班中最便宜 ====
    direct_cheapest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg(pl.col("totalPrice").min().alias("min_direct_price"))
    )

    df = (
        df.join(direct_cheapest, on="ranker_id", how="left")
        .with_columns(
            (
                (pl.col("is_direct_leg0") == 1)
                & (pl.col("totalPrice") == pl.col("min_direct_price"))
            )
            .cast(pl.Int8)
            .fill_null(0)
            .alias("is_direct_cheapest")
        )
        .drop("min_direct_price")
    )

    # ==== 4. 清理中间列 ====
    df = df.drop(
        [
            "grp_price_median",
            "grp_price_std",
            "grp_price_min",
            "grp_price_mean",
            "grp_price_max",
            "grp_price_q25",
            "grp_price_q75",
            "grp_price_iqr",
        ]
    )

    return df


@timer
def build_cabin_class_features(df: pl.DataFrame) -> pl.DataFrame:
    cabin_cols = [f"legs{l}_segments{s}_cabinClass" for l in (0, 1) for s in range(4)]

    # 填充空值，转成Int8
    df = df.with_columns(
        [pl.col(col).fill_null(0).cast(pl.Int8).alias(col) for col in cabin_cols]
    )

    # 是否有商务舱（等级3）
    df = df.with_columns(
        (pl.any_horizontal([pl.col(col) == 3 for col in cabin_cols]))
        .cast(pl.Int8)
        .alias("has_business_class")
    )

    # 是否有头等舱（等级4）
    df = df.with_columns(
        (pl.any_horizontal([pl.col(col) == 4 for col in cabin_cols]))
        .cast(pl.Int8)
        .alias("has_first_class")
    )

    df = df.with_columns(
        ((pl.col("isVip") == True) & pl.col("has_first_class")).alias("vip_first_class")
    )

    # 最高舱位等级（1-4），如果 cabin_cols 为空则默认为0
    df = df.with_columns(
        pl.max_horizontal([pl.col(col) for col in cabin_cols]).alias("max_cabin_level")
    )

    # 是否全部为经济舱（都是1），所有存在的舱位列都==1，且至少有一个段存在
    valid_mask = pl.any_horizontal(
        [pl.col(col).is_not_null() & (pl.col(col) != 0) for col in cabin_cols]
    )

    # 选出所有有效列的值，判断它们是否都等于1
    all_economy = (
        pl.all_horizontal(
            [
                (pl.col(col) == 1) | (pl.col(col).is_null()) | (pl.col(col) == 0)
                for col in cabin_cols
            ]
        )
        & valid_mask
    ).cast(pl.Int8)

    df = df.with_columns(all_economy.alias("all_cabin_level_1"))

    return df


@timer
def build_company_features(df, FULL):
    train_size = TRAIN_VAL_SIZE if not FULL else TRAIN_ALL_SIZE
    train_df = df.slice(0, train_size)
    selected_df = train_df.filter(pl.col("selected") == 1)

    avg_price = selected_df.select(pl.col("log_price").mean()).item()
    std_price = selected_df.select(pl.col("log_price").std()).item()

    company_pref = (
        selected_df.group_by("companyID")
        .agg(
            [
                pl.count().alias("company_ocurrence"),
                pl.col("log_price").mean().alias("avg_selected_price"),
                pl.col("log_price").std().alias("std_selected_price"),
                pl.col("is_direct_leg0").mean().alias("selected_legs0_direct_ratio"),
                pl.col("is_direct_leg1").mean().alias("selected_legs1_direct_ratio"),
                pl.col("both_direct").mean().alias("selected_direct_ratio"),
                (pl.col("legs0_departureAt_hour") < 6)
                .cast(pl.Int32)
                .mean()
                .alias("selected_legs0_night_ratio"),
                (pl.col("legs1_departureAt_hour") < 6)
                .cast(pl.Int32)
                .mean()
                .alias("selected_legs1_night_ratio"),
                pl.col("price_quantile_rank").mean().alias("company_avg_pct"),
                pl.col("duration_quantile_rank").mean().alias("company_avg_dct"),
                pl.col("corporate_policy_compliant")
                .mean()
                .alias("company_policy_rate"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("company_ocurrence") <= 1)
                .then(avg_price)
                .otherwise(pl.col("avg_selected_price"))
                .alias("avg_selected_price"),
            ]
        )
    )

    df = df.join(company_pref, on="companyID", how="left")

    df = df.with_columns(
        [
            pl.col("company_ocurrence").fill_null(0),
            pl.col("avg_selected_price").fill_null(avg_price),
            pl.col("std_selected_price").fill_null(std_price),
            pl.col("selected_legs0_direct_ratio").fill_null(0),
            pl.col("selected_legs1_direct_ratio").fill_null(0),
            pl.col("selected_direct_ratio").fill_null(0),
            pl.col("selected_legs0_night_ratio").fill_null(0),
            pl.col("selected_legs1_night_ratio").fill_null(0),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("totalPrice") - pl.col("avg_selected_price"))
                / (pl.col("std_selected_price") + 1e-6)
            ).alias("z_price_vs_company_selected"),
        ]
    )

    return df


@timer
def build_carrier_features(df: pl.DataFrame, leg: str, FULL: bool) -> pl.DataFrame:
    # 列名定义
    carrier_marketing_col = f"{leg}_segments0_marketingCarrier_code"
    carrier_operating_col = f"{leg}_segments0_operatingCarrier_code"
    # carrier_cols = [carrier_marketing_col, carrier_operating_col]
    carrier_cols = [carrier_marketing_col]
    alias_col = f"{leg}_Carrier_code"
    cabin_col = f"{leg}_segments0_cabinClass"
    dep_hour_col = f"{leg}_departureAt_hour"

    train_size = TRAIN_ALL_SIZE if FULL else TRAIN_VAL_SIZE
    train_df = df.slice(0, train_size)
    selected_df = train_df.filter(pl.col("selected") == 1)

    # === 1. 航司整体选择率特征 ===
    carrier_stats = (
        train_df.group_by(carrier_cols)
        .agg(total_count=pl.count(), selected_count=pl.col("selected").sum())
        .with_columns(
            [
                (pl.col("selected_count") / pl.col("total_count")).alias(
                    f"{alias_col}_selection_rate"
                ),
                pl.col("total_count")
                .cast(pl.Float64)
                .log1p()
                .alias(f"{alias_col}_log_total_count"),
                pl.col("selected_count")
                .cast(pl.Float64)
                .log1p()
                .alias(f"{alias_col}_log_selected_count"),
            ]
        )
        .select(
            carrier_cols
            + [
                f"{alias_col}_selection_rate",
                f"{alias_col}_log_total_count",
                f"{alias_col}_log_selected_count",
            ]
        )
    )

    # === 2. cabin 选择率特征 ===
    cabin_stats_wide_df = (
        selected_df.group_by(carrier_cols + [cabin_col])
        .agg(cabin_selected_count=pl.count())
        .join(
            selected_df.group_by(carrier_cols).agg(total_selected_count=pl.count()),
            on=carrier_cols,
            how="inner",
        )
        .with_columns(
            (
                pl.col("cabin_selected_count") / (pl.col("total_selected_count") + 1e-5)
            ).alias("cabin_select_ratio"),
            pl.col(cabin_col).cast(pl.Utf8),
        )
        .pivot(
            values="cabin_select_ratio",
            index=carrier_cols,
            columns=cabin_col,
        )
    )

    # 填充必须存在的 cabin 列
    cabin_cols = {
        "1": f"{alias_col}_cabin1_select_ratio",
        "2": f"{alias_col}_cabin2_select_ratio",
        "3": f"{alias_col}_cabin3_select_ratio",
        "4": f"{alias_col}_cabin4_select_ratio",
    }
    for raw_cabin, target_col in cabin_cols.items():
        cabin_stats_wide_df = cabin_stats_wide_df.with_columns(
            pl.col(raw_cabin).alias(target_col)
            if raw_cabin in cabin_stats_wide_df.columns
            else pl.lit(0.0).alias(target_col)
        )

    cabin_stats_wide_df = cabin_stats_wide_df.select(
        carrier_cols + list(cabin_cols.values())
    )

    # === 3. 价格、持续时间、时间等统计特征 ===
    selected_df = selected_df.with_columns(dep_hour=pl.col(dep_hour_col))
    stats_df = selected_df.group_by(carrier_cols).agg(
        [
            pl.col("price_quantile_rank").mean().alias(f"{alias_col}_avg_price_rank"),
            pl.col("duration_quantile_rank")
            .mean()
            .alias(f"{alias_col}_avg_duration_rank"),
            pl.col("dep_hour").mean().alias(f"{alias_col}_avg_dep_hour"),
            (pl.col("dep_hour") < 6)
            .cast(pl.Int32)
            .mean()
            .alias(f"{alias_col}_night_ratio"),
            pl.col("companyID").n_unique().alias(f"{alias_col}_company_diversity"),
            pl.col("profileId").n_unique().alias(f"{alias_col}_user_diversity"),
        ]
    )

    global_avg_price = selected_df.select(pl.col("price_quantile_rank").mean()).item()
    global_avg_duration = selected_df.select(
        pl.col("duration_quantile_rank").mean()
    ).item()

    # === 4. Join 合并所有特征 ===
    df = (
        df.join(carrier_stats, on=carrier_cols, how="left")
        .join(stats_df, on=carrier_cols, how="left")
        .join(cabin_stats_wide_df, on=carrier_cols, how="left")
        .with_columns(
            [
                pl.col(f"{alias_col}_selection_rate").fill_null(0.0),
                pl.col(f"{alias_col}_log_total_count").fill_null(0.0),
                pl.col(f"{alias_col}_log_selected_count").fill_null(0.0),
                pl.col(f"{alias_col}_avg_price_rank").fill_null(global_avg_price),
                pl.col(f"{alias_col}_avg_duration_rank").fill_null(global_avg_duration),
                pl.col(f"{alias_col}_avg_dep_hour").fill_null(12.0),
                pl.col(f"{alias_col}_night_ratio").fill_null(0.0),
                pl.col(f"{alias_col}_company_diversity").fill_null(0),
                pl.col(f"{alias_col}_user_diversity").fill_null(0),
            ]
        )
    )

    return df


@timer
def build_frequent_flyer_features(
    df: pl.DataFrame, major_carriers: list[list[str]]
) -> pl.DataFrame:
    for leg in [0]:
        carrier_marketing_col = f"legs{leg}_segments0_marketingCarrier_code"
        carrier_operating_col = f"legs{leg}_segments0_operatingCarrier_code"
        cabin_col = f"legs{leg}_segments0_cabinClass"

        # 是否是 frequentFlyer 的会员航司
        flag_ff_col = f"{carrier_marketing_col}_in_frequentFlyer"
        df = df.with_columns(
            pl.col("frequentFlyer")
            .fill_null("")
            .str.split("/")
            .list.contains(pl.col(carrier_marketing_col))
            .cast(pl.Int8)
            .alias(flag_ff_col)
        )

        # Only one
        df = df.with_columns(
            (
                (pl.col("frequentFlyer").fill_null("").str.split("/").list.len() == 1)
                & pl.col("frequentFlyer")
                .fill_null("")
                .str.contains(pl.col(carrier_marketing_col))
            )
            .cast(pl.Int8)
            .alias(f"{carrier_marketing_col}_is_only_frequentFlyer")
        )

        # 是否是 major carrier
        flag_major_col = f"is_major_carrier_{leg}_0"
        df = df.with_columns(
            pl.col(carrier_marketing_col)
            .is_in(major_carriers)
            .cast(pl.Int8)
            .alias(flag_major_col)
        )

        # frequent flyer + 低舱位
        df = df.with_columns(
            (
                (pl.col(flag_ff_col) == 1)
                & ((pl.col(cabin_col) == 1) | (pl.col(cabin_col) == 2))
            )
            .cast(pl.Int8)
            .alias(f"{carrier_marketing_col}_ff_and_economic")
        )

    return df


@timer
def build_profile_features(df, FULL=False):
    train_size = TRAIN_VAL_SIZE if not FULL else TRAIN_ALL_SIZE
    train_df = df.slice(0, train_size)
    train_selected = train_df.filter(pl.col("selected") == 1)

    agg = train_selected.group_by("profileId").agg(
        [
            pl.count().alias("user_selected_count"),
            pl.mean("price_quantile_rank").alias("user_avg_price_rank"),
            pl.mean("duration_quantile_rank").alias("user_avg_duration_rank"),
            pl.mean("legs0_segments0_cabinClass").alias(
                "user_avg_cabinClass_leg0_seg0"
            ),
        ]
    )

    total_counts = train_selected.group_by("profileId").agg(
        pl.count().alias("user_total_count")
    )
    profile_stats = agg.join(total_counts, on="profileId", how="left")

    # Use kde to approximate the peak
    price_arr = train_selected.select("price_quantile_rank").to_numpy().flatten()
    duration_arr = train_selected.select("duration_quantile_rank").to_numpy().flatten()
    cabin_arr = train_selected.select("legs0_segments0_cabinClass").to_numpy().flatten()

    price_mode = kde_mode(price_arr)
    duration_mode = kde_mode(duration_arr)
    cabin_mode = kde_mode(cabin_arr)

    fill_na_dict = {
        "user_selected_count": 0,
        "user_avg_price_rank": price_mode,
        "user_avg_duration_rank": duration_mode,
        "user_avg_cabinClass_leg0_seg0": cabin_mode,
    }

    df = df.join(profile_stats, on="profileId", how="left")
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in fill_na_dict.items()])

    df = df.with_columns(
        [
            (pl.col("price_quantile_rank") - pl.col("user_avg_price_rank")).alias(
                "price_rank_diff"
            ),
            (pl.col("duration_quantile_rank") - pl.col("user_avg_duration_rank")).alias(
                "duration_rank_diff"
            ),
        ]
    )

    df = df.drop(["user_avg_price_rank", "user_avg_duration_rank"])

    # TODO: 加入起飞、到达时间的用户习惯

    return df


@timer
def build_airport_feature(df, leg):
    airport = pl.read_csv(f"{DATA_DIR}/airports.csv")

    # 起飞机场字段保留的列
    departure_cols = [
        "IATA",
        "UTC_Offset_Hours",
        "Country_CodeA2",
        # "GeoPointLat",
        # "GeoPointLong",
    ]
    departure_airport_info = airport.select(departure_cols).rename(
        {
            col: f"{leg}_departure_airport_{col}"
            for col in departure_cols
            if col != "IATA"
        }
    )

    # 到达机场字段保留的列
    arrival_cols = [
        "IATA",
        "UTC_Offset_Hours",
        "Country_CodeA2",
        # "GeoPointLat",
        # "GeoPointLong",
    ]
    arrival_airport_info = airport.select(arrival_cols).rename(
        {col: f"{leg}_arrival_airport_{col}" for col in arrival_cols if col != "IATA"}
    )

    # join 起飞机场信息
    df = df.join(
        departure_airport_info,
        left_on=f"{leg}_segments0_departureFrom_airport_iata",
        right_on="IATA",
        how="left",
    )

    arrival_expr = (
        pl.when(pl.col("n_segments_leg0") == 1)
        .then(pl.col(f"{leg}_segments0_arrivalTo_airport_iata"))
        .when(pl.col("n_segments_leg0") == 2)
        .then(pl.col(f"{leg}_segments1_arrivalTo_airport_iata"))
        .when(pl.col("n_segments_leg0") == 3)
        .then(pl.col(f"{leg}_segments2_arrivalTo_airport_iata"))
        .when(pl.col("n_segments_leg0") == 4)
        .then(pl.col(f"{leg}_segments3_arrivalTo_airport_iata"))
        .otherwise(pl.lit(None))
        .alias("final_arrival_airport")
    )
    df = df.with_columns(arrival_expr)

    # join 到达机场信息
    df = df.join(
        arrival_airport_info,
        left_on="final_arrival_airport",
        right_on="IATA",
        how="left",
    ).drop("final_arrival_airport")

    # def haversine_distance(row):
    #     vals = list(row.values())
    #     if any(v is None for v in vals):
    #         return None
    #     lat1, lon1, lat2, lon2 = map(float, vals)
    #     return haversine((lat1, lon1), (lat2, lon2))

    df = df.with_columns(
        [
            # 是否跨国（不同国家）
            (
                pl.col(f"{leg}_departure_airport_Country_CodeA2")
                != pl.col(f"{leg}_arrival_airport_Country_CodeA2")
            )
            .cast(pl.Int8)
            .fill_null(0)
            .alias(f"{leg}_is_cross_country"),
            # 时区差（绝对值）
            (
                pl.col(f"{leg}_departure_airport_UTC_Offset_Hours")
                != pl.col(f"{leg}_arrival_airport_UTC_Offset_Hours")
            )
            .cast(pl.Boolean)
            .fill_null(False)
            .alias(f"{leg}_is_cross_timezone"),
            # # 经纬度距离（haversine）
            # pl.struct(
            #     [
            #         pl.col("departure_airport_GeoPointLat"),
            #         pl.col("departure_airport_GeoPointLong"),
            #         pl.col("arrival_airport_GeoPointLat"),
            #         pl.col("arrival_airport_GeoPointLong"),
            #     ]
            # )
            # .map_elements(haversine_distance)
            # .alias("geo_distance_km"),
        ]
    )

    # df = df.with_columns(
    #     (pl.col("totalPrice") * pl.col("geo_distance_km")).alias("price_distance_mul")
    # )

    return df


@timer
def build_corporateTariffCode_feature(df, FULL):
    train_size = TRAIN_VAL_SIZE if not FULL else TRAIN_ALL_SIZE
    train_df = df.slice(0, train_size)
    selected_df = train_df.filter(pl.col("selected") == 1)

    ctc_df = selected_df.group_by("corporateTariffCode").agg(
        [
            pl.count().alias("corporateTariffCode_hotness"),
            pl.col("price_quantile_rank")
            .mean()
            .alias("corporateTariffCode_price_rank_mean"),
            pl.col("duration_quantile_rank")
            .mean()
            .alias("corporateTariffCode_duration_rank_mean"),
        ]
    )

    no_ctc_df = selected_df.filter(pl.col("corporateTariffCode").is_null())
    null_hotness = no_ctc_df.height
    mean_price_rank_no_ctc = no_ctc_df["price_quantile_rank"].mean()
    mean_duration_rank_no_ctc = no_ctc_df["duration_quantile_rank"].mean()

    # df = df.with_columns(pl.col("corporateTariffCode").fill_null(-1))

    df = df.join(ctc_df, on="corporateTariffCode", how="left").with_columns(
        [
            pl.col("corporateTariffCode_hotness").fill_null(null_hotness),
            pl.col("corporateTariffCode_price_rank_mean").fill_null(
                mean_price_rank_no_ctc
            ),
            pl.col("corporateTariffCode_duration_rank_mean").fill_null(
                mean_duration_rank_no_ctc
            ),
        ]
    )

    return df


@timer
def build_additional_feature(df):
    additional_df = pl.read_parquet(f"{DATA_DIR}/raw_summary.parquet")

    REF_YEAR = 2024
    AGE_MIN, AGE_MAX = 2, 90
    AGE_BINS = [18, 25, 35, 45, 55, 65, 75]
    AGE_LABELS = ["2-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-92"]

    # Add an "outlier" category for ages outside the bins
    OUTLIER_LABEL = "outlier"
    age_bin_map = {label: idx for idx, label in enumerate(AGE_LABELS)}
    outlier_code = len(age_bin_map)
    age_bin_map[OUTLIER_LABEL] = outlier_code

    # 1. Join df with additional_df on ranker_id
    df = df.join(additional_df, on="ranker_id", how="left")

    # 2. Calculate age from yearOfBirth
    df = df.with_columns(
        (pl.lit(REF_YEAR) - pl.col("yearOfBirth").cast(pl.Int32)).alias("age")
    )

    # 3. Generate bin safely (None for out-of-range)
    valid_age_expr = (
        pl.when((pl.col("age") >= AGE_MIN) & (pl.col("age") <= AGE_MAX))
        .then(pl.col("age"))
        .otherwise(None)
    )

    df = df.with_columns(
        valid_age_expr.cut(breaks=AGE_BINS, labels=AGE_LABELS).alias("age_bin_fixed")
    )

    # 4. Fill outliers with OUTLIER_LABEL
    df = df.with_columns(
        pl.when(pl.col("age_bin_fixed").is_null())
        .then(pl.lit(OUTLIER_LABEL))
        .otherwise(pl.col("age_bin_fixed"))
        .alias("age_bin_fixed")
    )

    # 6. Handle age outliers
    df = df.with_columns(
        pl.when((pl.col("age") >= AGE_MIN) & (pl.col("age") <= AGE_MAX))
        .then(pl.col("age"))
        .otherwise(-1)
        .alias("age")
    )

    # 7. Drop intermediate columns if not needed
    df = df.drop(["yearOfBirth"])

    return df


@timer
def build_group_feature(df):
    df = df.with_columns(
        [
            pl.col("log_price").mean().over("ranker_id").alias("group_price_mean"),
            pl.col("log_price").std().over("ranker_id").alias("group_price_std"),
            (
                pl.col("log_price").max().over("ranker_id")
                - pl.col("log_price").min().over("ranker_id")
            ).alias("group_price_range"),
            pl.col("total_duration").mean().over("ranker_id").alias("group_dur_mean"),
            pl.col("total_duration").std().over("ranker_id").alias("group_dur_std"),
            (
                pl.col("total_duration").max().over("ranker_id")
                - pl.col("total_duration").min().over("ranker_id")
            ).alias("group_dur_range"),
            # pl.col("legs0_segments0_marketingCarrier_code")
            # .n_unique()
            # .over("ranker_id")
            # .alias("group_n_carriers"),
            pl.int_range(1, pl.count() + 1).over("ranker_id").alias("group_position"),
        ]
    )

    for leg in ["legs0", "legs1"]:
        carrier_col = f"{leg}_segments0_marketingCarrier_code"
        carrier_counts = df.group_by(["ranker_id", carrier_col]).agg(
            pl.count().alias(f"{leg}_carrier_count_in_group")
        )
        df = df.join(carrier_counts, on=["ranker_id", carrier_col], how="left")

    def safe_str(col):
        return (
            pl.when(col.is_not_null())
            .then(col.cast(pl.Utf8))
            .otherwise(pl.lit("missing"))
        )

    # Hash to segments
    df = df.with_columns(
        [
            pl.concat_str(
                [
                    safe_str(pl.col("legs0_departureAt")),
                    safe_str(pl.col("legs0_arrivalAt")),
                    safe_str(pl.col("legs0_segments0_flightNumber")),
                    safe_str(pl.col("legs0_segments1_flightNumber")),
                ],
                separator="_",
            ).alias("legs0_segment_id"),
            pl.concat_str(
                [
                    safe_str(pl.col("legs1_departureAt")),
                    safe_str(pl.col("legs1_arrivalAt")),
                    safe_str(pl.col("legs1_segments0_flightNumber")),
                    safe_str(pl.col("legs1_segments1_flightNumber")),
                ],
                separator="_",
            ).alias("legs1_segment_id"),
        ]
    )

    # Count of ocurrence
    legs0_counts = df.group_by(["ranker_id", "legs0_segment_id"]).agg(
        pl.count().alias("legs0_segment_count_in_group")
    )
    df = df.join(legs0_counts, on=["ranker_id", "legs0_segment_id"], how="left")

    legs1_counts = df.group_by(["ranker_id", "legs1_segment_id"]).agg(
        pl.count().alias("legs1_segment_count_in_group")
    )
    df = df.join(legs1_counts, on=["ranker_id", "legs1_segment_id"], how="left")

    df = df.with_columns(
        [
            pl.col("legs0_segment_id")
            .n_unique()
            .over("ranker_id")
            .alias("group_legs0_n_segments"),
            pl.col("legs1_segment_id")
            .n_unique()
            .over("ranker_id")
            .alias("group_legs1_n_segments"),
        ]
    )

    # Combination
    df = df.with_columns(
        pl.concat_str(
            [pl.col("legs0_segment_id"), pl.col("legs1_segment_id")], separator="_"
        ).alias("leg_combo")
    )

    df = df.with_columns(
        pl.count().over("ranker_id").alias("group_total_options"),
        pl.col("leg_combo").n_unique().over("ranker_id").alias("group_n_leg_combos"),
    )

    df = df.with_columns(
        (
            pl.col("totalPrice").rank("ordinal").over(["ranker_id", "leg_combo"])
            / pl.count().over(["ranker_id", "leg_combo"])
        ).alias("group_option_price_rank_ratio")
    )

    df = df.drop(["legs0_segment_id", "legs1_segment_id", "leg_combo"])

    return df


@timer
def build_aircraft_feature(df, FULL):
    train_size = TRAIN_VAL_SIZE if not FULL else TRAIN_ALL_SIZE
    train_df = df.slice(0, train_size)
    selected_df = train_df.filter(pl.col("selected") == 1)

    route_total = (
        selected_df.group_by("searchRoute").len().rename({"len": "route_total_flights"})
    )

    # 计算某航线某机型的航班数
    route_aircraft_freq = (
        selected_df.group_by(["searchRoute", "legs0_segments0_aircraft_code"])
        .len()
        .rename({"len": "route_aircraft_freq"})
    )

    # 合并，计算比例
    route_aircraft_stats = route_aircraft_freq.join(route_total, on="searchRoute")
    route_aircraft_stats = route_aircraft_stats.with_columns(
        (pl.col("route_aircraft_freq") / pl.col("route_total_flights")).alias(
            "route_aircraft_ratio"
        )
    ).select(["searchRoute", "legs0_segments0_aircraft_code", "route_aircraft_ratio"])

    # Mean rank
    route_aircraft_rank_mean = selected_df.group_by(
        ["searchRoute", "legs0_segments0_aircraft_code"]
    ).agg(
        [
            pl.col("price_quantile_rank").mean().alias("aircraft_avg_pct"),
            pl.col("duration_quantile_rank").mean().alias("aircraft_avg_dct"),
        ]
    )

    route_aircraft_stats = route_aircraft_stats.join(
        route_aircraft_rank_mean, on=["searchRoute", "legs0_segments0_aircraft_code"]
    )

    df = df.join(
        route_aircraft_stats,
        on=["searchRoute", "legs0_segments0_aircraft_code"],
        how="left",
    )

    return df


@timer
def build_label_feature(df: pl.DataFrame):
    label_df = pl.read_parquet(f"{DATA_DIR}/label.parquet")

    # NOTE: the row order in parquet file and json file are not the same
    df = df.with_columns(pl.arange(1, pl.len() + 1).over("ranker_id").alias("row_id"))

    # join
    df = df.join(
        label_df.select(["ranker_id", "row_id", "labels", "category"]),
        on=["ranker_id", "row_id"],
        how="left",
    )

    # all_labels = [
    #     "BestPrice",
    #     "BestPriceTravelPolicy",
    #     "BestPriceDirect",
    #     "Convenience",
    #     "MinTime",
    #     "BestPriceCorporateTariff",
    # ]

    # # binary feature
    # for label in all_labels:
    #     df = df.with_columns(
    #         (pl.col("labels").list.contains(label))
    #         .cast(pl.Int8)
    #         .alias(f"label_{label}")
    #     )
    # df = df.with_columns(pl.col("labels").list.len().log1p().alias("labels_count"))

    df = df.with_columns(
        (pl.col("labels").list.len() > 0).cast(pl.Int8).alias("has_any_label")
    )

    df = df.drop(["row_id", "labels"])

    return df


@timer
def build_option_feature(df: pl.DataFrame) -> pl.DataFrame:
    # 用于区分相同航班的hash字段
    flight_hash = (
        pl.col("legs0_departureAt").cast(str)
        + "_"
        + pl.col("legs0_arrivalAt").cast(str)
        + "_"
        + pl.col("legs1_departureAt").cast(str)
        + "_"
        + pl.col("legs1_arrivalAt").cast(str)
        + "_"
        + pl.col("legs0_segments0_flightNumber").cast(str)
        + "_"
        + pl.col("legs1_segments0_flightNumber").cast(str)
    ).alias("flight_hash")

    df = df.with_columns(flight_hash)

    # 提取同一航班在同一 ranker_id 下的价格统计特征
    df = df.with_columns(
        [
            pl.count().over(["ranker_id", "flight_hash"]).alias("flight_option_count"),
            pl.col("totalPrice")
            .min()
            .over(["ranker_id", "flight_hash"])
            .alias("flight_price_min"),
            pl.col("totalPrice")
            .rank("dense", descending=False)
            .over(["ranker_id", "flight_hash"])
            .alias("flight_price_rank"),
        ]
    )

    # Derived features
    df = df.with_columns(
        (pl.col("totalPrice") == pl.col("flight_price_min"))
        .cast(pl.Int8)
        .alias("is_flight_cheapest"),
        (pl.col("totalPrice") / (pl.col("flight_price_min") + 1e-6)).alias(
            "price_to_flight_min_ratio"
        ),
        (pl.col("flight_price_rank") / pl.col("flight_option_count")).alias(
            "flight_price_percentile"
        ),
        (pl.col("flight_option_count") == 1)
        .cast(pl.Int8)
        .alias("is_flight_only_option"),
    )

    # 删除不再需要的字段
    df = df.drop(["flight_price_min", "flight_price_rank", "flight_hash"])

    return df


@timer
def build_route_feature(df: pl.DataFrame, FULL: bool) -> pl.DataFrame:
    # Search Route
    df = df.with_columns(
        [
            pl.col("searchRoute").cast(pl.Utf8),
            pl.col("searchRoute").str.split_exact("/", 1).alias("route_struct"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("route_struct").struct.field("field_0").alias("outbound_route"),
            pl.col("route_struct").struct.field("field_1").alias("return_route"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("outbound_route").str.slice(0, 3).alias("outbound_origin"),
            pl.col("outbound_route").str.slice(3, 3).alias("outbound_destination"),
            pl.col("return_route").str.slice(0, 3).alias("return_origin"),
            pl.col("return_route").str.slice(3, 3).alias("return_destination"),
        ]
    )
    df = df.with_columns(
        [
            (
                (pl.col("outbound_origin") == pl.col("return_destination"))
                & (pl.col("outbound_destination") == pl.col("return_origin"))
            )
            .cast(pl.Int8)
            .alias("is_exact_round_trip")
            .fill_null(-1),
        ]
    )
    df = df.drop(["route_struct"])

    # Route-level statistical features based on selected samples
    train_size = TRAIN_VAL_SIZE if not FULL else TRAIN_ALL_SIZE
    train_df = df.slice(0, train_size)
    selected_df = train_df.filter(pl.col("selected") == 1)

    # Calculate route hotness
    outbound_hot = selected_df.group_by("outbound_route").agg(
        pl.len().alias("outbound_route_hotness")
    )
    return_hot = selected_df.group_by("return_route").agg(
        pl.len().alias("return_route_hotness")
    )

    # Compute outbound route statistics
    outbound_stats = selected_df.group_by("outbound_route").agg(
        [
            pl.mean("log_price").alias("outbound_route_avg_price"),
            pl.mean("total_duration").alias("outbound_route_avg_duration"),
            pl.mean("has_corporate_tariff").alias("outbound_route_policy_rate"),
            pl.mean("is_direct_leg0").alias("outbound_route_direct_ratio"),
        ]
    )

    # Compute return route statistics
    return_stats = selected_df.group_by("return_route").agg(
        [
            pl.mean("log_price").alias("return_route_avg_price"),
            pl.mean("total_duration").alias("return_route_avg_duration"),
            pl.mean("has_corporate_tariff").alias("return_route_policy_rate"),
            pl.mean("is_direct_leg1").alias("return_route_direct_ratio"),
        ]
    )

    # Calculate global averages to fill missing values after joining
    global_means = {
        "outbound_route_avg_price": selected_df["log_price"].mean(),
        "outbound_route_avg_duration": selected_df["total_duration"].mean(),
        "return_route_avg_price": selected_df["log_price"].mean(),
        "return_route_avg_duration": selected_df["total_duration"].mean(),
    }

    # Merge all route features back to the original dataframe
    df = (
        df.join(outbound_hot, on="outbound_route", how="left")
        .join(return_hot, on="return_route", how="left")
        .join(outbound_stats, on="outbound_route", how="left")
        .join(return_stats, on="return_route", how="left")
        .with_columns(
            [
                # Fill nulls with global averages instead of zero
                pl.col("outbound_route_hotness").fill_null(pl.lit(0)),
                pl.col("return_route_hotness").fill_null(pl.lit(0)),
                pl.col("outbound_route_policy_rate").fill_null(pl.lit(0)),
                pl.col("return_route_policy_rate").fill_null(pl.lit(0)),
                pl.col("outbound_route_avg_price").fill_null(
                    pl.lit(global_means["outbound_route_avg_price"])
                ),
                pl.col("outbound_route_avg_duration").fill_null(
                    pl.lit(global_means["outbound_route_avg_duration"])
                ),
                pl.col("return_route_avg_price").fill_null(
                    pl.lit(global_means["return_route_avg_price"])
                ),
                pl.col("return_route_avg_duration").fill_null(
                    pl.lit(global_means["return_route_avg_duration"])
                ),
            ]
        )
    )

    return df


@timer
def handle_features_with_extreme(df):
    # Taxes
    df = df.with_columns((pl.col("taxes")).log1p().alias("log_taxes"))
    # Winsorization of price per tax
    low = df.select(pl.col("price_per_tax").quantile(0.01)).item()
    high = df.select(pl.col("price_per_tax").quantile(0.99)).item()

    df = df.with_columns(
        pl.when(pl.col("price_per_tax") < low)
        .then(low)
        .when(pl.col("price_per_tax") > high)
        .then(high)
        .otherwise(pl.col("price_per_tax"))
        .alias("price_per_tax")
    )

    df = df.with_columns(
        [
            (pl.col("legs0_duration").log1p().alias("legs0_duration")),
            (pl.col("legs1_duration").log1p().alias("legs1_duration")),
            (
                pl.col("legs0_segments0_duration")
                .log1p()
                .alias("legs0_segments0_duration")
            ),
            (
                pl.col("legs0_segments1_duration")
                .log1p()
                .alias("legs0_segments1_duration")
            ),
            (
                pl.col("legs1_segments0_duration")
                .log1p()
                .alias("legs1_segments0_duration")
            ),
            (
                pl.col("legs1_segments1_duration")
                .log1p()
                .alias("legs1_segments1_duration")
            ),
            (pl.col("total_duration").log1p().alias("total_duration")),
            (pl.col("hours_to_departure").log1p().alias("hours_to_departure")),
        ]
    )
    return df


@timer
def feature_engineering(df, full):
    # Process duration columns
    dur_cols = ["legs0_duration", "legs1_duration"] + [
        f"legs{l}_segments{s}_duration" for l in (0, 1) for s in range(4)
    ]
    dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

    # Apply duration transformations first
    if dur_exprs:
        df = df.with_columns(dur_exprs)

    # Combine all initial transformations
    df = initial_transformations(df, full)

    # Segment feature
    df = build_segment_features(df)

    # More derived features
    df = df.with_columns(
        [
            (pl.col("is_direct_leg0") & pl.col("is_direct_leg1"))
            .cast(pl.Int32)
            .alias("both_direct"),
            ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0))
            .cast(pl.Int32)
            .alias("is_vip_freq"),
            # pl.col("baggage_min")
            # .fill_null(-1)
            # .map_elements(
            #     lambda x: 1 if x > 0 else (0 if x == 0 else -1), return_dtype=pl.Int32
            # )
            # .cast(pl.Int32)
            # .alias("has_baggage"),
            # (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
            # (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
            pl.col("Id").count().over("ranker_id").alias("group_size"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("legs0_segments0_baggageAllowance_weightMeasurementType").fill_null(
                -1
            ),
            pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(-1),
        ]
    )

    df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

    # Time features - batch process
    df = build_time_features(df)

    # Batch rank computations - more efficient with single pass
    df = build_rank_features(df)

    # Handle abnormal features
    df = handle_features_with_extreme(df)

    # Price-specific features
    df = build_price_features(df)

    # Cabin class features
    df = build_cabin_class_features(df)

    # Company features
    df = build_company_features(df, full)

    # Carrier features
    df = build_carrier_features(df, "legs0", full)
    df = build_carrier_features(df, "legs1", full)

    df = df.with_columns(
        [
            (
                pl.col("legs0_Carrier_code_log_total_count")
                + pl.col("legs1_Carrier_code_log_total_count")
            ).alias("carrier_pop_prod")
        ]
    )

    # Frequent Flyer features
    major_carriers = ["SU", "S7"]
    df = build_frequent_flyer_features(df, major_carriers)

    # Profile features
    df = build_profile_features(df, full)

    # Airport features
    df = build_airport_feature(df, "legs0")
    df = build_airport_feature(df, "legs1")

    # corporateTariffCode features
    df = build_corporateTariffCode_feature(df, full)

    # additional features from raw data
    # df = build_additional_feature(df)

    # Group features
    df = build_group_feature(df)

    # Aircraft features
    # df = build_aircraft_feature(df, full)

    # Label feature
    df = build_label_feature(df)

    # Option feature
    # df = build_option_feature(df)

    # Route feature
    df = build_route_feature(df, full)

    return df


###########################################################
#                      Feature Selection                  #
###########################################################


def feature_selection(data, trial=None):
    available_cols = set(data.columns)
    num_features, cat_features, exclude_cols = get_all_feature_lists()

    all_features = [f for f in cat_features + num_features if f in available_cols]
    if trial:
        all_features = [
            f for f in all_features if trial.suggest_categorical(f, [True, False])
        ]

    feature_cols = [col for col in all_features if col not in exclude_cols]
    cat_features_final = [col for col in cat_features if col in feature_cols]
    num_features_final = [col for col in num_features if col in feature_cols]

    print(
        f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical, {len(num_features_final)} numerical)"
    )

    X = data.select(feature_cols)
    y = data.select("selected")
    groups = data.select("ranker_id")

    return X, y, groups, cat_features_final, num_features_final
