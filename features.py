"""Cookie-local feature generators used by the fixed AntiBot solution."""
import numpy as np
import pandas as pd
USE_SOURCE_COMPATIBLE_SAME_SECOND_TRANSITIONS = False

RAW_EVENT_COLUMNS = [
    "cookie_id",
    "eid",
    "event_name",
    "platform",
    "user_agent",
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
    "event_ts",
]

SCRIPT_PATTERNS = (
    "scrapy/",
    "curl/",
    "python-urllib",
    "python-requests",
    "node-fetch",
    "go-http-client",
)

IMPORTANT_TEMPORAL_TRANSITIONS = [
    ("search_results_view", "item_view"),
    ("search_results_view", "search_results_view"),
    ("item_view", "item_view"),
    ("item_view", "search_results_view"),
    ("item_view", "photo_swipe"),
    ("item_view", "seller_page_view"),
    ("seller_page_view", "item_view"),
]

def normalize_platform_source(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip().lower()

    # Source solution объединяло desktop/web.
    if x in {"web", "desktop"}:
        return "web"

    if x == "android":
        return "android"

    if x in {"ios", "iphone"}:
        return "ios"

    return x

def enrich_event_level(ev):
    out = ev.copy()

    out["platform_source"] = (
        out["platform"]
        .map(normalize_platform_source)
    )

    ua = (
        out["user_agent"]
        .fillna("")
        .astype(str)
    )

    ua_lower = ua.str.lower()

    out["ua_is_headless"] = (
        ua_lower
        .str.contains(
            "headlesschrome",
            regex=False,
        )
        .astype("int8")
    )

    out["ua_is_script"] = (
        ua_lower
        .apply(
            lambda s: int(
                any(
                    pattern in s
                    for pattern in SCRIPT_PATTERNS
                )
            )
        )
        .astype("int8")
    )

    out["ua_is_avito_app"] = (
        ua.str.startswith("Avito/")
        .astype("int8")
    )

    out["ua_is_mobile"] = (
        ua_lower.str.contains("mobile", regex=False)
        | ua_lower.str.contains("android", regex=False)
        | ua_lower.str.contains("iphone", regex=False)
    ).astype("int8")

    out["ua_family"] = np.select(
        [
            out["ua_is_script"].astype(bool),
            out["ua_is_headless"].astype(bool),
            ua_lower.str.contains("yabrowser", regex=False),
            ua_lower.str.contains("firefox", regex=False),
            out["ua_is_avito_app"].astype(bool),
            ua_lower.str.contains("chrome", regex=False),
            ua_lower.str.contains("safari", regex=False),
        ],
        [
            "script",
            "headless",
            "yabrowser",
            "firefox",
            "avito_app",
            "chrome",
            "safari",
        ],
        default="other",
    )

    version = (
        ua.str.extract(r"HeadlessChrome/(\d+)", expand=False)
        .combine_first(
            ua.str.extract(r"YaBrowser/(\d+)", expand=False)
        )
        .combine_first(
            ua.str.extract(r"Firefox/(\d+)", expand=False)
        )
        .combine_first(
            ua.str.extract(r"Chrome/(\d+)", expand=False)
        )
        .combine_first(
            ua.str.extract(r"Avito/(\d+)", expand=False)
        )
        .combine_first(
            ua.str.extract(r"Version/(\d+)", expand=False)
        )
    )

    out["ua_major_version"] = pd.to_numeric(
        version,
        errors="coerce",
    )

    return out

PLATFORM_TYPES = [
    "web",
    "android",
    "ios",
    "unknown",
]

UA_FAMILIES = [
    "headless",
    "yabrowser",
    "firefox",
    "chrome",
    "safari",
    "avito_app",
    "script",
    "other",
]

def prepare_sequence_events(ev):
    subset = [
        c for c in RAW_EVENT_COLUMNS
        if c in ev.columns
    ]

    return (
        ev.drop_duplicates(
            subset=subset
        )
        .sort_values(
            ["cookie_id", "event_ts"],
            kind="mergesort",
        )
        .copy()
    )

def valid_directed_transition_mask(e):
    if USE_SOURCE_COMPATIBLE_SAME_SECOND_TRANSITIONS:
        return e["prev_event_name"].notna()

    return (
        e["prev_event_name"].notna()
        & e["prev_event_ts"].notna()
        & (e["event_ts"] > e["prev_event_ts"])
    )

def safe_div(num, den):
    return num / den.replace(0, np.nan)

def entropy_from_counts(frame):
    arr = frame.to_numpy(dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)

    p = np.divide(
        arr,
        row_sum,
        out=np.zeros_like(arr),
        where=row_sum != 0,
    )

    logp = np.zeros_like(p)
    mask = p > 0
    logp[mask] = np.log(p[mask])

    return -(p * logp).sum(axis=1)

def generate_source_basic_features(ev, meta, dup_stats):
    base = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)
    g = e.groupby("cookie_id", sort=False)

    base["n_events"] = g.size()

    base["item_nunique"] = g["item_id"].nunique()
    base["category_nunique"] = g["item_category"].nunique()
    base["location_nunique"] = g["item_location"].nunique()
    base["search_page_nunique"] = g["search_page"].nunique()

    item_views = (
        e["event_name"]
        .eq("item_view")
        .groupby(e["cookie_id"])
        .sum()
    )

    base["unique_items_per_item_event"] = safe_div(
        base["item_nunique"],
        item_views,
    )

    base["unique_categories_per_event"] = safe_div(
        base["category_nunique"],
        base["n_events"],
    )

    base["unique_locations_per_event"] = safe_div(
        base["location_nunique"],
        base["n_events"],
    )

    event_counts = pd.crosstab(
        e["cookie_id"],
        e["event_name"],
    )

    event_counts = event_counts.reindex(
        columns=EVENT_TYPES,
        fill_value=0,
    )

    for event_name in EVENT_TYPES:
        count = event_counts[event_name].reindex(base.index).fillna(0)

        base[f"event_ratio__{event_name}"] = safe_div(
            count,
            base["n_events"],
        )

    base["event_type_entropy"] = entropy_from_counts(
        event_counts.reindex(base.index).fillna(0)
    )

    search = e[
        e["event_name"].eq("search_results_view")
    ].copy()

    if len(search):
        sg = search.groupby("cookie_id")

        base["search_page_mean"] = sg["search_page"].mean()
        base["search_page_max"] = sg["search_page"].max()
        base["search_page_std"] = sg["search_page"].std()

        base["search_query_repeat_ratio"] = (
            1
            - safe_div(
                sg["search_query"].nunique(),
                sg["search_query"].count(),
            )
        )

    platform_counts = pd.crosstab(
        e["cookie_id"],
        e["platform_source"],
    )

    for platform in PLATFORM_TYPES:
        if platform not in platform_counts.columns:
            platform_counts[platform] = 0

        base[f"platform_ratio__{platform}"] = safe_div(
            platform_counts[platform].reindex(base.index).fillna(0),
            base["n_events"],
        )

    ua_counts = pd.crosstab(
        e["cookie_id"],
        e["ua_family"],
    )

    for family in UA_FAMILIES:
        if family not in ua_counts.columns:
            ua_counts[family] = 0

        base[f"ua_ratio__{family}"] = safe_div(
            ua_counts[family].reindex(base.index).fillna(0),
            base["n_events"],
        )

    base["pointer_present_ratio"] = (
        (
            e["pointer_x"].notna()
            & e["pointer_y"].notna()
        )
        .groupby(e["cookie_id"])
        .mean()
    )

    base["seller_type_present_ratio"] = (
        e["seller_type"]
        .notna()
        .groupby(e["cookie_id"])
        .mean()
    )

    ptr = e[
        e["pointer_x"].notna()
        & e["pointer_y"].notna()
    ].copy()

    if len(ptr):
        pg = ptr.groupby("cookie_id")

        base["pointer_count"] = pg.size()
        base["pointer_x_nunique"] = pg["pointer_x"].nunique()
        base["pointer_y_nunique"] = pg["pointer_y"].nunique()
        base["pointer_x_std"] = pg["pointer_x"].std()
        base["pointer_y_std"] = pg["pointer_y"].std()
        base["pointer_x_range"] = (
            pg["pointer_x"].max()
            - pg["pointer_x"].min()
        )
        base["pointer_y_range"] = (
            pg["pointer_y"].max()
            - pg["pointer_y"].min()
        )

        ptr["pointer_dx"] = (
            pg["pointer_x"].diff()
        )

        ptr["pointer_dy"] = (
            pg["pointer_y"].diff()
        )

        ptr["pointer_move"] = np.sqrt(
            ptr["pointer_dx"] ** 2
            + ptr["pointer_dy"] ** 2
        )

        move = (
            ptr.groupby("cookie_id")["pointer_move"]
            .agg(["mean", "median", "max"])
        )

        move.columns = [
            "pointer_move_mean",
            "pointer_move_median",
            "pointer_move_max",
        ]

        base = base.join(
            move,
            how="left",
        )

    # Transition ratios.
    e["prev_event_name"] = (
        e.groupby("cookie_id")["event_name"]
        .shift()
    )

    e["prev_event_ts"] = (
        e.groupby("cookie_id")["event_ts"]
        .shift()
    )

    valid_transition = (
        valid_directed_transition_mask(e)
    )

    trans = e.loc[
        valid_transition,
        [
            "cookie_id",
            "prev_event_name",
            "event_name",
        ],
    ].copy()

    trans["transition"] = (
        trans["prev_event_name"]
        + "__to__"
        + trans["event_name"]
    )

    if len(trans):
        counts = pd.crosstab(
            trans["cookie_id"],
            trans["transition"],
        )

        transition_total = (
            counts.sum(axis=1)
            .reindex(base.index)
        )

        base["transition_entropy"] = (
            entropy_from_counts(
                counts.reindex(base.index).fillna(0)
            )
        )

        for transition in counts.columns:
            base[
                f"transition_ratio__{transition}"
            ] = safe_div(
                counts[transition].reindex(base.index).fillna(0),
                transition_total,
            )

    # Correct duplicate rate from Stage 1, not from already-deduped stream.
    dup = (
        dup_stats.set_index("cookie_id")[
            "exact_duplicate_rate"
        ]
    )

    base["duplicate_ratio"] = dup.reindex(base.index)

    meta_idx = meta.set_index("cookie_id")

    base["cookie_age_days"] = (
        (
            meta_idx["window_start_ts"]
            - meta_idx["cookie_created_at"]
        )
        .dt.total_seconds()
        / 86400
    )

    return (
        base.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_temporal_features(ev, meta):
    temporal = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)

    g = e.groupby("cookie_id", sort=False)

    e["gap_sec"] = (
        g["event_ts"]
        .diff()
        .dt.total_seconds()
    )

    gap = e[
        e["gap_sec"].notna()
    ].copy()

    gg = gap.groupby("cookie_id")

    temporal["gap_mean_sec"] = gg["gap_sec"].mean()
    temporal["gap_std_sec"] = gg["gap_sec"].std()
    temporal["gap_min_sec"] = gg["gap_sec"].min()
    temporal["gap_max_sec"] = gg["gap_sec"].max()
    temporal["gap_median_sec"] = gg["gap_sec"].median()

    for q in [0.10, 0.25, 0.75, 0.90]:
        temporal[
            f"gap_q{int(q * 100):02d}_sec"
        ] = gg["gap_sec"].quantile(q)

    for threshold in [1, 5, 10, 30, 60]:
        temporal[
            f"gap_share_lt_{threshold}s"
        ] = (
            gap["gap_sec"]
            .lt(threshold)
            .groupby(gap["cookie_id"])
            .mean()
        )

    for threshold in [300, 600, 1800]:
        temporal[
            f"gap_share_gt_{threshold}s"
        ] = (
            gap["gap_sec"]
            .gt(threshold)
            .groupby(gap["cookie_id"])
            .mean()
        )

    first_ts = g["event_ts"].min()
    last_ts = g["event_ts"].max()

    temporal["active_span_sec"] = (
        last_ts - first_ts
    ).dt.total_seconds()

    n_events = g.size()

    temporal["events_per_active_minute"] = (
        n_events
        / (
            temporal["active_span_sec"]
            / 60
        ).clip(lower=1)
    )

    e["event_hour"] = e["event_ts"].dt.hour
    e["event_minute"] = (
        e["event_ts"].dt.floor("min")
    )

    temporal["active_hour_nunique"] = (
        e.groupby("cookie_id")["event_hour"]
        .nunique()
    )

    temporal["active_minute_nunique"] = (
        e.groupby("cookie_id")["event_minute"]
        .nunique()
    )

    minute_counts = (
        e.groupby(
            ["cookie_id", "event_minute"]
        )
        .size()
    )

    minute_stats = (
        minute_counts
        .groupby(level=0)
        .agg(["max", "mean"])
    )

    minute_stats.columns = [
        "max_events_per_minute",
        "mean_events_per_active_minute",
    ]

    temporal = temporal.join(
        minute_stats,
        how="left",
    )

    # 30-minute session baseline from strong source representation.
    e["new_session"] = (
        e["gap_sec"].isna()
        | e["gap_sec"].gt(1800)
    )

    e["session_id"] = (
        e.groupby("cookie_id")["new_session"]
        .cumsum()
    )

    session_sizes = (
        e.groupby(
            ["cookie_id", "session_id"]
        )
        .size()
    )

    session_stats = (
        session_sizes
        .groupby(level=0)
        .agg(["count", "mean", "max"])
    )

    session_stats.columns = [
        "session_count",
        "session_size_mean",
        "session_size_max",
    ]

    temporal = temporal.join(
        session_stats,
        how="left",
    )

    def gap_unique_ratio(x):
        x = x.dropna().round()

        if len(x) == 0:
            return np.nan

        return x.nunique() / len(x)

    def gap_mode_share(x):
        x = x.dropna().round()

        if len(x) == 0:
            return np.nan

        return (
            x.value_counts(
                normalize=True
            )
            .iloc[0]
        )

    temporal["gap_unique_ratio"] = (
        g["gap_sec"]
        .apply(gap_unique_ratio)
    )

    temporal["gap_mode_share"] = (
        g["gap_sec"]
        .apply(gap_mode_share)
    )

    gap_mean = g["gap_sec"].mean()
    gap_std = g["gap_sec"].std()

    temporal["gap_cv"] = (
        gap_std
        / gap_mean.replace(0, np.nan)
    )

    temporal["burstiness"] = (
        (gap_std - gap_mean)
        / (
            gap_std + gap_mean
        ).replace(0, np.nan)
    )

    e["prev_gap_sec"] = (
        e.groupby("cookie_id")["gap_sec"]
        .shift()
    )

    e["gap_change_abs"] = (
        e["gap_sec"]
        - e["prev_gap_sec"]
    ).abs()

    gap_change = (
        e.groupby("cookie_id")["gap_change_abs"]
    )

    temporal["gap_change_mean"] = (
        gap_change.mean()
    )

    temporal["gap_change_median"] = (
        gap_change.median()
    )

    e["fast_gap"] = (
        e["gap_sec"].lt(30)
    )

    prev_fast = (
        e.groupby("cookie_id")["fast_gap"]
        .shift()
        .fillna(False)
        .astype(bool)
    )

    e["fast_burst_start"] = (
        e["fast_gap"]
        & ~prev_fast
    )

    temporal["fast_burst_count"] = (
        e.groupby("cookie_id")[
            "fast_burst_start"
        ]
        .sum()
    )

    # Transition timing.
    e["prev_event_name"] = (
        e.groupby("cookie_id")["event_name"]
        .shift()
    )

    e["prev_event_ts"] = (
        e.groupby("cookie_id")["event_ts"]
        .shift()
    )

    transition_valid = (
        valid_directed_transition_mask(e)
    )

    for prev_event, current_event in IMPORTANT_TEMPORAL_TRANSITIONS:
        mask = (
            transition_valid
            & e["prev_event_name"].eq(prev_event)
            & e["event_name"].eq(current_event)
            & e["gap_sec"].notna()
        )

        tr = e.loc[
            mask,
            ["cookie_id", "gap_sec"],
        ]

        if len(tr) == 0:
            continue

        tg = (
            tr.groupby("cookie_id")["gap_sec"]
        )

        prefix = (
            f"transition_gap__{prev_event}"
            f"__to__{current_event}"
        )

        temporal[f"{prefix}__mean"] = tg.mean()
        temporal[f"{prefix}__median"] = tg.median()
        temporal[f"{prefix}__q25"] = tg.quantile(0.25)
        temporal[f"{prefix}__q75"] = tg.quantile(0.75)

    return (
        temporal.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_ua_features(ev, meta):
    features = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)
    g = e.groupby("cookie_id", sort=False)

    features["ua_is_mobile_ratio"] = (
        g["ua_is_mobile"].mean()
    )

    features["ua_major_version_nunique"] = (
        g["ua_major_version"].nunique()
    )

    prev_ua = (
        g["user_agent"].shift()
    )

    prev_family = (
        g["ua_family"].shift()
    )

    e["ua_changed"] = (
        prev_ua.notna()
        & e["user_agent"].ne(prev_ua)
    )

    e["ua_family_changed"] = (
        prev_family.notna()
        & e["ua_family"].ne(prev_family)
    )

    n_events = g.size()

    transition_count = (
        n_events - 1
    ).clip(lower=1)

    features["ua_change_ratio"] = (
        e.groupby("cookie_id")[
            "ua_changed"
        ]
        .sum()
        / transition_count
    )

    features["ua_family_change_ratio"] = (
        e.groupby("cookie_id")[
            "ua_family_changed"
        ]
        .sum()
        / transition_count
    )

    return (
        features.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_diversity_features(ev, meta):
    features = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)

    search = e[
        e["event_name"].eq(
            "search_results_view"
        )
    ].copy()

    page = search[
        search["search_page"].notna()
    ].copy()

    page["search_page"] = pd.to_numeric(
        page["search_page"],
        errors="coerce",
    )

    page = page[
        page["search_page"].notna()
    ]

    if len(page):
        features["page_ge5_share"] = (
            page["search_page"]
            .ge(5)
            .groupby(page["cookie_id"])
            .mean()
        )

    query_page = search[
        search["search_query"].notna()
        & search["search_page"].notna()
    ].copy()

    query_page["search_page"] = pd.to_numeric(
        query_page["search_page"],
        errors="coerce",
    )

    query_page = query_page[
        query_page["search_page"].notna()
    ]

    if len(query_page):
        per_query = (
            query_page
            .groupby(
                ["cookie_id", "search_query"]
            )["search_page"]
            .max()
        )

        features["mean_max_page_per_query"] = (
            per_query
            .groupby(level=0)
            .mean()
        )

    return (
        features.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_behavior_features(ev, meta):
    features = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)

    item_actions = e[
        e["item_id"].notna()
    ]

    if len(item_actions):
        actions_per_item = (
            item_actions
            .groupby(
                ["cookie_id", "item_id"]
            )
            .size()
        )

        features["mean_actions_per_item"] = (
            actions_per_item
            .groupby(level=0)
            .mean()
        )

    return (
        features.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_pointer_features(ev, meta):
    features = pd.DataFrame(
        index=pd.Index(
            meta["cookie_id"],
            name="cookie_id",
        )
    )

    e = prepare_sequence_events(ev)

    ptr = e[
        e["pointer_x"].notna()
        & e["pointer_y"].notna()
    ].copy()

    if len(ptr) == 0:
        return features.reset_index()

    pg = ptr.groupby("cookie_id")

    ptr["pointer_position"] = list(
        zip(
            ptr["pointer_x"],
            ptr["pointer_y"],
        )
    )

    position_count = pg.size()

    position_nunique = (
        ptr.groupby("cookie_id")[
            "pointer_position"
        ]
        .nunique()
    )

    features["pointer_position_unique_ratio"] = safe_div(
        position_nunique,
        position_count,
    )

    position_counts = (
        ptr.groupby(
            [
                "cookie_id",
                "pointer_x",
                "pointer_y",
            ]
        )
        .size()
    )

    top1_count = (
        position_counts
        .groupby(level=0)
        .max()
    )

    features["pointer_top1_position_share"] = safe_div(
        top1_count,
        position_count,
    )

    ptr["pointer_dx"] = (
        pg["pointer_x"].diff()
    )

    ptr["pointer_dy"] = (
        pg["pointer_y"].diff()
    )

    ptr["pointer_move"] = np.sqrt(
        ptr["pointer_dx"] ** 2
        + ptr["pointer_dy"] ** 2
    )

    mg = (
        ptr.groupby("cookie_id")[
            "pointer_move"
        ]
    )

    move_mean = mg.mean()
    move_std = mg.std()

    features["pointer_move_std"] = (
        move_std
    )

    features["pointer_move_cv"] = safe_div(
        move_std,
        move_mean,
    )

    return (
        features.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .reset_index()
    )

def generate_source_features(ev, meta, dup_stats):
    frames = [
        generate_source_basic_features(
            ev,
            meta,
            dup_stats,
        ),
        generate_source_temporal_features(
            ev,
            meta,
        ),
        generate_source_ua_features(
            ev,
            meta,
        ),
        generate_source_diversity_features(
            ev,
            meta,
        ),
        generate_source_behavior_features(
            ev,
            meta,
        ),
        generate_source_pointer_features(
            ev,
            meta,
        ),
    ]

    result = frames[0]

    seen = set(
        result.columns
    ) - {"cookie_id"}

    for frame in frames[1:]:
        current = set(
            frame.columns
        ) - {"cookie_id"}

        duplicated = (
            seen & current
        )

        if duplicated:
            raise ValueError(
                f"Duplicate features: {sorted(duplicated)}"
            )

        result = result.merge(
            frame,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )

        seen |= current

    return (
        result.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

import numpy as np
import pandas as pd

def describe(out,p,x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 for k,v in (dict(mean=x.mean(),std=x.std(),q25=np.quantile(x,.25),median=np.median(x),q75=np.quantile(x,.75),max=x.max()) if len(x) else dict.fromkeys(['mean','std','q25','median','q75','max'],np.nan)).items():out[p+'_'+k]=v


def build_pointer_features(events,meta):
 rows=[]
 for cid,e in events.groupby("cookie_id",sort=False):
  e=e.sort_values("event_ts",kind="stable");o={"cookie_id":cid}
  ptr=e.dropna(subset=['pointer_x','pointer_y']);xy=ptr[['pointer_x','pointer_y']].to_numpy()
  if len(xy):
   for k,axis in enumerate(['x','y']):describe(o,'ptr_'+axis,xy[:,k])
   med=np.median(xy,axis=0);r=np.linalg.norm(xy-med,axis=1);describe(o,'ptr_radius',r)
   if len(xy)>2:
    cov=np.cov(xy.T);val=np.linalg.eigvalsh(cov);o['ptr_anisotropy']=val[0]/(val[1]+1);o['ptr_area']=np.ptp(xy[:,0])*np.ptp(xy[:,1])
   # Median position per timestamp makes the trajectory invariant to tie ordering.
   pg=ptr.groupby('event_ts')[['pointer_x','pointer_y']].median();pos=pg.to_numpy();dt=np.diff(pg.index.to_numpy(dtype='datetime64[s]').astype('int64'));delta=np.diff(pos,axis=0);dist=np.linalg.norm(delta,axis=1)
   describe(o,'ptr_motion',dist)
   if len(dist):
    describe(o,'ptr_speed',dist/dt);o['ptr_stationary']=np.mean(dist==0);o['ptr_axis_aligned']=np.mean((delta==0).any(axis=1));o['ptr_straightness']=np.linalg.norm(pos[-1]-pos[0])/(dist.sum()+1)
   if len(delta)>1:
    cosine=np.sum(delta[:-1]*delta[1:],axis=1)/(dist[:-1]*dist[1:]+1e-6);describe(o,'ptr_turn_cos',cosine)
  rows.append(o)
 return pd.DataFrame(rows).set_index("cookie_id").reindex(meta.cookie_id).replace([np.inf,-np.inf],np.nan)

POINTER_COLUMNS = ['ptr_x_mean', 'ptr_x_std', 'ptr_x_q25', 'ptr_x_median', 'ptr_x_q75', 'ptr_x_max', 'ptr_y_mean', 'ptr_y_std', 'ptr_y_q25', 'ptr_y_median', 'ptr_y_q75', 'ptr_y_max', 'ptr_radius_mean', 'ptr_radius_std', 'ptr_radius_q25', 'ptr_radius_median', 'ptr_radius_q75', 'ptr_radius_max', 'ptr_anisotropy', 'ptr_area', 'ptr_motion_mean', 'ptr_motion_std', 'ptr_motion_q25', 'ptr_motion_median', 'ptr_motion_q75', 'ptr_motion_max', 'ptr_speed_mean', 'ptr_speed_std', 'ptr_speed_q25', 'ptr_speed_median', 'ptr_speed_q75', 'ptr_speed_max', 'ptr_stationary', 'ptr_axis_aligned', 'ptr_straightness', 'ptr_turn_cos_mean', 'ptr_turn_cos_std', 'ptr_turn_cos_q25', 'ptr_turn_cos_median', 'ptr_turn_cos_q75', 'ptr_turn_cos_max']
