"""Shared RavenPack query: DAILY news aggregates per entity, computed on the WRDS server.

Used by pull_pit.py and pull_pit_fallback.py so both apply exactly the same rules:
    - relevance >= MIN_RELEVANCE (company is a main subject of the story)
    - one count per story (a story can contain several events)
    - sentiment over all events (sent_all) and over non-price events only (sent_news)
    - effective date = the story's New York date, +1 day if at/after the 4 PM close
      (mapped to the next trading day later, in build_pit.py)
No story-level rows are downloaded.
"""

MIN_RELEVANCE = 90
PRICE_GROUPS = ("stock-prices", "technical-analysis")   # RavenPack event groups that just describe price moves


def daily_news_sql(year, entity_list_sql, price_groups_sql):
    """SQL for one year's table. entity_list_sql / price_groups_sql: already-quoted IN (...) lists."""
    return f"""
        WITH story AS (
            SELECT rp_entity_id, rp_story_id, min(timestamp_utc) AS ts,
                   avg(event_sentiment_score) AS sent_all,
                   avg(event_sentiment_score) FILTER (WHERE coalesce("group", '') NOT IN ({price_groups_sql})) AS sent_news,
                   bool_or(coalesce("group", '') NOT IN ({price_groups_sql})) AS has_news
            FROM ravenpack_dj.rpa_djpr_equities_{int(year)}
            WHERE relevance >= {MIN_RELEVANCE} AND rp_entity_id IN ({entity_list_sql})
            GROUP BY rp_entity_id, rp_story_id
        ), local AS (
            SELECT rp_entity_id, sent_all, sent_news, has_news,
                   (ts AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York' AS et
            FROM story
        )
        SELECT rp_entity_id,
               (CAST(et AS date) + CASE WHEN extract(hour FROM et) >= 16 THEN 1 ELSE 0 END) AS eff_date,
               count(*) AS n_all, avg(sent_all) AS sent_all,
               count(*) FILTER (WHERE has_news) AS n_news, avg(sent_news) FILTER (WHERE has_news) AS sent_news
        FROM local GROUP BY 1, 2
    """
