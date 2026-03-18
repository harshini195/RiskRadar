-- RiskRadar Database Schema (PostgreSQL + PostGIS)
-- Run: psql -U postgres -d riskradar -f schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Raw accident records ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS accidents (
    id                  SERIAL PRIMARY KEY,
    location            GEOGRAPHY(Point, 4326) NOT NULL,
    latitude            DOUBLE PRECISION NOT NULL,
    longitude           DOUBLE PRECISION NOT NULL,
    severity            SMALLINT CHECK (severity BETWEEN 1 AND 3),
    road_type           VARCHAR(50),
    road_condition      VARCHAR(50),
    junction_control    VARCHAR(50),
    weather             VARCHAR(50),
    num_vehicles        SMALLINT DEFAULT 1,
    num_casualties      SMALLINT DEFAULT 0,
    accident_date       TIMESTAMP NOT NULL DEFAULT NOW(),
    description         TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_accidents_location ON accidents USING GIST (location);
CREATE INDEX idx_accidents_date     ON accidents (accident_date);

-- ── Hotspot clusters (output of DBSCAN) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS hotspots (
    id              SERIAL PRIMARY KEY,
    cluster_id      INTEGER NOT NULL,
    location        GEOGRAPHY(Point, 4326) NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    name            VARCHAR(200),
    accident_count  INTEGER DEFAULT 0,
    avg_severity    DOUBLE PRECISION,
    risk_score      DOUBLE PRECISION CHECK (risk_score BETWEEN 0 AND 1),
    risk_label      VARCHAR(20),
    main_cause      VARCHAR(100),
    computed_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hotspots_location   ON hotspots USING GIST (location);
CREATE INDEX idx_hotspots_risk_score ON hotspots (risk_score DESC);

-- ── Road segments (joined to risk predictions) ────────────────────────────
CREATE TABLE IF NOT EXISTS road_segments (
    id                  SERIAL PRIMARY KEY,
    osm_way_id          BIGINT,
    geom                GEOGRAPHY(LineString, 4326),
    road_name           VARCHAR(200),
    road_type           VARCHAR(50),
    speed_limit         INTEGER,
    risk_score          DOUBLE PRECISION,
    risk_label          VARCHAR(20),
    last_scored_at      TIMESTAMP
);

CREATE INDEX idx_road_segments_geom ON road_segments USING GIST (geom);

-- ── Useful spatial query: hotspots within radius ───────────────────────────
-- SELECT * FROM hotspots
-- WHERE ST_DWithin(
--     location,
--     ST_MakePoint(:lon, :lat)::geography,
--     :radius_meters
-- )
-- ORDER BY risk_score DESC;

-- ── Useful spatial query: accidents near a route polyline ──────────────────
-- SELECT COUNT(*) FROM accidents
-- WHERE ST_DWithin(
--     location,
--     ST_GeomFromText('LINESTRING(...)', 4326)::geography,
--     500   -- 500 metres buffer
-- );
