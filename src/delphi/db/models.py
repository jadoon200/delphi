"""Canonical provenance, workload, demand, capacity, and live-signal schema."""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from delphi.db.base import Base
from delphi.timeutil import utc_now


class DemandSource(Base):
    """Provenance and licence record for one trace or live signal."""

    __tablename__ = "demand_sources"

    source_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    vendor: Mapped[str] = mapped_column(String(128))
    licence: Mapped[str] = mapped_column(String(128))
    licence_verified_on: Mapped[date] = mapped_column(Date())
    citation: Mapped[str | None] = mapped_column(Text())
    url: Mapped[str] = mapped_column(String(2048))
    sha256: Mapped[str | None] = mapped_column(String(64))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    terms_note: Mapped[str] = mapped_column(Text(), default="")
    epoch_assumption: Mapped[str] = mapped_column(Text())


class WorkloadProfile(Base):
    """How one demand series translates into provisioned capacity."""

    __tablename__ = "workload_profiles"
    __table_args__ = (
        CheckConstraint("capacity_per_replica > 0", name="ck_profile_capacity_positive"),
        CheckConstraint("startup_seconds >= 0", name="ck_profile_startup_nonnegative"),
        CheckConstraint("teardown_seconds >= 0", name="ck_profile_teardown_nonnegative"),
        CheckConstraint("min_replicas >= 0", name="ck_profile_min_replicas_nonnegative"),
        CheckConstraint("max_replicas >= min_replicas", name="ck_profile_replica_bounds"),
    )

    workload_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("demand_sources.source_id"), index=True)
    resource_kind: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str] = mapped_column(String(64))
    capacity_per_replica: Mapped[float] = mapped_column(Float())
    startup_seconds: Mapped[float] = mapped_column(Float())
    teardown_seconds: Mapped[float] = mapped_column(Float())
    min_replicas: Mapped[int] = mapped_column(Integer())
    max_replicas: Mapped[int] = mapped_column(Integer())
    scale_to_zero: Mapped[bool] = mapped_column(Boolean(), default=False)
    tier: Mapped[str] = mapped_column(String(32))
    deadline_slack_seconds: Mapped[float | None] = mapped_column(Float())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DemandPoint(Base):
    """One regularly spaced, UTC-explicit observation for one workload."""

    __tablename__ = "demand_points"
    __table_args__ = (
        CheckConstraint("value >= 0", name="ck_demand_value_nonnegative"),
        CheckConstraint("quality >= 0 AND quality <= 1", name="ck_demand_quality_unit"),
        Index("ix_demand_points_ts", "ts"),
    )

    workload_id: Mapped[str] = mapped_column(
        ForeignKey("workload_profiles.workload_id"), primary_key=True
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[float] = mapped_column(Float())
    is_imputed: Mapped[bool] = mapped_column(Boolean(), default=False)
    quality: Mapped[float] = mapped_column(Float(), default=1.0)


class CapacityEvent(Base):
    __tablename__ = "capacity_events"
    __table_args__ = (Index("ix_capacity_events_workload_ts", "workload_id", "ts"),)

    event_id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    workload_id: Mapped[str] = mapped_column(ForeignKey("workload_profiles.workload_id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    replicas_before: Mapped[int] = mapped_column(Integer())
    replicas_after: Mapped[int] = mapped_column(Integer())
    reason: Mapped[str] = mapped_column(String(128))


class PricePoint(Base):
    __tablename__ = "price_points"
    __table_args__ = (
        Index("ix_price_points_sku_region_retrieved", "sku", "region", "retrieved_at"),
    )

    price_id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(String(256))
    region: Mapped[str] = mapped_column(String(96))
    unit: Mapped[str] = mapped_column(String(64))
    price: Mapped[float] = mapped_column(Float())
    currency: Mapped[str] = mapped_column(String(8))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_id: Mapped[str] = mapped_column(ForeignKey("demand_sources.source_id"))


class CarbonPoint(Base):
    __tablename__ = "carbon_points"
    __table_args__ = (Index("ix_carbon_points_region_ts", "region", "ts"),)

    carbon_id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    region: Mapped[str] = mapped_column(String(96))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    intensity_gco2_kwh: Mapped[float] = mapped_column(Float())
    is_forecast: Mapped[bool] = mapped_column(Boolean())
    horizon_minutes: Mapped[int] = mapped_column(Integer())
    source_id: Mapped[str] = mapped_column(ForeignKey("demand_sources.source_id"))


class CollectorRun(Base):
    __tablename__ = "collector_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("demand_sources.source_id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32))
    rows: Mapped[int] = mapped_column(Integer(), default=0)
    error: Mapped[str | None] = mapped_column(Text())


class CoverageOutage(Base):
    __tablename__ = "coverage_outages"
    __table_args__ = (Index("ix_coverage_outages_source_start", "source_id", "start_ts"),)

    outage_id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("demand_sources.source_id"))
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    cause: Mapped[str] = mapped_column(Text())
