"""Initial provenance, workload, demand, capacity, and live-signal schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "demand_sources",
        sa.Column("source_id", sa.String(96), primary_key=True),
        sa.Column("vendor", sa.String(128), nullable=False),
        sa.Column("licence", sa.String(128), nullable=False),
        sa.Column("licence_verified_on", sa.Date(), nullable=False),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("terms_note", sa.Text(), nullable=False),
        sa.Column("epoch_assumption", sa.Text(), nullable=False),
    )
    op.create_table(
        "workload_profiles",
        sa.Column("workload_id", sa.String(160), primary_key=True),
        sa.Column(
            "source_id", sa.String(96), sa.ForeignKey("demand_sources.source_id"), nullable=False
        ),
        sa.Column("resource_kind", sa.String(32), nullable=False),
        sa.Column("unit", sa.String(64), nullable=False),
        sa.Column("capacity_per_replica", sa.Float(), nullable=False),
        sa.Column("startup_seconds", sa.Float(), nullable=False),
        sa.Column("teardown_seconds", sa.Float(), nullable=False),
        sa.Column("min_replicas", sa.Integer(), nullable=False),
        sa.Column("max_replicas", sa.Integer(), nullable=False),
        sa.Column("scale_to_zero", sa.Boolean(), nullable=False),
        sa.Column("tier", sa.String(32), nullable=False),
        sa.Column("deadline_slack_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("capacity_per_replica > 0", name="ck_profile_capacity_positive"),
        sa.CheckConstraint("startup_seconds >= 0", name="ck_profile_startup_nonnegative"),
        sa.CheckConstraint("teardown_seconds >= 0", name="ck_profile_teardown_nonnegative"),
        sa.CheckConstraint("min_replicas >= 0", name="ck_profile_min_replicas_nonnegative"),
        sa.CheckConstraint("max_replicas >= min_replicas", name="ck_profile_replica_bounds"),
    )
    op.create_index("ix_workload_profiles_source_id", "workload_profiles", ["source_id"])
    op.create_table(
        "demand_points",
        sa.Column(
            "workload_id",
            sa.String(160),
            sa.ForeignKey("workload_profiles.workload_id"),
            primary_key=True,
        ),
        sa.Column("ts", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("is_imputed", sa.Boolean(), nullable=False),
        sa.Column("quality", sa.Float(), nullable=False),
        sa.CheckConstraint("value >= 0", name="ck_demand_value_nonnegative"),
        sa.CheckConstraint("quality >= 0 AND quality <= 1", name="ck_demand_quality_unit"),
    )
    op.create_index("ix_demand_points_ts", "demand_points", ["ts"])
    op.create_table(
        "capacity_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "workload_id",
            sa.String(160),
            sa.ForeignKey("workload_profiles.workload_id"),
            nullable=False,
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("replicas_before", sa.Integer(), nullable=False),
        sa.Column("replicas_after", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(128), nullable=False),
    )
    op.create_index("ix_capacity_events_workload_ts", "capacity_events", ["workload_id", "ts"])
    op.create_table(
        "price_points",
        sa.Column("price_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sku", sa.String(256), nullable=False),
        sa.Column("region", sa.String(96), nullable=False),
        sa.Column("unit", sa.String(64), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(8), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_id", sa.String(96), sa.ForeignKey("demand_sources.source_id"), nullable=False
        ),
    )
    op.create_index(
        "ix_price_points_sku_region_retrieved", "price_points", ["sku", "region", "retrieved_at"]
    )
    op.create_table(
        "carbon_points",
        sa.Column("carbon_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("region", sa.String(96), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("intensity_gco2_kwh", sa.Float(), nullable=False),
        sa.Column("is_forecast", sa.Boolean(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column(
            "source_id", sa.String(96), sa.ForeignKey("demand_sources.source_id"), nullable=False
        ),
    )
    op.create_index("ix_carbon_points_region_ts", "carbon_points", ["region", "ts"])
    op.create_table(
        "collector_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column(
            "source_id", sa.String(96), sa.ForeignKey("demand_sources.source_id"), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_collector_runs_source_id", "collector_runs", ["source_id"])
    op.create_table(
        "coverage_outages",
        sa.Column("outage_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "source_id", sa.String(96), sa.ForeignKey("demand_sources.source_id"), nullable=False
        ),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cause", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_coverage_outages_source_start", "coverage_outages", ["source_id", "start_ts"]
    )


def downgrade() -> None:
    op.drop_index("ix_coverage_outages_source_start", table_name="coverage_outages")
    op.drop_table("coverage_outages")
    op.drop_index("ix_collector_runs_source_id", table_name="collector_runs")
    op.drop_table("collector_runs")
    op.drop_index("ix_carbon_points_region_ts", table_name="carbon_points")
    op.drop_table("carbon_points")
    op.drop_index("ix_price_points_sku_region_retrieved", table_name="price_points")
    op.drop_table("price_points")
    op.drop_index("ix_capacity_events_workload_ts", table_name="capacity_events")
    op.drop_table("capacity_events")
    op.drop_index("ix_demand_points_ts", table_name="demand_points")
    op.drop_table("demand_points")
    op.drop_index("ix_workload_profiles_source_id", table_name="workload_profiles")
    op.drop_table("workload_profiles")
    op.drop_table("demand_sources")
