from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    universal_transaction_number: Optional[str] = None
    reversal_flag: Optional[int] = None
    svfe_trace_number: Optional[str] = None
    svfe_message_type: Optional[str] = None
    response_code: Optional[str] = None
    svfe_transaction_type: Optional[str] = None
    svfe_response_code: Optional[str] = None
    completion_status: Optional[int] = None
    stood_in_for: Optional[int] = None
    issuer_posted: Optional[int] = None
    ts: datetime
    gap_to_next_sec: Optional[float] = None
    gap_label: Optional[str] = None
    local_expected_gap_sec: Optional[float] = None
    local_ceiling_sec: Optional[float] = None
    duration_to_expected_ratio: Optional[float] = None
    hour_of_day: Optional[float] = None
    day_of_week: Optional[float] = None
    is_active_hours: Optional[bool] = None
    minutes_from_open: Optional[float] = None
    minutes_from_close: Optional[float] = None
    confidence_tier: Optional[str] = None


class BulkTransactionsIn(BaseModel):
    records: list[TransactionIn] = Field(default_factory=list)


class PipelineRunRequest(BaseModel):
    predict_day: date
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    granularity: str = "H"
    model_version: str = "statistical_v1"


class PipelineRunResponse(BaseModel):
    prediction_run_ts: datetime
    predict_day: date
    train_start: date
    train_end: date
    model_version: str
    granularity: str
    rows_written: int


class HealthResponse(BaseModel):
    status: str
    database: str
