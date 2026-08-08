export interface WorkloadSummary {
  workload_id: string;
  label: string;
  source_id: string;
  licence: string;
  resource_kind: string;
  unit: string;
  step_seconds: number;
  bins: number;
  days: number;
  daily_autocorrelation: number;
  minute_autocorrelation: number;
  weekly_autocorrelation: number | null;
  peak_to_mean: number;
  mean_demand: number;
  forecastable: boolean;
  band: DiagnosticBand;
  verdict: string;
}

export type DiagnosticBand = 'strong' | 'borderline' | 'weak' | 'none';

export interface HorizonRow {
  horizon_label: string;
  horizon_steps: number;
  trailing_mase: number;
  seasonal_mase: number;
  winner: string;
}

export interface Finding {
  question_id: string;
  question: string;
  prior: string;
  answer: string;
  verdict: 'confirmed' | 'refuted' | 'mixed' | 'open';
  evidence: string;
}

export interface DemandPoint {
  ts: string;
  value: number;
  is_imputed: boolean;
}

export interface DemandSeries {
  workload_id: string;
  step_seconds: number;
  downsample_factor: number;
  points: DemandPoint[];
}

export interface Health {
  status: string;
  snapshot_mode: 'live' | 'demo' | 'replay';
  generated_at: string;
  workloads: number;
  delphi_version: string;
}

export interface WorkloadDetail {
  workload: WorkloadSummary;
  horizons: HorizonRow[];
  assumptions: string[];
}

export interface DiagnosticResult {
  bins: number;
  days: number;
  minute_autocorrelation: number | null;
  daily_autocorrelation: number | null;
  weekly_autocorrelation: number | null;
  peak_to_mean: number;
  forecastable: boolean;
  band: DiagnosticBand;
  verdict: string;
  q_star: number;
  demand_at_q_star: number;
  recommended_replicas: number;
  notes: string[];
}
