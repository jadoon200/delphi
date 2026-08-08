import type {
  DemandSeries,
  DiagnosticResult,
  Finding,
  Health,
  WorkloadDetail,
  WorkloadSummary,
} from './types';

// Same-origin in the deployed single container; overridable for local split running.
const BASE = import.meta.env.VITE_API_BASE ?? '';

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const fetchHealth = () => get<Health>('/health');

export const fetchWorkloads = () =>
  get<{ workloads: WorkloadSummary[]; snapshot_mode: string; generated_at: string }>(
    '/workloads',
  );

export const fetchWorkload = (id: string) => get<WorkloadDetail>(`/workloads/${id}`);

export const fetchDemand = (id: string, limit = 900) =>
  get<DemandSeries>(`/demand/${id}?limit=${limit}`);

export const fetchFindings = () =>
  get<{ findings: Finding[]; assumptions: string[] }>('/findings');

export async function scoreDiagnostic(body: {
  values: number[];
  step_seconds: number;
  underage_cost: number;
  overage_cost: number;
  capacity_per_replica: number;
}): Promise<DiagnosticResult> {
  const response = await fetch(`${BASE}/diagnostic/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof detail.detail === 'string' ? detail.detail : 'scoring failed');
  }
  return (await response.json()) as DiagnosticResult;
}
