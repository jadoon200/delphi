import { useMemo, useState } from 'react';
import { scoreDiagnostic } from '../api';
import type { DiagnosticResult, WorkloadSummary } from '../types';

/** Demand shapes a visitor can try without pasting anything.
 *
 * Each is seeded and carries realistic noise. A noiseless sine would score an
 * autocorrelation of exactly 1.000 at every lag, which is the kind of implausibly clean
 * number this project treats as an instrument artefact — it would make the demo look
 * rigged rather than illustrative. The targets below are chosen to land near the real
 * traces: ~0.73 daily for the structured shape, ~0.25 for the flat one.
 */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Box-Muller, so the noise is Gaussian rather than uniform. */
function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const DAY = 288; // 5-minute bins

const SAMPLES: Record<string, { label: string; note: string; build: () => number[] }> = {
  daily: {
    label: 'Strong daily rhythm',
    note: 'Business-hours traffic with realistic noise — the shape that rewards forecasting.',
    build: () => {
      const rand = seeded(20260805);
      return Array.from({ length: DAY * 10 }, (_, i) => {
        const phase = (2 * Math.PI * i) / DAY;
        const base = 60 + 26 * Math.sin(phase - Math.PI / 2) + 5 * Math.sin(phase * 3);
        return Math.max(1, base + gaussian(rand) * 11);
      });
    },
  },
  flat: {
    label: 'Persistent, no rhythm',
    note: 'A stable estate. Recent load predicts everything; an explicit forecast adds only its own error.',
    build: () => {
      const rand = seeded(7);
      let level = 50;
      return Array.from({ length: DAY * 10 }, () => {
        level += gaussian(rand) * 1.6;
        level = Math.max(12, Math.min(88, level));
        return level;
      });
    },
  },
  bursty: {
    label: 'Bursty and unpredictable',
    note: 'Spikes that yesterday does not predict. No amount of lead time rescues this one.',
    build: () => {
      const rand = seeded(99);
      return Array.from({ length: DAY * 10 }, () => {
        const spike = rand() < 0.025 ? 90 + rand() * 110 : 0;
        return Math.max(4, 38 + gaussian(rand) * 7 + spike);
      });
    },
  },
};

function Sparkline({ values }: { values: number[] }) {
  const path = useMemo(() => {
    if (!values.length) return '';
    const step = Math.max(1, Math.floor(values.length / 400));
    const sampled = values.filter((_, i) => i % step === 0);
    const min = Math.min(...sampled);
    const max = Math.max(...sampled);
    const range = max - min || 1;
    return sampled
      .map((v, i) => {
        const x = (i / (sampled.length - 1)) * 100;
        const y = 100 - ((v - min) / range) * 100;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  }, [values]);
  return (
    <svg className="spark" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function AutocorrBar({ label, value }: { label: string; value: number | null }) {
  if (value === null) {
    return (
      <div className="stat">
        <div className="label">{label}</div>
        <div className="value muted">n/a</div>
        <div className="note">series too short to measure</div>
      </div>
    );
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tone = value >= 0.5 ? 'good' : value >= 0.3 ? '' : 'bad';
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value.toFixed(3)}</div>
      <div className={`bar ${tone}`}>
        <span style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function Diagnostic({ workloads }: { workloads: WorkloadSummary[] }) {
  const [sample, setSample] = useState<keyof typeof SAMPLES>('daily');
  const [underage, setUnderage] = useState(19);
  const [overage, setOverage] = useState(1);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const values = useMemo(() => SAMPLES[sample].build(), [sample]);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await scoreDiagnostic({
          values,
          step_seconds: 300,
          underage_cost: underage,
          overage_cost: overage,
          capacity_per_replica: 10,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Before you build a predictive autoscaler, find out if prediction can help</h2>
        <p className="sub">
          Across seven workloads from three providers, a single measured number — the
          autocorrelation of demand at a one-day lag — predicted whether forecasting would beat
          the trailing-percentile recommender that Kubernetes already ships. It costs seconds to
          compute. Building the forecaster costs weeks.
        </p>

        <div className="grid two">
          <div>
            <label htmlFor="sample">Demand shape</label>
            <select
              id="sample"
              value={sample}
              onChange={(e) => {
                setSample(e.target.value as keyof typeof SAMPLES);
                setResult(null);
              }}
            >
              {Object.entries(SAMPLES).map(([key, s]) => (
                <option key={key} value={key}>
                  {s.label}
                </option>
              ))}
            </select>
            <p className="sub" style={{ marginTop: 8 }}>{SAMPLES[sample].note}</p>
            <Sparkline values={values} />
          </div>

          <div>
            <div className="grid two">
              <div>
                <label htmlFor="cu">Cost of unmet demand (C_u)</label>
                <input
                  id="cu"
                  type="number"
                  min={0.01}
                  step={1}
                  value={underage}
                  onChange={(e) => setUnderage(Math.max(0.01, Number(e.target.value)))}
                />
              </div>
              <div>
                <label htmlFor="co">Cost of idle capacity (C_o)</label>
                <input
                  id="co"
                  type="number"
                  min={0.01}
                  step={1}
                  value={overage}
                  onChange={(e) => setOverage(Math.max(0.01, Number(e.target.value)))}
                />
              </div>
            </div>
            <p className="sub" style={{ marginTop: 10 }}>
              These two prices decide how much to buy:{' '}
              <code>q* = C_u / (C_u + C_o)</code> ={' '}
              <strong>{(underage / (underage + overage)).toFixed(4)}</strong>. The default 19:1 is
              what a p95 target silently assumes — most autoscalers assert a price of failure
              they never state.
            </p>
            <button className="primary" onClick={run} disabled={busy}>
              {busy ? 'Scoring…' : 'Run the diagnostic'}
            </button>
            {error && <p className="error" style={{ marginTop: 10 }}>{error}</p>}
          </div>
        </div>

        {result && (
          <>
            <h3>Verdict</h3>
            <p>
              <span className={`badge ${result.forecastable ? 'good' : 'bad'}`}>
                {result.forecastable ? 'forecasting should pay' : 'forecasting will not pay'}
              </span>{' '}
              {result.verdict}
            </p>
            <div className="grid three" style={{ marginTop: 14 }}>
              <AutocorrBar label="autocorr @ 1 step" value={result.minute_autocorrelation} />
              <AutocorrBar label="autocorr @ 1 day" value={result.daily_autocorrelation} />
              <AutocorrBar label="autocorr @ 1 week" value={result.weekly_autocorrelation} />
            </div>
            <div className="grid three" style={{ marginTop: 14 }}>
              <div className="stat">
                <div className="label">derived q*</div>
                <div className="value">{result.q_star.toFixed(4)}</div>
                <div className="note">from your prices, not convention</div>
              </div>
              <div className="stat">
                <div className="label">demand at q*</div>
                <div className="value">{result.demand_at_q_star.toFixed(1)}</div>
                <div className="note">what to provision for</div>
              </div>
              <div className="stat">
                <div className="label">replicas</div>
                <div className="value">{result.recommended_replicas}</div>
                <div className="note">at 10 units served per replica</div>
              </div>
            </div>
            {result.notes.length > 0 && (
              <div className="callout">
                {result.notes.map((n) => (
                  <p key={n} style={{ margin: '4px 0' }}>{n}</p>
                ))}
              </div>
            )}
          </>
        )}
      </section>

      <section className="panel">
        <h2>The seven workloads this was measured on</h2>
        <p className="sub">
          Sorted by daily autocorrelation. Every trace is free and publicly available; two carry a
          licence caveat that is stated rather than hidden.
        </p>
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th>Workload</th>
                <th className="num">days</th>
                <th className="num">autocorr @ 1 day</th>
                <th className="num">p95 / mean</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {workloads.map((w) => (
                <tr key={w.workload_id}>
                  <td>
                    <code>{w.workload_id}</code>
                  </td>
                  <td className="num">{w.days.toFixed(0)}</td>
                  <td className="num">{w.daily_autocorrelation.toFixed(3)}</td>
                  <td className="num">{w.peak_to_mean.toFixed(2)}</td>
                  <td>
                    <span className={`badge ${w.forecastable ? 'good' : 'bad'}`}>
                      {w.forecastable ? 'forecastable' : 'not forecastable'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
