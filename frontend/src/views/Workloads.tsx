import { useEffect, useMemo, useState } from 'react';
import { fetchDemand, fetchWorkload } from '../api';
import type { DemandSeries, WorkloadDetail, WorkloadSummary } from '../types';

function DemandChart({ series }: { series: DemandSeries }) {
  const { path, imputed, max } = useMemo(() => {
    const points = series.points;
    if (!points.length) return { path: '', imputed: [] as string[], max: 0 };
    const values = points.map((p) => p.value);
    const hi = Math.max(...values) || 1;
    const d = points
      .map((p, i) => {
        const x = (i / (points.length - 1)) * 100;
        const y = 100 - (p.value / hi) * 100;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(3)},${y.toFixed(3)}`;
      })
      .join(' ');
    const gaps = points
      .map((p, i) => (p.is_imputed ? ((i / (points.length - 1)) * 100).toFixed(3) : null))
      .filter((v): v is string => v !== null);
    return { path: d, imputed: gaps, max: hi };
  }, [series]);

  return (
    <>
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ width: '100%', height: 190, display: 'block' }}
        role="img"
        aria-label="demand over time"
      >
        {imputed.map((x) => (
          <line key={x} x1={x} y1="0" x2={x} y2="100" stroke="var(--bad)" strokeWidth="0.3" opacity="0.5" />
        ))}
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </svg>
      <p className="sub" style={{ marginTop: 8 }}>
        Peak {max.toFixed(0)}. Averaged in blocks of {series.downsample_factor} to bound the
        response. Red marks bins flagged as imputed — where the estate shrank rather than idled,
        which is a different question and is never silently read as low demand.
      </p>
    </>
  );
}

export default function Workloads({ workloads }: { workloads: WorkloadSummary[] }) {
  const [selected, setSelected] = useState(workloads[0]?.workload_id ?? '');
  const [detail, setDetail] = useState<WorkloadDetail | null>(null);
  const [series, setSeries] = useState<DemandSeries | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    setSeries(null);
    setError(null);
    fetchWorkload(selected).then(setDetail).catch((e) => setError(String(e)));
    fetchDemand(selected).then(setSeries).catch(() => setSeries(null));
  }, [selected]);

  return (
    <>
      <section className="panel">
        <h2>Where each predictor wins, per workload</h2>
        <p className="sub">
          The crossover moves with the workload. High daily autocorrelation makes seasonal
          forecasting win at shorter lead times; below roughly 0.25 it never wins at any horizon.
          That ordering held across all seven traces.
        </p>
        <label htmlFor="wl">Workload</label>
        <select id="wl" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {workloads.map((w) => (
            <option key={w.workload_id} value={w.workload_id}>
              {w.label} — r_day {w.daily_autocorrelation.toFixed(3)}
            </option>
          ))}
        </select>
        {error && <p className="error">{error}</p>}
      </section>

      {detail && (
        <section className="panel">
          <h2>{detail.workload.label}</h2>
          <p className="sub">
            <code>{detail.workload.source_id}</code> · {detail.workload.bins.toLocaleString()} bins
            of {detail.workload.step_seconds}s · {detail.workload.days.toFixed(0)} days ·{' '}
            {detail.workload.resource_kind} in {detail.workload.unit}
          </p>
          <div className="callout">
            <strong>Licence:</strong> {detail.workload.licence}
          </div>

          {series && <DemandChart series={series} />}

          <div className="grid three" style={{ marginTop: 16 }}>
            <div className="stat">
              <div className="label">autocorr @ 1 day</div>
              <div className="value">{detail.workload.daily_autocorrelation.toFixed(3)}</div>
            </div>
            <div className="stat">
              <div className="label">autocorr @ 1 week</div>
              <div className="value">
                {detail.workload.weekly_autocorrelation === null
                  ? 'n/a'
                  : detail.workload.weekly_autocorrelation.toFixed(3)}
              </div>
              <div className="note">
                {detail.workload.weekly_autocorrelation === null
                  ? 'trace shorter than two weeks'
                  : 'measurable'}
              </div>
            </div>
            <div className="stat">
              <div className="label">p95 / mean</div>
              <div className="value">{detail.workload.peak_to_mean.toFixed(2)}</div>
              <div className="note">how peaky</div>
            </div>
          </div>

          {detail.horizons.length > 0 && (
            <>
              <h3>Forecast error by lead time (MASE — lower is better)</h3>
              <div className="scroll-x">
                <table>
                  <thead>
                    <tr>
                      <th>Lead time</th>
                      <th className="num">trailing window</th>
                      <th className="num">same time yesterday</th>
                      <th>Winner</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.horizons.map((h) => (
                      <tr key={h.horizon_label}>
                        <td>{h.horizon_label}</td>
                        <td className="num">{h.trailing_mase.toFixed(3)}</td>
                        <td className="num">{h.seasonal_mase.toFixed(3)}</td>
                        <td>
                          <span className={`badge ${h.winner === 'seasonal' ? 'good' : ''}`}>
                            {h.winner}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}
    </>
  );
}
