import { useEffect, useState } from 'react';
import { fetchFindings, fetchHealth, fetchWorkloads } from './api';
import type { Finding, Health, WorkloadSummary } from './types';
import Diagnostic from './views/Diagnostic';
import Findings from './views/Findings';
import HowItWorks from './views/HowItWorks';
import Workloads from './views/Workloads';

type Tab = 'diagnostic' | 'workloads' | 'findings' | 'how';

const TABS: { id: Tab; label: string }[] = [
  { id: 'diagnostic', label: 'Diagnostic' },
  { id: 'workloads', label: 'Workloads' },
  { id: 'findings', label: 'Findings' },
  { id: 'how', label: 'How it works' },
];

export default function App() {
  const [tab, setTab] = useState<Tab>('diagnostic');
  const [health, setHealth] = useState<Health | null>(null);
  const [workloads, setWorkloads] = useState<WorkloadSummary[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchWorkloads(), fetchFindings()])
      .then(([h, w, f]) => {
        setHealth(h);
        setWorkloads(w.workloads);
        setFindings(f.findings);
        setAssumptions(f.assumptions);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="shell">
      <header className="masthead">
        <h1>DELPHI</h1>
        <p className="tagline">
          Capacity planning that tells you whether prediction can help before it sells you a
          predictor — then sizes capacity from the price of failure rather than from convention.
        </p>
        <p style={{ margin: '0 0 14px' }}>
          {health ? (
            <>
              <span className={`badge ${health.snapshot_mode === 'live' ? 'good' : 'warn'}`}>
                {health.snapshot_mode}
              </span>{' '}
              <span className="muted mono" style={{ fontSize: '0.78rem' }}>
                {workloads.length} workloads · snapshot built{' '}
                {new Date(health.generated_at).toISOString().slice(0, 16).replace('T', ' ')} UTC
              </span>
            </>
          ) : (
            <span className="badge">connecting…</span>
          )}
        </p>
        <nav>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              aria-current={tab === t.id ? 'page' : undefined}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {error && (
        <section className="panel">
          <h2>Cannot reach the API</h2>
          <p className="error">{error}</p>
          <p className="sub">
            The snapshot may not be built. Run <code>python scripts/build_snapshot.py</code>, then{' '}
            <code>make api</code>.
          </p>
        </section>
      )}

      {!error && tab === 'diagnostic' && <Diagnostic workloads={workloads} />}
      {!error && tab === 'workloads' && workloads.length > 0 && (
        <Workloads workloads={workloads} />
      )}
      {!error && tab === 'findings' && (
        <Findings findings={findings} assumptions={assumptions} />
      )}
      {tab === 'how' && <HowItWorks />}

      <footer>
        Public traces only, all free and licence-checked. No API key is required for anything.
        DELPHI recommends capacity and never applies a change.
      </footer>
    </div>
  );
}
