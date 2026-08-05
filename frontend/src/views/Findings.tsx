import type { Finding } from '../types';

const TONE: Record<Finding['verdict'], string> = {
  confirmed: 'good',
  refuted: 'bad',
  mixed: 'warn',
  open: 'warn',
};

export default function Findings({
  findings,
  assumptions,
}: {
  findings: Finding[];
  assumptions: string[];
}) {
  const refuted = findings.filter((f) => f.verdict === 'refuted').length;

  return (
    <>
      <section className="panel">
        <h2>Pre-registered questions, answered</h2>
        <p className="sub">
          Each question below was written down with a stated prior <em>before</em> the measurement
          existed, so the answer could not be quietly reframed afterwards.{' '}
          <strong>{refuted} of {findings.length} came back refuted</strong> — including the one the
          project was originally built around. Those are kept in full rather than dropped.
        </p>
      </section>

      {findings.map((f) => (
        <section className="panel" key={f.question_id}>
          <h2>
            <span className={`badge ${TONE[f.verdict]}`}>{f.verdict}</span>{' '}
            <span className="mono muted">{f.question_id}</span> {f.question}
          </h2>
          <h3>Prior</h3>
          <p className="sub" style={{ margin: 0 }}>{f.prior}</p>
          <h3>What the measurement said</h3>
          <p style={{ margin: 0 }}>{f.answer}</p>
          <h3>Evidence</h3>
          <p className="sub" style={{ margin: 0 }}>{f.evidence}</p>
        </section>
      ))}

      <section className="panel">
        <h2>Standing caveats</h2>
        <p className="sub">
          These travel with every number on this site, as structured data on every API response
          that shows a comparison — not as a footnote.
        </p>
        {assumptions.map((a) => (
          <div className="callout limits" key={a.slice(0, 40)}>
            {a}
          </div>
        ))}
      </section>
    </>
  );
}
