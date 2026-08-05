export default function HowItWorks() {
  return (
    <>
      <section className="panel">
        <h2>What this is</h2>
        <p className="sub">
          DELPHI is capacity-planning decision support. It answers two questions in order:{' '}
          <strong>can this workload be predicted at the lead time you care about</strong>, and{' '}
          <strong>how much capacity should you buy given what failure costs you</strong>. It
          recommends; it never applies a change to anything.
        </p>
      </section>

      <section className="panel">
        <h2>The pipeline</h2>
        <ol className="sub" style={{ paddingLeft: 20 }}>
          <li>
            <strong>Trace → demand series.</strong> Public traces (Azure LLM inference, Azure
            Functions, Bitbrains, Materna) become one canonical contract: regularly spaced,
            UTC-explicit, provenance-stamped. Gaps are flagged, never forward-filled.
          </li>
          <li>
            <strong>Diagnostic.</strong> Autocorrelation at one step, one day and one week. This
            decides whether anything downstream is worth building.
          </li>
          <li>
            <strong>Forecast → calibrated quantiles.</strong> No model's native quantile is
            trusted; coverage is measured on held-out time, because a controller consumes a
            quantile and therefore consumes calibration, not accuracy.
          </li>
          <li>
            <strong>Newsvendor sizing.</strong> <code>q* = C_u / (C_u + C_o)</code>. The
            compliance target is derived from the two prices rather than chosen by convention.
          </li>
          <li>
            <strong>Replay simulation.</strong> The plan is scored against the trace with real
            prices, churn cost and an explicit actuation delay.
          </li>
        </ol>
      </section>

      <section className="panel">
        <h2>What was actually found</h2>
        <p className="sub">
          The project was built to show that forecasting improves capacity control. Measured
          honestly, it mostly does not — and where it does is narrow and identifiable in advance.
        </p>
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Regime</th>
                <th>What wins</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>minutes</td>
                <td>autoscaling</td>
                <td>trailing percentile</td>
                <td>persistence r ≈ 0.98; a heuristic already extracts it</td>
              </tr>
              <tr>
                <td>1–2 hours</td>
                <td>commitment</td>
                <td>trailing percentile</td>
                <td>short enough that reaction still covers the error</td>
              </tr>
              <tr>
                <td>
                  <strong>6–12 hours</strong>
                </td>
                <td>commitment</td>
                <td>
                  <strong>forecasting, dominant</strong>
                </td>
                <td>must cover a window you cannot react within</td>
              </tr>
              <tr>
                <td>one week</td>
                <td>commitment</td>
                <td>a flat average</td>
                <td>no structure left to exploit</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="sub" style={{ marginTop: 14 }}>
          At a 12-hour commitment on the one workload with strong daily structure, forecasting
          halved the violation rate <em>while costing less</em>. On the two workloads with weak
          structure it never won, at any commitment length.
        </p>
      </section>

      <section className="panel">
        <h2>Limits — read these as carefully as the results</h2>
        <div className="callout limits">
          <strong>Replay is a counterfactual.</strong> A trace records what happened under the
          original operator's policy. Scoring a different policy against it is only valid because
          every arrival process used here is externally driven and cannot respond to provisioning.
          Rankings transfer between simulation and reality; absolute costs do not.
        </div>
        <div className="callout limits">
          <strong>Simulated numbers are a ceiling.</strong> Every simplification — the utilisation
          model, modelled service times, modelled cold starts — flatters the proactive methods.
          A sensitivity sweep bounds this; it cannot remove it.
        </div>
        <div className="callout limits">
          <strong>Nothing here is scaled for real.</strong> DELPHI produces a recommendation and a
          stated regret. A human decides.
        </div>
        <div className="callout limits">
          <strong>C<sub>u</sub> is a judgement, not a measurement.</strong> The cost of idle
          capacity is a live, citable price. The cost of a breach is a business call nobody can
          measure, so results are reported as a curve over it rather than a single number.
        </div>
        <div className="callout limits">
          <strong>Two traces carry unverifiable terms.</strong> The canonical Grid Workloads
          Archive host has been unreachable since 2026-08-02, so Bitbrains and Materna come from a
          mirror and their terms-of-use page could not be read. Checksums are recorded; the data is
          never redistributed.
        </div>
        <div className="callout limits">
          <strong>The diagnostic threshold is a rule of thumb.</strong> It is drawn from seven
          workloads, not fitted on a large sample. Treat it as a measurement with a number behind
          it, not a calibrated boundary.
        </div>
      </section>

      <section className="panel">
        <h2>Zero cost, by construction</h2>
        <p className="sub">
          Every dataset is free and licence-checked. Every live API used is keyless — Azure Retail
          Prices for the cost side of the newsvendor ratio, and the UK carbon-intensity feed. No
          API key is required for anything, the deploy runs on a free tier, and a test in the suite
          asserts that no hosted-model SDK can enter the dependency set.
        </p>
      </section>
    </>
  );
}
