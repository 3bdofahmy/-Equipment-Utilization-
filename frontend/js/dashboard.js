/**
 * js/dashboard.js
 * Main controller — wires all modules together.
 */

(async () => {

  const FAST_INTERVAL  = 2000;   // utilization + cards
  const SLOW_INTERVAL  = 10000;  // model info + health
  const DET_INTERVAL   = 5000;   // detection log

  // ── Clock ──────────────────────────────────────────────────────────
  function updateClock() {
    const el = document.getElementById("clock");
    if (el) el.textContent = new Date().toUTCString().replace("GMT", "UTC");
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ── Init charts and stream ─────────────────────────────────────────
  Charts.init();
  Stream.init();
  Stream.start();

  // ── Detection log renderer ─────────────────────────────────────────
  function renderDetectionLog(detections) {
    const tbody   = document.getElementById("detection-tbody");
    const countEl = document.getElementById("log-count");
    if (!tbody) return;
    if (!detections || !detections.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="no-data">NO DETECTIONS YET</td></tr>`;
      return;
    }
    if (countEl) countEl.textContent = `${detections.length} entries`;

    tbody.innerHTML = detections.slice(0, 80).map(d => {
      const isActive   = d.utilization_state === "ACTIVE";
      const stateClass = isActive ? "state-active-cell" : "state-inactive-cell";
      const t          = new Date(d.time).toLocaleTimeString();
      const motion     = (d.motion_score * 100).toFixed(1) + "%";
      const llm        = d.llm_verified
        ? `<span class="llm-verified">✓</span>`
        : `<span style="color:var(--text-muted)">–</span>`;

      return `
        <tr>
          <td class="mono">${t}</td>
          <td class="mono">${d.equipment_id}</td>
          <td>${(d.equipment_type || "").replace("_"," ")}</td>
          <td class="${stateClass}">${d.utilization_state}</td>
          <td>${d.activity}</td>
          <td class="mono">${motion}</td>
          <td>${llm}</td>
        </tr>`;
    }).join("");
  }

  // ── Fast refresh (utilization + cards + charts) ───────────────────
  async function fastRefresh() {
    const [utilData, perf] = await Promise.all([
      API.utilization(),
      API.modelPerformance(),
    ]);
    Equipment.render(utilData);
    Equipment.updateFleetMetrics(utilData, perf);
    if (utilData) Charts.updateUtil(utilData);
  }

  // ── Detection refresh ──────────────────────────────────────────────
  async function detectionRefresh() {
    const detections = await API.detections({ minutes: 5, limit: 100 });
    if (detections) {
      renderDetectionLog(detections);
      Charts.updateActivity(detections);
    }
  }

  // ── Slow refresh (health + model info) ────────────────────────────
  async function slowRefresh() {
    const [health, info, perf] = await Promise.all([
      API.health(),
      API.modelInfo(),
      API.modelPerformance(),
    ]);
    Model.updateHealth(health);
    Model.renderInfo(info, perf);
  }

  // ── Run immediately then on intervals ────────────────────────────
  await Promise.all([fastRefresh(), detectionRefresh(), slowRefresh()]);

  setInterval(fastRefresh,      FAST_INTERVAL);
  setInterval(detectionRefresh, DET_INTERVAL);
  setInterval(slowRefresh,      SLOW_INTERVAL);

})();
