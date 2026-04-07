/**
 * js/equipment.js
 * Renders equipment status cards from utilization data.
 */

window.Equipment = (() => {

  const ACTIVITY_COLORS = {
    "Digging":          "#ff8c42",
    "Swinging/Loading": "#ffd166",
    "Dumping":          "#c77dff",
    "Traveling":        "#4cc9f0",
    "Waiting":          "#6c757d",
  };

  function fmtTime(sec) {
    const s = Math.floor(sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(ss).padStart(2,"0")}`;
  }

  function render(utilData) {
    const container = document.getElementById("equipment-cards");
    const countEl   = document.getElementById("eq-count");
    if (!utilData || !utilData.length) {
      container.innerHTML = `<div class="no-data">NO EQUIPMENT DETECTED</div>`;
      if (countEl) countEl.textContent = "0 machines";
      return;
    }

    if (countEl) countEl.textContent = `${utilData.length} machine${utilData.length !== 1 ? "s" : ""}`;

    container.innerHTML = utilData.map(u => {
      const isActive  = (u.last_state || "").toUpperCase() === "ACTIVE";
      const cardClass = isActive ? "active" : "inactive";
      const stateClass= isActive ? "state-active" : "state-inactive";
      const stateText = isActive ? "ACTIVE" : "INACTIVE";
      const barW      = Math.min(100, Math.max(0, u.utilization_pct));
      const aColor    = ACTIVITY_COLORS[u.last_activity] || "#4a9fd4";
      const eqType    = (u.equipment_type || "unknown").replace("_", " ").toUpperCase();

      return `
        <div class="eq-card ${cardClass}">
          <div class="eq-card-top">
            <div>
              <div class="eq-id">${u.equipment_id}</div>
              <div class="eq-type">${eqType}</div>
            </div>
            <span class="state-badge ${stateClass}">${stateText}</span>
          </div>
          <div class="activity-tag" style="border-color:${aColor}44;color:${aColor};">
            ⚙ ${u.last_activity || "Waiting"}
          </div>
          <div class="util-bar-wrap">
            <div class="util-bar-label">
              <span>UTILISATION</span>
              <span class="util-bar-pct">${u.utilization_pct.toFixed(1)}%</span>
            </div>
            <div class="util-bar-bg">
              <div class="util-bar-fg" style="width:${barW}%"></div>
            </div>
          </div>
          <div class="eq-times">
            <span><span>🟢 ACTIVE</span><strong>${fmtTime(u.total_active_sec)}</strong></span>
            <span><span>🔴 IDLE</span><strong>${fmtTime(u.total_inactive_sec)}</strong></span>
          </div>
        </div>`;
    }).join("");
  }

  function updateFleetMetrics(utilData, perfData) {
    const total  = utilData ? utilData.length : 0;
    const active = utilData ? utilData.filter(u =>
      (u.last_state || "").toUpperCase() === "ACTIVE"
    ).length : 0;
    const avgUtil = utilData && utilData.length
      ? (utilData.reduce((s, u) => s + u.utilization_pct, 0) / utilData.length).toFixed(1) + "%"
      : "–";
    const fps = perfData ? perfData.avg_fps.toFixed(1) : "–";

    _set("m-total",  total  || "–");
    _set("m-active", active || "–");
    _set("m-util",   avgUtil);
    _set("m-fps",    fps);
  }

  function _set(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  return { render, updateFleetMetrics };
})();
