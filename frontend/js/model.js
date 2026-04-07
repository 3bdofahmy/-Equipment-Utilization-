/**
 * js/model.js
 * Renders model info panel and updates health pills.
 */

window.Model = (() => {

  function renderInfo(info, perf) {
    const panel = document.getElementById("model-info-panel");
    if (!panel) return;
    if (!info) {
      panel.innerHTML = `<div class="no-data">MODEL NOT LOADED</div>`;
      return;
    }

    const classes = info.classes
      ? Object.values(info.classes).join(", ")
      : "–";

    const rows = [
      ["Name",       info.model_name   || "–"],
      ["Backend",    info.backend       || "–"],
      ["Task",       info.task          || "–"],
      ["Input size", info.input_size ? `${info.input_size}px` : "–"],
      ["Classes",    classes],
      ["Avg ms",     perf ? `${perf.avg_inference_ms} ms` : "–"],
      ["Avg FPS",    perf ? `${perf.avg_fps} FPS`         : "–"],
      ["GPU",        perf && perf.gpu_name ? perf.gpu_name : "–"],
      ["GPU Mem",    perf && perf.gpu_memory_used_mb
        ? `${perf.gpu_memory_used_mb} / ${perf.gpu_memory_total_mb} MB`
        : "–"],
    ];

    panel.innerHTML = rows.map(([k, v]) => `
      <div class="model-row">
        <span class="model-key">${k}</span>
        <span class="model-val">${v}</span>
      </div>`).join("");
  }

  function updateHealth(health) {
    _setPill("health-db",    health && health.database === "ok");
    _setPill("health-kafka", health && health.kafka    === "ok");
    _setPill("health-model", health && health.model    === "running");
  }

  function _setPill(id, ok) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("ok",    ok);
    el.classList.toggle("error", !ok);
  }

  return { renderInfo, updateHealth };
})();
