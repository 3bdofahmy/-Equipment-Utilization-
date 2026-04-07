/**
 * js/api.js
 * All fetch calls to FastAPI in one place.
 * Every other module imports from window.API.
 */

const API_BASE = window.API_BASE || "";

window.API = {

  async get(path) {
    try {
      const res = await fetch(API_BASE + path);
      if (res.status === 204) return null;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn("API error:", path, e.message);
      return null;
    }
  },

  health()              { return this.get("/health"); },
  modelInfo()           { return this.get("/model/info"); },
  modelPerformance()    { return this.get("/model/performance"); },
  modelStatus()         { return this.get("/model/status"); },
  equipment()           { return this.get("/equipment"); },
  utilization()         { return this.get("/utilization"); },
  utilizationById(id)   { return this.get(`/utilization/${id}`); },
  utilizationHistory(id, minutes = 30) {
    return this.get(`/utilization/${id}/history?minutes=${minutes}`);
  },
  detections(params = {}) {
    const q = new URLSearchParams(params).toString();
    return this.get(`/detections${q ? "?" + q : ""}`);
  },
  latestFrame()         { return this.get("/stream/latest-frame"); },
};
