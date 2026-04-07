/**
 * js/charts.js
 * Utilisation bar chart + activity doughnut chart using Chart.js.
 */

window.Charts = (() => {

  const CHART_DEFAULTS = {
    color:      "#d0d8e8",
    gridColor:  "#1e2a3a",
    bgColor:    "#0f141c",
  };

  const ACTIVITY_COLORS = {
    "Digging":          "#ff8c42",
    "Swinging/Loading": "#ffd166",
    "Dumping":          "#c77dff",
    "Traveling":        "#4cc9f0",
    "Waiting":          "#6c757d",
  };

  let _utilChart     = null;
  let _activityChart = null;

  function init() {
    Chart.defaults.color     = CHART_DEFAULTS.color;
    Chart.defaults.font.family = "'Barlow', sans-serif";

    _utilChart = new Chart(
      document.getElementById("util-chart"),
      {
        type: "bar",
        data: { labels: [], datasets: [{
          label:           "Utilisation %",
          data:            [],
          backgroundColor: "#00e5a044",
          borderColor:     "#00e5a0",
          borderWidth:     1,
          borderRadius:    4,
        }]},
        options: {
          responsive:          true,
          maintainAspectRatio: true,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              min:   0,
              max:   100,
              ticks: { callback: v => v + "%" },
              grid:  { color: CHART_DEFAULTS.gridColor },
            },
            x: { grid: { color: CHART_DEFAULTS.gridColor } },
          },
        },
      }
    );

    _activityChart = new Chart(
      document.getElementById("activity-chart"),
      {
        type: "doughnut",
        data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
        options: {
          responsive:          true,
          maintainAspectRatio: true,
          cutout:              "55%",
          plugins: {
            legend: {
              position:  "bottom",
              labels:    { boxWidth: 12, padding: 10, font: { size: 11 } },
            },
          },
        },
      }
    );
  }

  function updateUtil(utilData) {
    if (!utilData || !utilData.length) return;
    _utilChart.data.labels   = utilData.map(u => u.equipment_id);
    _utilChart.data.datasets[0].data = utilData.map(u => u.utilization_pct);
    _utilChart.update("none");
  }

  function updateActivity(detections) {
    if (!detections || !detections.length) return;

    const counts = {};
    detections.forEach(d => {
      counts[d.activity] = (counts[d.activity] || 0) + 1;
    });

    const labels = Object.keys(counts);
    const data   = Object.values(counts);
    const colors = labels.map(l => ACTIVITY_COLORS[l] || "#4a9fd4");

    _activityChart.data.labels                        = labels;
    _activityChart.data.datasets[0].data              = data;
    _activityChart.data.datasets[0].backgroundColor   = colors;
    _activityChart.update("none");
  }

  return { init, updateUtil, updateActivity };
})();
