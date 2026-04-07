/**
 * js/stream.js
 * Polls /stream/latest-frame and renders JPEG to canvas.
 */

window.Stream = (() => {
  const POLL_MS    = 300;   // poll every 300ms for smooth feel
  let   _timer     = null;
  let   _canvas    = null;
  let   _ctx       = null;
  let   _noSignal  = null;
  let   _counter   = null;

  function init() {
    _canvas   = document.getElementById("feed-canvas");
    _ctx      = _canvas.getContext("2d");
    _noSignal = document.getElementById("no-signal");
    _counter  = document.getElementById("frame-counter");
  }

  function start() {
    if (_timer) return;
    _timer = setInterval(_poll, POLL_MS);
  }

  function stop() {
    clearInterval(_timer);
    _timer = null;
  }

  async function _poll() {
    const data = await API.latestFrame();
    if (!data || !data.jpeg_b64) {
      _noSignal.style.display = "flex";
      return;
    }
    _noSignal.style.display = "none";
    if (_counter) _counter.textContent = `FRAME ${data.frame_index}`;

    const img    = new Image();
    img.onload   = () => {
      _canvas.width  = img.naturalWidth;
      _canvas.height = img.naturalHeight;
      _ctx.drawImage(img, 0, 0);
    };
    img.src = "data:image/jpeg;base64," + data.jpeg_b64;
  }

  return { init, start, stop };
})();
