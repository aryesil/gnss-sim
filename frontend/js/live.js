// frontend/live.js
window._liveChannels = window._liveChannels || new Set();
function _updateRfBar() {
  const bar = document.getElementById('rf-status-bar');
  if (!bar) return;
  let textSpan = bar.querySelector('#rf-status-text');
  if (!textSpan) {
    Array.from(bar.childNodes).forEach(n => {
      if (n.nodeType === Node.TEXT_NODE) bar.removeChild(n);
    });
    textSpan = document.createElement('span');
    textSpan.id = 'rf-status-text';
    const btn = bar.querySelector('#btn-api-key');
    if (btn) bar.insertBefore(textSpan, btn);
    else bar.appendChild(textSpan);
  }
  if (window._liveChannels && window._liveChannels.size > 0) {
    textSpan.textContent = '● RF TRANSMITTING — SHIELDED TEST ENVIRONMENT ONLY';
    bar.className = 'rf-on';
  } else {
    textSpan.textContent = 'RF OUTPUT: OFF';
    bar.className = 'rf-off';
  }
}

window.enableLiveTabs = function (channelId, txSlot) {
  const id = channelId;
  channelState(id).txSlot = txSlot;
  window._liveChannels.add(id); _updateRfBar();
  document.getElementById(`${id}-live-hint`).hidden = true;
  document.getElementById(`${id}-jog-controls`).hidden = false;
  document.getElementById(`${id}-time-controls`).hidden = false;

  document.querySelectorAll(`#${id}-jog-controls button[data-dir]`).forEach(btn => {
    btn.onclick = async () => {
      const distance_m = Number(document.getElementById(`${id}-jog-step`).value);
      const r = await fetch('/api/live/jog', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot: txSlot, direction: btn.dataset.dir, distance_m }),
      });
      const d = await r.json();
      if (r.ok) document.getElementById(`${id}-live-llh`).textContent = JSON.stringify(d.llh);
      else if (window.logLine) logLine('live jog failed: ' + JSON.stringify(d), 'error');
    };
  });

  document.querySelectorAll(`#${id}-time-controls button[data-field]`).forEach(btn => {
    btn.onclick = async () => {
      const r = await fetch('/api/live/time_shift', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot: txSlot, field: btn.dataset.field, delta: Number(btn.dataset.delta) }),
      });
      const d = await r.json();
      if (!r.ok) { if (window.logLine) logLine('live time_shift failed: ' + JSON.stringify(d), 'error'); return; }
      const v = d.time_offset_s;
      document.getElementById(`${id}-time-offset`).value = (v >= 0 ? '+' : '') + v;
    };
  });
};

window.disableLiveTabs = function (channelId) {
  window._liveChannels.delete(channelId); _updateRfBar();
  const hint = document.getElementById(`${channelId}-live-hint`);
  if (hint) hint.hidden = false;
  const jog = document.getElementById(`${channelId}-jog-controls`);
  if (jog) jog.hidden = true;
  const time = document.getElementById(`${channelId}-time-controls`);
  if (time) time.hidden = true;
  const st = typeof channelState === 'function' ? channelState(channelId) : null;
  if (st) st.txSlot = null;
};
