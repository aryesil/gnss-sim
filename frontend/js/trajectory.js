// frontend/trajectory.js
let _tbMap = null, _tbMarkers = [], _tbWaypoints = [];
let _tbLine = null;

function _tbRedrawTable() {
  const t = document.getElementById('trajectory-table');
  t.innerHTML = '<tr><th>#</th><th>lat</th><th>lon</th><th>alt</th><th>speed m/s</th><th>accel m/s^2</th></tr>' +
    _tbWaypoints.map((w, i) => `<tr>
      <td>${i}</td>
      <td><input data-i="${i}" data-f="lat" value="${w.lat.toFixed(6)}" size="10"></td>
      <td><input data-i="${i}" data-f="lon" value="${w.lon.toFixed(6)}" size="10"></td>
      <td><input data-i="${i}" data-f="alt" value="${w.alt.toFixed(1)}" size="6"></td>
      <td><input data-i="${i}" data-f="speed" value="${w.speed}" size="5"></td>
      <td><input data-i="${i}" data-f="accel" value="${w.accel}" size="5"></td>
    </tr>`).join('');
  t.querySelectorAll('input').forEach(inp => {
    inp.onchange = () => {
      _tbWaypoints[Number(inp.dataset.i)][inp.dataset.f] = Number(inp.value);
      _tbRedrawMap();
    };
  });
  document.getElementById('tb-apply').disabled = _tbWaypoints.length < 2;
}

function _tbRedrawMap() {
  _tbMarkers.forEach(m => _tbMap.removeLayer(m));
  _tbMarkers = [];
  const latlngs = _tbWaypoints.map(w => [w.lat, w.lon]);
  _tbWaypoints.forEach((w, i) => {
    const m = L.marker([w.lat, w.lon], { draggable: true }).addTo(_tbMap)
      .bindTooltip(String(i), { permanent: true });
    m.on('dragend', () => {
      const ll = m.getLatLng();
      _tbWaypoints[i].lat = ll.lat; _tbWaypoints[i].lon = ll.lng;
      _tbRedrawTable(); _tbRedrawMap();
    });
    _tbMarkers.push(m);
  });
  if (_tbLine) _tbMap.removeLayer(_tbLine);
  _tbLine = L.polyline(latlngs, { color: 'red' }).addTo(_tbMap);
}

window.addEventListener('DOMContentLoaded', () => {
  _tbMap = window._tbMap = L.map('trajectory-map').setView([52.0, 19.0], 6);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(_tbMap);
  _tbMap.on('click', (ev) => {
    _tbWaypoints.push({ lat: ev.latlng.lat, lon: ev.latlng.lng, alt: 100, speed: 5.0, accel: 1.0 });
    _tbRedrawTable(); _tbRedrawMap();
  });

  document.getElementById('tb-new').onclick = () => {
    _tbWaypoints = []; _tbRedrawTable(); _tbRedrawMap();
  };

  document.getElementById('tb-save').onclick = async () => {
    const name = prompt('Save trajectory as:');
    if (!name) return;
    const r = await fetch('/api/trajectory/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, waypoints: _tbWaypoints }),
    });
    if (!r.ok) {
      const d = await r.json();
      if (window.logLine) logLine('trajectory save failed: ' + (d.detail || JSON.stringify(d)), 'error');
      alert('save failed: ' + (d.detail || JSON.stringify(d)));
    }
  };

  function _escHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  document.getElementById('tb-open').onclick = async () => {
    const r = await fetch('/api/trajectory/list');
    const d = await r.json();
    const sel = document.getElementById('tb-open-list');
    sel.innerHTML = '<option value="">-- select --</option>' +
      (d.names || []).map(n => `<option value="${_escHtml(n)}">${_escHtml(n)}</option>`).join('');
  };

  document.getElementById('tb-open-list').onchange = async (ev) => {
    const name = ev.target.value;
    if (!name) return;
    const r = await fetch(`/api/trajectory/load?name=${encodeURIComponent(name)}`);
    const d = await r.json();
    // Hata yanıtında (404 vb.) d.waypoints yoktur; undefined atanırsa tablo
    // çizimi (.map) patlar. Dizi biçimli [lat,lon,alt] kayıtlar da nesneye
    // çevrilir, çünkü tablo w.lat/w.lon alanlarını okur.
    if (!r.ok || !Array.isArray(d.waypoints)) {
      if (window.logLine) logLine('trajectory load failed: ' + (d.detail || JSON.stringify(d)), 'error');
      return;
    }
    _tbWaypoints = d.waypoints.map(w => Array.isArray(w)
      ? { lat: w[0], lon: w[1], alt: w[2], speed: w[3] ?? 5.0, accel: w[4] ?? 1.0 }
      : w);
    _tbRedrawTable(); _tbRedrawMap();
  };

  document.getElementById('tb-apply').onclick = () => {
    const activeCards = Array.from(document.querySelectorAll('.channel-card.active, .channel-card.selected'));
    const targetCards = activeCards.length > 0
      ? activeCards
      : Array.from(document.querySelectorAll('.channel-card'));
    if (!targetCards.length) return alert('add a channel first');
    const route = _tbWaypoints.map(w => Array.isArray(w) ? [w[0], w[1], w[2]] : [w.lat, w.lon, w.alt]);
    targetCards.forEach(card => {
      const st = typeof channelState === 'function' ? channelState(card.id) : null;
      if (st) st.route = route.map(pt => [pt[0], pt[1], pt[2]]);
    });
    alert(`Applied ${_tbWaypoints.length} waypoints to ${targetCards.map(c => c.id).join(', ')}`);
  };
});
