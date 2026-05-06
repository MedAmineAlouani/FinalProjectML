/**
 * Frontend logic for the Flange-Invariant Acoustic Bolt-Looseness Detector.
 *
 * Responsibilities:
 *   - Flange selection
 *   - Live mic recording via MediaRecorder + audio file upload
 *   - POST /api/predict with the audio blob and chosen flange
 *   - Render waveform + envelope + hit markers on a canvas
 *   - Render per-hit cards, averaged probabilities, and final badge
 */

(() => {
  // ------------------------------------------------------------------ //
  // Element shortcuts
  // ------------------------------------------------------------------ //
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const flangeButtons = $$('.flange-btn');
  const btnRecord = $('#btn-record');
  const btnStop   = $('#btn-stop');
  const btnPredict = $('#btn-predict');
  const fileInput  = $('#file-input');
  const fileNameEl = $('#file-name');
  const recStatus  = $('#record-status');
  const recTimer   = $('#record-timer');
  const healthPill = $('#health-pill');

  const finalCard   = $('#final-card');
  const finalBadge  = $('#final-badge');
  const finalConf   = $('#final-conf');
  const finalConfPill = $('#final-conf-pill');
  const finalMeta   = $('#final-meta');
  const probBars    = $('#prob-bars');
  const explainStrip = $('#explain-strip');
  const warningsBox = $('#warnings');
  const waveCard    = $('#wave-card');
  const hitsCard    = $('#hits-card');
  const hitGallery  = $('#hit-gallery');
  const hitCountPill = $('#hit-count-pill');
  const emptyState  = $('#empty-state');
  const waveCanvas  = $('#wave-canvas');

  // ------------------------------------------------------------------ //
  // State
  // ------------------------------------------------------------------ //
  const state = {
    flangeId:    null,
    audioBlob:   null,
    audioName:   null,
    mediaRec:    null,
    mediaStream: null,
    chunks:      [],
    recStart:    0,
    recTimerId:  null,
    busy:        false,
  };

  const CLASSES = [0, 25, 50];

  // ------------------------------------------------------------------ //
  // Health probe
  // ------------------------------------------------------------------ //
  async function checkHealth() {
    try {
      const r = await fetch('/api/health');
      const j = await r.json();
      if (j.model_loaded) {
        const acc = j.model?.lofo_calibrated_hit_accuracy;
        healthPill.classList.remove('pill-cyan', 'pill-red');
        healthPill.classList.add('pill-green');
        healthPill.innerHTML =
          `<span class="w-1.5 h-1.5 rounded-full bg-accent-green inline-block"></span>` +
          `model ready${acc != null ? ` · LOFO ${(acc * 100).toFixed(1)}%` : ''}`;
      } else {
        healthPill.classList.remove('pill-cyan', 'pill-green');
        healthPill.classList.add('pill-red');
        healthPill.innerHTML =
          `<span class="w-1.5 h-1.5 rounded-full bg-accent-red inline-block"></span>` +
          `model not loaded`;
      }
    } catch (e) {
      healthPill.classList.remove('pill-cyan', 'pill-green');
      healthPill.classList.add('pill-red');
      healthPill.innerHTML =
        `<span class="w-1.5 h-1.5 rounded-full bg-accent-red inline-block"></span>` +
        `backend offline`;
    }
  }
  checkHealth();

  // ------------------------------------------------------------------ //
  // Flange selection
  // ------------------------------------------------------------------ //
  flangeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      flangeButtons.forEach((b) => {
        b.classList.remove('bg-accent-cyan/15', 'border-accent-cyan/60', 'text-accent-cyan');
        b.classList.add('metallic', 'text-white');
      });
      btn.classList.remove('metallic', 'text-white');
      btn.classList.add('bg-accent-cyan/15', 'border-accent-cyan/60', 'text-accent-cyan');
      state.flangeId = parseInt(btn.dataset.flange, 10);
      refreshPredictBtn();
    });
  });

  // ------------------------------------------------------------------ //
  // File upload
  // ------------------------------------------------------------------ //
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    state.audioBlob = f;
    state.audioName = f.name;
    fileNameEl.textContent = `Selected: ${f.name} · ${(f.size / 1024).toFixed(1)} KB`;
    refreshPredictBtn();
  });

  // ------------------------------------------------------------------ //
  // Live recording
  // ------------------------------------------------------------------ //
  btnRecord.addEventListener('click', async () => {
    if (state.mediaRec) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate:   48000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl:  false,
        },
      });
      state.mediaStream = stream;
      const mime = pickSupportedMime();
      state.mediaRec = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      state.chunks = [];
      state.mediaRec.ondataavailable = (e) => { if (e.data.size > 0) state.chunks.push(e.data); };
      state.mediaRec.onstop = () => {
        const blob = new Blob(state.chunks, { type: state.mediaRec.mimeType || 'audio/webm' });
        state.audioBlob = blob;
        state.audioName = `recording${extFromMime(blob.type)}`;
        fileNameEl.textContent = `Recorded: ${state.audioName} · ${(blob.size / 1024).toFixed(1)} KB`;
        cleanupRecording();
        refreshPredictBtn();
      };
      state.mediaRec.start();
      state.recStart = Date.now();
      btnRecord.disabled = true;
      btnStop.disabled = false;
      recStatus.classList.remove('hidden');
      recStatus.classList.add('inline-flex');
      state.recTimerId = setInterval(updateTimer, 100);
    } catch (e) {
      showWarnings([`Microphone access denied or unavailable: ${e.message || e}`]);
    }
  });

  btnStop.addEventListener('click', () => {
    if (state.mediaRec && state.mediaRec.state !== 'inactive') {
      state.mediaRec.stop();
    }
  });

  function cleanupRecording() {
    if (state.recTimerId) { clearInterval(state.recTimerId); state.recTimerId = null; }
    if (state.mediaStream) { state.mediaStream.getTracks().forEach((t) => t.stop()); state.mediaStream = null; }
    state.mediaRec = null;
    btnRecord.disabled = false;
    btnStop.disabled = true;
    recStatus.classList.add('hidden');
    recStatus.classList.remove('inline-flex');
  }

  function updateTimer() {
    const ms = Date.now() - state.recStart;
    const s = Math.floor(ms / 1000);
    const mm = String(Math.floor(s / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    recTimer.textContent = `${mm}:${ss}`;
  }

  function pickSupportedMime() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus',
    ];
    for (const m of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    }
    return null;
  }

  function extFromMime(mime) {
    if (!mime) return '.webm';
    if (mime.includes('webm')) return '.webm';
    if (mime.includes('mp4'))  return '.m4a';
    if (mime.includes('ogg'))  return '.ogg';
    if (mime.includes('wav'))  return '.wav';
    return '.webm';
  }

  // ------------------------------------------------------------------ //
  // Predict
  // ------------------------------------------------------------------ //
  function refreshPredictBtn() {
    btnPredict.disabled = !(state.flangeId && state.audioBlob && !state.busy);
  }

  btnPredict.addEventListener('click', async () => {
    if (state.busy || !state.audioBlob || !state.flangeId) return;
    state.busy = true;
    btnPredict.disabled = true;
    btnPredict.textContent = 'Analyzing audio…';
    showSkeleton();

    const fd = new FormData();
    fd.append('audio', state.audioBlob, state.audioName || 'audio.webm');
    fd.append('flange_id', String(state.flangeId));

    try {
      const r = await fetch('/api/predict', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) {
        showWarnings([j.detail || `Server error (${r.status})`]);
        clearResults();
      } else {
        renderResults(j);
      }
    } catch (e) {
      showWarnings([`Request failed: ${e.message || e}`]);
      clearResults();
    } finally {
      state.busy = false;
      btnPredict.textContent = 'Predict Torque';
      refreshPredictBtn();
    }
  });

  function showSkeleton() {
    emptyState.classList.add('hidden');
    waveCard.classList.remove('hidden');
    hitsCard.classList.add('hidden');
    finalCard.classList.add('hidden');
    warningsBox.classList.add('hidden');
    const ctx = waveCanvas.getContext('2d');
    fitCanvas(waveCanvas);
    ctx.fillStyle = 'rgba(155,166,185,0.07)';
    ctx.fillRect(0, 0, waveCanvas.width, waveCanvas.height);
  }

  function clearResults() {
    waveCard.classList.add('hidden');
    hitsCard.classList.add('hidden');
    finalCard.classList.add('hidden');
    emptyState.classList.remove('hidden');
  }

  // ------------------------------------------------------------------ //
  // Rendering
  // ------------------------------------------------------------------ //
  function renderResults(j) {
    emptyState.classList.add('hidden');

    showWarnings(j.warnings || []);

    // Waveform
    waveCard.classList.remove('hidden');
    hitCountPill.textContent = `${j.n_hits} hit${j.n_hits === 1 ? '' : 's'}`;
    drawWaveform(waveCanvas, j);

    if (!j.ok || !j.final_prediction) {
      hitsCard.classList.add('hidden');
      finalCard.classList.add('hidden');
      return;
    }

    // Per-hit gallery
    hitsCard.classList.remove('hidden');
    hitGallery.innerHTML = '';
    j.per_hit.forEach((hit) => hitGallery.appendChild(renderHitCard(hit)));

    // Final card
    finalCard.classList.remove('hidden');
    const fp = j.final_prediction;
    const torque = fp.torque_ftlbs;
    const conf = fp.confidence;
    const level = fp.confidence_level; // 'high' | 'medium' | 'low'

    finalBadge.textContent = `Predicted Torque: ${torque} ft-lbs`;
    finalBadge.className = 'mt-2 text-3xl sm:text-4xl font-extrabold tracking-tight badge-pulse ' + colorForLevel(level, 'text');
    finalCard.className = 'glass rounded-2xl p-5 sm:p-6 ' + shadowForLevel(level);

    finalConf.textContent = `${(conf * 100).toFixed(1)}%`;
    finalConfPill.innerHTML = `<span class="pill ${pillForLevel(level)}">${level} confidence</span>`;

    finalMeta.innerHTML = `
      Flange F${j.flange_id} · ${j.n_hits} hit${j.n_hits === 1 ? '' : 's'} · ${j.duration_sec.toFixed(2)} s · ${j.sample_rate} Hz
    `;

    // Probability bars (averaged across hits)
    probBars.innerHTML = '';
    CLASSES.forEach((c) => {
      const p = j.averaged_probabilities[String(c)] || 0;
      probBars.appendChild(renderProbBar(c, p, c === torque));
    });

    // Explanation strip
    explainStrip.innerHTML = `
      ${explainTile('Hits detected', `${j.n_hits}`)}
      ${explainTile('Aggregation', 'Soft-vote average of per-hit probabilities')}
      ${explainTile('Decision rule', 'argmax of averaged probabilities')}
    `;
  }

  function renderHitCard(hit) {
    const card = document.createElement('div');
    card.className = 'metallic rounded-xl p-3';
    const cls = hit.predicted_torque;
    const lvl = hit.confidence >= 0.7 ? 'high' : hit.confidence >= 0.5 ? 'medium' : 'low';

    card.innerHTML = `
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs font-mono text-steel-400">Hit #${hit.hit_id} · ${hit.time_sec != null ? hit.time_sec.toFixed(2) + 's' : ''}</div>
        <span class="pill ${pillForLevel(lvl)}">${cls} ft-lbs</span>
      </div>
      <div class="relative h-12 bg-steel-950/70 rounded-md overflow-hidden border border-steel-700">
        <canvas class="hit-canvas"></canvas>
      </div>
      <div class="mt-2 space-y-1.5">
        ${CLASSES.map((c) => {
          const p = hit.probabilities[String(c)] || 0;
          const isMax = c === cls;
          const barColor = isMax ? colorForLevel(lvl, 'bg') : 'bg-steel-600';
          return `
            <div>
              <div class="flex items-center justify-between text-[10px] font-mono text-steel-400">
                <span>${c} ft-lbs</span>
                <span class="text-steel-200">${(p * 100).toFixed(1)}%</span>
              </div>
              <div class="h-1.5 bg-steel-800 rounded-full overflow-hidden">
                <div class="prob-fill h-full ${barColor}" style="width:${(p * 100).toFixed(2)}%"></div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
    requestAnimationFrame(() => {
      const c = card.querySelector('.hit-canvas');
      fitCanvas(c);
      drawMiniWaveform(c, hit.waveform);
    });
    return card;
  }

  function renderProbBar(cls, p, isMax) {
    const div = document.createElement('div');
    const colorMain = isMax ? 'bg-gradient-to-r from-accent-green to-accent-cyan' : 'bg-steel-600';
    div.innerHTML = `
      <div class="flex items-center justify-between text-xs mb-1">
        <span class="font-mono ${isMax ? 'text-white font-semibold' : 'text-steel-400'}">${cls} ft-lbs</span>
        <span class="font-mono ${isMax ? 'text-white' : 'text-steel-300'}">${(p * 100).toFixed(1)}%</span>
      </div>
      <div class="h-2.5 rounded-full bg-steel-800 overflow-hidden">
        <div class="prob-fill h-full ${colorMain}" style="width:${(p * 100).toFixed(2)}%"></div>
      </div>
    `;
    return div;
  }

  function explainTile(label, value) {
    return `
      <div class="metallic rounded-xl px-3 py-2.5">
        <div class="text-[10px] uppercase tracking-wide text-steel-500 font-semibold">${label}</div>
        <div class="text-xs text-steel-200 mt-0.5">${value}</div>
      </div>
    `;
  }

  function showWarnings(list) {
    if (!list || list.length === 0) {
      warningsBox.classList.add('hidden');
      warningsBox.innerHTML = '';
      return;
    }
    warningsBox.classList.remove('hidden');
    warningsBox.innerHTML = list.map((w) => `
      <div class="glass rounded-xl px-4 py-3 border border-accent-amber/30 flex items-start gap-3">
        <svg class="w-4 h-4 text-accent-amber mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2 1 21h22L12 2Zm0 6 8 14H4l8-14Zm-1 4v4h2v-4h-2Zm0 6v2h2v-2h-2Z"/>
        </svg>
        <div class="text-sm text-steel-200">${escapeHtml(w)}</div>
      </div>
    `).join('');
  }

  // ------------------------------------------------------------------ //
  // Canvas rendering
  // ------------------------------------------------------------------ //
  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function drawMiniWaveform(canvas, waveObj) {
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    const vals = waveObj?.values || [];
    if (!vals.length) return;
    const w = rect.width, h = rect.height, mid = h / 2;
    const max = Math.max(0.01, ...vals.map(Math.abs));
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
      const x = (i / (vals.length - 1)) * w;
      const y = mid - (vals[i] / max) * mid * 0.95;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawWaveform(canvas, j) {
    const ctx = fitCanvas(canvas);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height, mid = h / 2;
    ctx.clearRect(0, 0, w, h);

    // background gridlines
    ctx.strokeStyle = 'rgba(155, 166, 185, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      const y = (h / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const wave = j.waveform?.values || [];
    const env  = j.envelope?.values || [];
    const dur  = j.duration_sec || 1;
    const peakTimes = j.hit_times_sec || [];

    // raw waveform (cyan)
    if (wave.length) {
      const max = Math.max(0.01, ...wave.map(Math.abs));
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < wave.length; i++) {
        const x = (i / (wave.length - 1)) * w;
        const y = mid - (wave[i] / max) * mid * 0.95;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // envelope (green) — sample-aware so it lines up with the trimmed signal
    if (env.length) {
      const max = Math.max(0.01, ...env.map(Math.abs));
      const off = j.envelope.offset_sec || 0;
      const len = j.envelope.length || env.length;
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.95)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < env.length; i++) {
        const tFrac = (i / (env.length - 1)) * (len / (j.sample_rate || 1));
        const xSec = off + tFrac;
        const x = (xSec / dur) * w;
        const y = mid - (env[i] / max) * mid * 0.85;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // peak markers (amber)
    ctx.fillStyle = 'rgba(245, 158, 11, 0.85)';
    ctx.strokeStyle = 'rgba(245, 158, 11, 1)';
    ctx.lineWidth = 1.5;
    peakTimes.forEach((t, i) => {
      const x = (t / dur) * w;
      ctx.beginPath();
      ctx.moveTo(x, 4); ctx.lineTo(x, h - 4); ctx.stroke();
      ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillText(`#${i + 1}`, x + 4, 14);
    });
  }

  // ------------------------------------------------------------------ //
  // Color helpers
  // ------------------------------------------------------------------ //
  function colorForLevel(level, kind = 'text') {
    const map = {
      high:   { text: 'text-accent-green', bg: 'bg-accent-green' },
      medium: { text: 'text-accent-amber', bg: 'bg-accent-amber' },
      low:    { text: 'text-accent-red',   bg: 'bg-accent-red' },
    };
    return (map[level] || map.medium)[kind];
  }
  function shadowForLevel(level) {
    return level === 'high'   ? 'shadow-glow-green'
         : level === 'medium' ? 'shadow-glow-amber'
                              : 'shadow-glow-red';
  }
  function pillForLevel(level) {
    return level === 'high'   ? 'pill-green'
         : level === 'medium' ? 'pill-amber'
                              : 'pill-red';
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Re-fit canvases on resize
  window.addEventListener('resize', () => {
    if (!waveCard.classList.contains('hidden')) {
      // re-render the last drawn waveform if available
      // (handled lazily — next predict/upload will redraw cleanly)
    }
  });
})();
