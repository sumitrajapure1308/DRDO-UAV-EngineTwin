/**
 * DRDO MALE UAV Aero Piston Engine Digital Twin - Main GCS Dashboard Application
 * Handles real-time WebSocket telemetry ingestion, DOM updates, SVG engine animation,
 * fault injection dispatch, mission report generation, and mission replay controls.
 */

let socket = null;
let currentPacket = null;
let firingIndex = 0;
const FIRING_ORDER = [1, 4, 3, 2];

// Strip chart history data points (canvas rendering)
const chartHistory = {
  timestamps: [],
  rpm: [],
  map: [],
  cht_avg: [],
  egt_avg: [],
  oil_p: []
};
const MAX_CHART_POINTS = 60;

// Defense Voice and Acoustic Annunciator System
let audioEnabled = true;
let lastAnnouncedFault = null;
let lastAnnouncedTime = 0;
let audioContext = null;

function getAudioContext() {
  if (!audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioContext = new AudioContextClass();
    }
  }
  if (audioContext && audioContext.state === 'suspended') {
    audioContext.resume();
  }
  return audioContext;
}

function playMasterAnnunciatorTone(isCritical) {
  if (!audioEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (isCritical) {
      // Two-tone military warning alert: 900Hz -> 1350Hz
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(900, ctx.currentTime);
      osc.frequency.setValueAtTime(1350, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.35);
    } else {
      // Single soft caution chime: 750Hz
      osc.type = 'sine';
      osc.frequency.setValueAtTime(750, ctx.currentTime);
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.25);
    }
  } catch (e) {
    // Suppress uninitialized gesture errors
  }
}

function speakVoiceAnnunciation(text) {
  if (!audioEnabled || !('speechSynthesis' in window)) return;
  const now = Date.now();
  if (now - lastAnnouncedTime < 6000 && lastAnnouncedFault === text) return;
  lastAnnouncedTime = now;
  lastAnnouncedFault = text;

  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.volume = 0.9;
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    console.warn('Speech synthesis unavailable', e);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  initWebSocket();
  setupUIEventListeners();
  populateRecordingsList();
  startEngineAnimationLoop();
});

function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    document.getElementById('connection-status-dot').style.backgroundColor = '#00ff88';
    document.getElementById('connection-status-text').innerText = '10 Hz SYNCED';
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'TELEMETRY_UPDATE') {
        currentPacket = data;
        updateDashboard(data);
      }
    } catch (e) {
      console.error('Error parsing telemetry payload', e);
    }
  };

  socket.onclose = () => {
    document.getElementById('connection-status-dot').style.backgroundColor = '#ff3344';
    document.getElementById('connection-status-text').innerText = 'DISCONNECTED (RETRYING...)';
    setTimeout(initWebSocket, 2000);
  };
}

function updateDashboard(pkt) {
  const t = pkt.telemetry;
  const twin = pkt.digital_twin;
  const h = pkt.health;
  const prog = pkt.prognosis;
  const alerts = pkt.active_alerts;
  const xai = pkt.xai;
  const adv = pkt.advisories;

  // 1. Header & Flight State Pills
  document.getElementById('val-flight-time').innerText = formatTime(t.mission_time_s);
  document.getElementById('val-mission-phase').innerText = t.mission_phase;
  document.getElementById('val-alt').innerText = `${Math.round(t.altitude_ft)} FT`;
  document.getElementById('val-airspeed').innerText = `${Math.round(t.airspeed_kts)} KTS`;
  document.getElementById('val-ambient-temp').innerText = `${t.ambient_temp_c.toFixed(1)} °C`;

  // 2. Primary Telemetry Instruments
  document.getElementById('dial-rpm').innerText = Math.round(t.rpm);
  document.getElementById('dial-map').innerText = t.map_inhg.toFixed(1);
  document.getElementById('dial-power').innerText = `${twin.twin_power_brake_hp.toFixed(1)} HP`;
  document.getElementById('dial-torque').innerText = `${(twin.twin_power_brake_kw * 1000 / (2 * Math.PI * (t.rpm / 60))).toFixed(1)} N·m`;
  document.getElementById('dial-fuelflow').innerText = t.fuel_flow_lph.toFixed(1);
  document.getElementById('dial-fuelpress').innerText = `${t.fuel_pressure_bar.toFixed(2)} BAR`;
  document.getElementById('dial-oilpress').innerText = t.oil_pressure_bar.toFixed(2);
  document.getElementById('dial-oiltemp').innerText = `${t.oil_temp_c.toFixed(1)} °C`;
  document.getElementById('dial-coolant').innerText = `${t.coolant_temp_c.toFixed(1)} °C`;
  document.getElementById('dial-bsfc').innerText = `${twin.twin_bsfc_g_kwh.toFixed(0)} g/kWh`;
  document.getElementById('dial-busvolt').innerText = `${t.bus_voltage_v.toFixed(1)} V`;
  document.getElementById('dial-busamps').innerText = `${t.alternator_current_a.toFixed(1)} A`;

  // 3. Cylinder Status Cards
  updateCylinderCard(1, t.cht_1_c, t.egt_1_c, twin.residual_cht_1_c, twin.residual_egt_1_c, alerts);
  updateCylinderCard(2, t.cht_2_c, t.egt_2_c, twin.residual_cht_2_c, twin.residual_egt_2_c, alerts);
  updateCylinderCard(3, t.cht_3_c, t.egt_3_c, twin.residual_cht_3_c, twin.residual_egt_3_c, alerts);
  updateCylinderCard(4, t.cht_4_c, t.egt_4_c, twin.residual_cht_4_c, twin.residual_egt_4_c, alerts);
  document.getElementById('val-cht-spread').innerText = `ΔCHT: ${t.cht_spread_c.toFixed(1)} °C`;
  document.getElementById('val-egt-spread').innerText = `ΔEGT: ${t.egt_spread_c.toFixed(1)} °C`;

  // 4. Digital Twin Synchronizer & Residual Matrix
  document.getElementById('sync-meas-map').innerText = `${t.map_inhg.toFixed(1)} inHg`;
  document.getElementById('sync-twin-map').innerText = `${twin.twin_map_inhg.toFixed(1)} inHg`;
  setResidualCell('sync-res-map', twin.residual_map_pa / 3386.39, 'inHg', 1.5, 3.0);

  document.getElementById('sync-meas-cht').innerText = `${t.cht_avg_c.toFixed(1)} °C`;
  document.getElementById('sync-twin-cht').innerText = `${twin.twin_cht_1_c.toFixed(1)} °C`;
  setResidualCell('sync-res-cht', (twin.residual_cht_1_c + twin.residual_cht_2_c + twin.residual_cht_3_c + twin.residual_cht_4_c)/4, '°C', 8.0, 15.0);

  document.getElementById('sync-meas-egt').innerText = `${t.egt_avg_c.toFixed(0)} °C`;
  document.getElementById('sync-twin-egt').innerText = `${twin.twin_egt_1_c.toFixed(0)} °C`;
  setResidualCell('sync-res-egt', (twin.residual_egt_1_c + twin.residual_egt_2_c + twin.residual_egt_3_c + twin.residual_egt_4_c)/4, '°C', 35.0, 70.0);

  document.getElementById('sync-meas-ff').innerText = `${t.fuel_flow_lph.toFixed(1)} L/h`;
  document.getElementById('sync-twin-ff').innerText = `${twin.twin_fuel_flow_lph.toFixed(1)} L/h`;
  setResidualCell('sync-res-ff', twin.residual_fuel_flow_lph, 'L/h', 1.5, 3.0);

  document.getElementById('sync-meas-oilp').innerText = `${t.oil_pressure_bar.toFixed(2)} bar`;
  document.getElementById('sync-twin-oilp').innerText = `${twin.twin_oil_pressure_bar.toFixed(2)} bar`;
  setResidualCell('sync-res-oilp', twin.residual_oil_pressure_bar, 'bar', 0.5, 1.0);

  document.getElementById('val-res-norm').innerText = twin.composite_residual_norm.toFixed(2);

  // 5. Health Scores & Composite Engine Health Index (EHI)
  const ehiPct = (h.overall_health_index * 100).toFixed(1);
  const ehiEl = document.getElementById('val-ehi');
  ehiEl.innerText = `${ehiPct}%`;
  ehiEl.style.color = ehiPct > 85 ? '#00ff88' : ehiPct > 65 ? '#ffaa00' : '#ff3344';

  setProgressBar('bar-thermal', h.subsystem_scores.thermal_cht);
  setProgressBar('bar-exhaust', h.subsystem_scores.exhaust_gas_path);
  setProgressBar('bar-fuel', h.subsystem_scores.fuel_injection);
  setProgressBar('bar-lube', h.subsystem_scores.lubrication_system);
  setProgressBar('bar-turbo', h.subsystem_scores.turbo_charge);
  setProgressBar('bar-cool', h.subsystem_scores.cooling_system);
  setProgressBar('bar-comb', h.subsystem_scores.combustion_stability);
  setProgressBar('bar-elec', h.subsystem_scores.electrical_bus);

  // 6. Prognostics & RUL Section
  document.getElementById('rul-p50').innerText = `${prog.rul_hours_median.toFixed(1)} HRS`;
  document.getElementById('rul-confidence-bounds').innerText = `90% CI: [${prog.rul_hours_p10.toFixed(1)} - ${prog.rul_hours_p90.toFixed(1)}] HRS`;
  document.getElementById('rul-limiting-subsystem').innerText = prog.primary_limiting_subsystem;
  document.getElementById('rul-trend-badge').innerText = prog.degradation_trend;
  document.getElementById('val-mission-prob').innerText = `${prog.mission_completion_probability.toFixed(1)}%`;
  setProgressBar('bar-mission-prob', prog.mission_completion_probability);

  // 7. Vibration & Order Spectrum
  document.getElementById('vib-rms').innerText = `${t.vibration_rms_g.toFixed(2)} g`;
  document.getElementById('vib-peak').innerText = `${t.vibration_peak_g.toFixed(2)} g`;
  document.getElementById('vib-crest').innerText = t.crest_factor.toFixed(2);
  
  if (t.spectral_bins && t.spectral_bins.length >= 4) {
    document.getElementById('spec-05x').style.height = `${Math.min(100, t.spectral_bins[0].amplitude_g * 50)}%`;
    document.getElementById('spec-10x').style.height = `${Math.min(100, t.spectral_bins[1].amplitude_g * 45)}%`;
    document.getElementById('spec-20x').style.height = `${Math.min(100, t.spectral_bins[2].amplitude_g * 40)}%`;
    document.getElementById('spec-hf').style.height = `${Math.min(100, t.spectral_bins[4].amplitude_g * 40)}%`;
  }

  // 8. AI/ML Anomaly & Explainable AI (XAI)
  const aiScore = pkt.ai_anomaly ? pkt.ai_anomaly.anomaly_score : 0.0;
  document.getElementById('ai-anomaly-score').innerText = aiScore.toFixed(3);
  setProgressBar('bar-ai-anomaly', aiScore * 100);

  // XAI Drivers
  const xaiContainer = document.getElementById('xai-drivers-list');
  xaiContainer.innerHTML = '';
  if (xai && xai.top_drivers && xai.top_drivers.length > 0) {
    xai.top_drivers.forEach(d => {
      const row = document.createElement('div');
      row.className = 'xai-bar-row';
      row.innerHTML = `
        <div class="xai-bar-header">
          <span>${d.feature}</span>
          <span>${d.contribution_pct}%</span>
        </div>
        <div class="progress-track" style="height: 5px;">
          <div class="progress-fill warn" style="width: ${d.contribution_pct}%;"></div>
        </div>
      `;
      xaiContainer.appendChild(row);
    });
  } else {
    xaiContainer.innerHTML = '<div style="color: #546580; font-size: 11px;">Nominal operation. No anomalous deviation drivers detected.</div>';
  }

  // 9. Annunciator Alerts & Master Caution Handling
  const alertFeed = document.getElementById('alert-feed');
  const masterCautionPill = document.getElementById('master-caution-pill');
  const masterCautionText = document.getElementById('master-caution-text');
  alertFeed.innerHTML = '';

  if (alerts && alerts.length > 0) {
    let hasCritical = false;
    let hasWarning = false;
    let primaryAlert = alerts[0];

    alerts.forEach(a => {
      if (a.severity === 'CRITICAL') {
        hasCritical = true;
        primaryAlert = a;
      } else if (a.severity === 'WARNING' && !hasCritical) {
        hasWarning = true;
        primaryAlert = a;
      }
      const div = document.createElement('div');
      div.className = `alert-item ${a.severity}`;
      div.innerHTML = `
        <div class="alert-item-header">
          <span style="color: ${a.severity === 'CRITICAL' ? '#ff3344' : a.severity === 'WARNING' ? '#ffaa00' : '#00d2ff'}">[${a.fault_code}] ${a.title}</span>
          <span style="font-size: 10px;">${a.severity}</span>
        </div>
        <div class="alert-item-body">${a.description}</div>
      `;
      alertFeed.appendChild(div);
    });

    if (masterCautionPill) {
      masterCautionPill.style.display = 'inline-flex';
      if (hasCritical) {
        masterCautionPill.style.background = 'rgba(255, 51, 68, 0.25)';
        masterCautionPill.style.borderColor = 'var(--drdo-red)';
        masterCautionText.innerText = `CRITICAL: ${primaryAlert.title.toUpperCase()}`;
        masterCautionText.style.color = 'var(--drdo-red)';
        playMasterAnnunciatorTone(true);
        speakVoiceAnnunciation(`Warning: ${primaryAlert.title}`);
      } else if (hasWarning) {
        masterCautionPill.style.background = 'rgba(255, 170, 0, 0.25)';
        masterCautionPill.style.borderColor = 'var(--drdo-amber)';
        masterCautionText.innerText = `CAUTION: ${primaryAlert.title.toUpperCase()}`;
        masterCautionText.style.color = 'var(--drdo-amber)';
        playMasterAnnunciatorTone(false);
        speakVoiceAnnunciation(`Caution: ${primaryAlert.title}`);
      } else {
        masterCautionPill.style.background = 'rgba(0, 210, 255, 0.2)';
        masterCautionPill.style.borderColor = 'var(--drdo-cyan)';
        masterCautionText.innerText = `ADVISORY: ${primaryAlert.title.toUpperCase()}`;
        masterCautionText.style.color = 'var(--drdo-cyan)';
      }
    }
  } else {
    if (masterCautionPill) {
      masterCautionPill.style.display = 'none';
    }
    lastAnnouncedFault = null;
    alertFeed.innerHTML = '<div style="color: #00ff88; font-size: 11px; padding: 6px;">[ALL SYSTEMS NOMINAL] No active caution or warning alerts.</div>';
  }

  // 10. Prescriptive Maintenance Advisory List
  const advContainer = document.getElementById('advisory-container');
  advContainer.innerHTML = '';
  if (adv && adv.length > 0) {
    adv.slice(0, 3).forEach(item => {
      const div = document.createElement('div');
      div.style.background = '#0d131f';
      div.style.borderLeft = `3px solid ${item.priority === 'URGENT_AOG' ? '#ff3344' : item.priority === 'PRE_FLIGHT_ACTION' ? '#ffaa00' : '#38bdf8'}`;
      div.style.padding = '6px 8px';
      div.style.marginBottom = '6px';
      div.style.borderRadius = '3px';
      div.innerHTML = `
        <div style="font-weight: 700; color: #e6edf3; font-size: 11px;">${item.component} [${item.priority}]</div>
        <div style="color: #8b9bb4; font-size: 10px; margin: 2px 0;">${item.findings}</div>
        <div style="color: #38bdf8; font-size: 10px; font-style: italic;">Action: ${item.prescribed_action}</div>
      `;
      advContainer.appendChild(div);
    });
  }

  // Update strip chart
  appendChartData(t.mission_time_s, t.rpm, t.map_inhg, t.cht_avg_c, t.egt_avg_c, t.oil_pressure_bar);

  // 12. EKF Sensor Fusion & Indicator P-V Cycle
  if (pkt.sensor_fusion) {
    const ekf = pkt.sensor_fusion;
    const pmaxEl = document.getElementById('ekf-pmax');
    if (pmaxEl) pmaxEl.innerText = `${ekf.estimated_p_max_bar.toFixed(1)} ± ${ekf.p_max_confidence_pm_bar.toFixed(1)} bar`;
    const tcoreEl = document.getElementById('ekf-tcore');
    if (tcoreEl) tcoreEl.innerText = `${ekf.estimated_t_core_k.toFixed(0)} K`;
    const mtrapEl = document.getElementById('ekf-mtrapped');
    if (mtrapEl) mtrapEl.innerText = `${ekf.estimated_trapped_mass_mg.toFixed(1)} mg/cyl`;
    const consisEl = document.getElementById('ekf-consistency');
    if (consisEl) {
      consisEl.innerText = `D_M = ${ekf.sensor_fusion_mahalanobis_distance.toFixed(2)} [${ekf.sensor_consistency_status}]`;
      consisEl.style.color = ekf.sensor_consistency_status === 'CONSISTENT' ? 'var(--drdo-green)' : 'var(--drdo-red)';
    }
  }

  if (pkt.pv_cycle) {
    drawPVCycle(pkt.pv_cycle);
  }

  // 13. Universal Engine Performance Map & Compressor Margin
  drawEnginePerformanceMap(t.rpm, twin.twin_bmep_bar, twin.twin_bsfc_g_kwh, pkt.compressor_point);

  // 13b. Tactical Airfield Reachability & Dynamic Gliding Map
  drawTacticalMap(t, pkt.reliability_envelope);

  // 13c. Dual FADEC & Analytical Virtual Sensor Redundancy Status
  if (pkt.fadec_state) {
    const f = pkt.fadec_state;
    const virtEl = document.getElementById('fadec-virtual-text');
    const virtDot = document.getElementById('fadec-virtual-dot');
    if (virtEl && virtDot) {
      if (f.virtual_substitutions && f.virtual_substitutions.length > 0) {
        virtEl.innerText = `FAIL-OP: [${f.virtual_substitutions.join(', ').toUpperCase()} SYNTHESIZED]`;
        virtEl.style.color = 'var(--drdo-amber)';
        virtDot.style.background = 'var(--drdo-amber)';
        virtDot.className = 'fadec-dot warning';
      } else {
        virtEl.innerText = 'ALL SENSORS VALID';
        virtEl.style.color = '#38bdf8';
        virtDot.style.background = '#3b82f6';
        virtDot.className = 'fadec-dot';
      }
    }
  }

  // 14. Mission Reliability Enhancement HUD
  if (pkt.reliability_envelope) {
    const rel = pkt.reliability_envelope;
    const badge = document.getElementById('hud-directive-badge');
    if (badge) {
      badge.innerText = rel.mission_directive;
      if (rel.mission_directive === 'GO_FULL_MISSION') {
        badge.style.color = 'var(--drdo-green)';
        badge.style.borderColor = 'var(--drdo-green)';
        badge.style.background = 'rgba(0,255,136,0.15)';
      } else if (rel.mission_directive.includes('RESTRICTION')) {
        badge.style.color = 'var(--drdo-amber)';
        badge.style.borderColor = 'var(--drdo-amber)';
        badge.style.background = 'rgba(255,170,0,0.15)';
      } else {
        badge.style.color = 'var(--drdo-red)';
        badge.style.borderColor = 'var(--drdo-red)';
        badge.style.background = 'rgba(255,51,68,0.2)';
      }
    }

    const throtCapEl = document.getElementById('hud-throttle-cap');
    if (throtCapEl) throtCapEl.innerText = `${rel.max_safe_throttle_pct}%`;
    const totRangeEl = document.getElementById('hud-total-range');
    if (totRangeEl) totRangeEl.innerText = `${rel.total_reachable_range_nm} NM`;
    const powEndEl = document.getElementById('hud-powered-endurance');
    if (powEndEl) powEndEl.innerText = `${rel.endurance_remaining_hours} HRS`;

    const basesList = document.getElementById('hud-recovery-bases-list');
    if (basesList && rel.recovery_bases) {
      basesList.innerHTML = '';
      rel.recovery_bases.forEach(b => {
        const item = document.createElement('div');
        item.style.display = 'flex';
        item.style.justifyContent = 'space-between';
        item.style.background = '#090d14';
        item.style.padding = '2px 5px';
        item.style.borderRadius = '3px';
        item.innerHTML = `
          <span style="color: ${b.reachable ? '#e6edf3' : '#ff3344'};">${b.base_name} (${b.distance_nm} NM)</span>
          <span style="color: ${b.reachable ? 'var(--drdo-green)' : 'var(--drdo-red)'}; font-family: var(--font-mono);">${b.reachable ? '+' + b.safety_margin_nm + ' NM' : 'UNREACHABLE'}</span>
        `;
        basesList.appendChild(item);
      });
    }
  }

  // 11. Replay Status Bar Synchronization
  if (pkt.replay_status) {
    const r = pkt.replay_status;
    const exitBtn = document.getElementById('btn-exit-replay');
    const controlsDiv = document.getElementById('replay-controls');
    const scrubSlider = document.getElementById('replay-scrub-slider');
    const frameLabel = document.getElementById('label-replay-frame');

    if (r.is_replaying) {
      exitBtn.style.display = 'inline-block';
      controlsDiv.style.display = 'flex';
      scrubSlider.max = Math.max(1, r.total_frames - 1);
      scrubSlider.value = r.cursor;
      frameLabel.innerText = `${r.cursor} / ${r.total_frames}`;
    } else {
      exitBtn.style.display = 'none';
      controlsDiv.style.display = 'none';
    }
  }
}

function updateCylinderCard(cylNum, cht, egt, resCht, resEgt, alerts) {
  const card = document.getElementById(`cyl-card-${cylNum}`);
  const chtEl = document.getElementById(`cyl-${cylNum}-cht`);
  const egtEl = document.getElementById(`cyl-${cylNum}-egt`);
  const crownEl = document.getElementById(`cyl-${cylNum}-crown`);

  chtEl.innerText = `${cht.toFixed(0)}°C`;
  egtEl.innerText = `${egt.toFixed(0)}°C`;

  // Thermal color gradient for piston crown
  if (cht > 135.0) {
    crownEl.style.backgroundColor = '#ff3344';
  } else if (cht > 115.0) {
    crownEl.style.backgroundColor = '#ffaa00';
  } else {
    crownEl.style.backgroundColor = '#38bdf8';
  }

  // Check if misfire or fault is active on this cylinder
  const isMisfire = alerts.some(a => a.fault_code.includes('MISF') && a.affected_cylinder === cylNum);
  if (isMisfire) {
    card.classList.add('misfire');
  } else {
    card.classList.remove('misfire');
  }
}

function startEngineAnimationLoop() {
  setInterval(() => {
    if (!currentPacket) return;
    const rpm = currentPacket.telemetry.rpm;
    // Highlight firing cylinder in sequence [1 -> 4 -> 3 -> 2]
    firingIndex = (firingIndex + 1) % 4;
    const firingCyl = FIRING_ORDER[firingIndex];

    for (let i = 1; i <= 4; i++) {
      const card = document.getElementById(`cyl-card-${i}`);
      if (i === firingCyl) {
        card.classList.add('firing');
      } else {
        card.classList.remove('firing');
      }
    }
  }, 120);
}

function setProgressBar(elementId, valuePct) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const pct = Math.max(0, Math.min(100, valuePct));
  el.style.width = `${pct}%`;
  el.className = 'progress-fill ' + (pct < 60 ? 'danger' : pct < 80 ? 'warn' : '');
}

function setResidualCell(elementId, val, unit, warnThresh, critThresh) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const absVal = Math.abs(val);
  const sign = val >= 0 ? '+' : '';
  el.innerText = `${sign}${val.toFixed(1)} ${unit}`;
  if (absVal > critThresh) {
    el.className = 'res-crit';
  } else if (absVal > warnThresh) {
    el.className = 'res-warn';
  } else {
    el.className = 'res-good';
  }
}

function appendChartData(time_s, rpm, map, cht_avg, egt_avg, oil_p) {
  chartHistory.timestamps.push(time_s);
  chartHistory.rpm.push(rpm);
  chartHistory.map.push(map);
  chartHistory.cht_avg.push(cht_avg);
  chartHistory.egt_avg.push(egt_avg);
  chartHistory.oil_p.push(oil_p);

  if (chartHistory.timestamps.length > MAX_CHART_POINTS) {
    chartHistory.timestamps.shift();
    chartHistory.rpm.shift();
    chartHistory.map.shift();
    chartHistory.cht_avg.shift();
    chartHistory.egt_avg.shift();
    chartHistory.oil_p.shift();
  }
  renderCanvasChart();
}

function renderCanvasChart() {
  const canvas = document.getElementById('strip-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = '#141d2c';
  ctx.lineWidth = 1;
  for (let y = 20; y < h; y += 25) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const n = chartHistory.timestamps.length;
  if (n < 2) return;

  // Draw RPM curve (scaled 0 - 6000 to h)
  drawLine(ctx, chartHistory.rpm, 0, 6000, '#00ff88', w, h);
  // Draw CHT Avg curve (scaled 50 - 150 to h)
  drawLine(ctx, chartHistory.cht_avg, 50, 150, '#38bdf8', w, h);
  // Draw EGT Avg curve (scaled 500 - 900 to h)
  drawLine(ctx, chartHistory.egt_avg, 500, 900, '#f97316', w, h);
  // Draw Oil P curve (scaled 0 - 6 to h)
  drawLine(ctx, chartHistory.oil_p, 0, 6, '#ffaa00', w, h);
}

function drawLine(ctx, arr, minVal, maxVal, color, w, h) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const stepX = w / (MAX_CHART_POINTS - 1);
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    const normY = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
    const y = h - (normY * (h - 10) + 5);
    const x = i * stepX;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawPVCycle(pv) {
  const canvas = document.getElementById('pv-diagram-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#070a10';
  ctx.fillRect(0, 0, w, h);

  // Subtle gridlines
  ctx.strokeStyle = '#141c2c';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 35; x < w; x += 45) { ctx.moveTo(x, 0); ctx.lineTo(x, h - 18); }
  for (let y = 10; y < h - 18; y += 25) { ctx.moveTo(35, y); ctx.lineTo(w, y); }
  ctx.stroke();

  // Axis markings
  ctx.fillStyle = '#546580';
  ctx.font = '8px Consolas, monospace';
  ctx.fillText('0 bar', 2, h - 20);
  ctx.fillText('80 bar', 2, 12);
  ctx.fillText('Vc (TDC)', 38, h - 4);
  ctx.fillText('Vd (BDC)', w - 50, h - 4);

  const vols = pv.volume_cm3;
  const press = pv.pressure_bar;
  if (!vols || vols.length === 0) return;

  const vMin = 35.0, vMax = 390.0;
  const pMin = 0.0, pMax = 85.0;

  // Draw P-V indicator loop
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();

  for (let i = 0; i < vols.length; i++) {
    const x = 35 + ((vols[i] - vMin) / (vMax - vMin)) * (w - 45);
    const y = (h - 20) - ((press[i] - pMin) / (pMax - pMin)) * (h - 30);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Highlight peak firing pressure
  const peakP = pv.peak_cylinder_pressure_bar;
  const peakIdx = press.indexOf(peakP);
  if (peakIdx >= 0) {
    const px = 35 + ((vols[peakIdx] - vMin) / (vMax - vMin)) * (w - 45);
    const py = (h - 20) - ((peakP - pMin) / (pMax - pMin)) * (h - 30);
    ctx.fillStyle = '#ffaa00';
    ctx.shadowColor = '#ffaa00';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffaa00';
    ctx.fillText(`${peakP} bar`, px + 5, py - 3);
  }
}

function drawEnginePerformanceMap(rpm, bmep, bsfc, comp) {
  const canvas = document.getElementById('engine-map-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#070a10';
  ctx.fillRect(0, 0, w, h);

  const rpmMin = 1400, rpmMax = 5800;
  const bmepMin = 0, bmepMax = 14;

  const toX = (r) => 30 + ((r - rpmMin) / (rpmMax - rpmMin)) * (w - 40);
  const toY = (b) => (h - 18) - ((b - bmepMin) / (bmepMax - bmepMin)) * (h - 28);

  // Subtle grid
  ctx.strokeStyle = '#141c2c';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let r = 2000; r <= 5000; r += 1000) { ctx.moveTo(toX(r), 10); ctx.lineTo(toX(r), h - 18); }
  for (let b = 3; b <= 12; b += 3) { ctx.moveTo(30, toY(b)); ctx.lineTo(w - 10, toY(b)); }
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#546580';
  ctx.font = '8px Consolas, monospace';
  ctx.fillText('2k', toX(2000) - 6, h - 6);
  ctx.fillText('4k', toX(4000) - 6, h - 6);
  ctx.fillText('5.5k', toX(5500) - 10, h - 6);
  ctx.fillText('12b', 8, toY(12) + 3);
  ctx.fillText('6b', 12, toY(6) + 3);

  // Draw calibrated BSFC contour islands
  ctx.strokeStyle = '#1f334d';
  ctx.lineWidth = 1;
  [
    { rx: 50, ry: 20, label: '240 g/kWh' },
    { rx: 90, ry: 35, label: '260 g/kWh' },
    { rx: 130, ry: 48, label: '280 g/kWh' }
  ].forEach(island => {
    ctx.beginPath();
    ctx.ellipse(toX(4800), toY(9.5), island.rx * (w / 340), island.ry * (h / 135), -0.15, 0, 2 * Math.PI);
    ctx.stroke();
  });

  // Full-throttle envelope curve
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(toX(1800), toY(6.5));
  ctx.quadraticCurveTo(toX(3800), toY(13.2), toX(5800), toY(11.8));
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot live operating point dot
  const dotX = Math.max(30, Math.min(w - 10, toX(rpm)));
  const dotY = Math.max(10, Math.min(h - 18, toY(bmep)));

  ctx.fillStyle = '#00ff88';
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(dotX, dotY, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Update text readouts
  const bsfcEl = document.getElementById('label-engine-map-bsfc');
  if (bsfcEl) bsfcEl.innerText = `BMEP: ${bmep.toFixed(1)} bar | ${bsfc.toFixed(0)} g/kWh`;

  if (comp) {
    const smEl = document.getElementById('val-surge-margin');
    if (smEl) {
      smEl.innerText = `${comp.surge_margin_pct}%`;
      smEl.style.color = comp.surge_margin_pct < 15.0 ? 'var(--drdo-red)' : 'var(--drdo-green)';
    }
    const csEl = document.getElementById('val-compressor-status');
    if (csEl) {
      csEl.innerText = comp.aerodynamic_status;
      csEl.style.color = comp.aerodynamic_status === 'STABLE' ? '#38bdf8' : 'var(--drdo-red)';
    }
  }
}

function drawTacticalMap(telem, reliability) {
  const canvas = document.getElementById('tactical-map-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#05080f';
  ctx.fillRect(0, 0, w, h);

  // Radar range rings & coordinate grid
  ctx.strokeStyle = '#121b2a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 40; x < w; x += 60) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let y = 30; y < h; y += 35) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();

  // Grid coordinates (Western Sector / Rajasthan Border)
  ctx.fillStyle = '#3a4b66';
  ctx.font = '8px Consolas, monospace';
  ctx.fillText('27°30\'N', 5, 15);
  ctx.fillText('26°15\'N', 5, h - 6);
  ctx.fillText('70°15\'E', 50, h - 6);
  ctx.fillText('72°45\'E', w - 50, h - 6);

  // Airbases coordinates (normalized to canvas space)
  const bases = [
    { name: 'AFB Uttarlai', x: 130, y: 135, rwy: '02/20 (2743m)', dist_nm: 95 },
    { name: 'AFB Jaisalmer', x: 230, y: 48, rwy: '04/22 (2743m)', dist_nm: 42 },
    { name: 'AFB Nal (Bikaner)', x: 480, y: 35, rwy: '05/23 (2743m)', dist_nm: 115 },
    { name: 'FOB-Alpha', x: 330, y: 95, rwy: '1600m Strip', dist_nm: 28 },
    { name: 'EHS-Bravo', x: 270, y: 140, rwy: '900m Strip', dist_nm: 18 }
  ];

  // Dynamic UAV position in tactical patrol orbit
  const time_s = telem.timestamp_s || 0;
  const orbitRadius = 18;
  const uavX = 290 + Math.cos(time_s * 0.15) * orbitRadius;
  const uavY = 90 + Math.sin(time_s * 0.15) * (orbitRadius * 0.65);

  // Range footprint calculations
  const alt_ft = Math.max(500, telem.altitude_ft || 10000);
  const glide_nm = (alt_ft / 6076.12) * 18.2;
  const glide_px = Math.min(180, Math.max(25, glide_nm * 2.2));
  const powered_nm = reliability ? reliability.powered_range_nm || 250 : 250;
  const powered_px = Math.min(260, Math.max(50, (powered_nm / 3.0) * 1.5));

  // Draw Powered Loiter Boundary (Green circle)
  ctx.strokeStyle = 'rgba(0, 255, 136, 0.35)';
  ctx.fillStyle = 'rgba(0, 255, 136, 0.04)';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(uavX, uavY, powered_px, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Dead-Stick Gliding Cone (Amber ellipse)
  ctx.strokeStyle = 'rgba(255, 170, 0, 0.7)';
  ctx.fillStyle = 'rgba(255, 170, 0, 0.08)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(uavX, uavY, glide_px, glide_px * 0.75, 0, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();

  // Draw Airbases & Reachability vectors
  bases.forEach(b => {
    const distToUavPx = Math.hypot(b.x - uavX, b.y - uavY);
    const isGlideReachable = distToUavPx <= glide_px;
    const isPoweredReachable = distToUavPx <= powered_px;

    let baseColor = '#ff3344'; // Unreachable
    let statusText = 'UNREACHABLE';
    if (isPoweredReachable) {
      baseColor = '#00ff88';
      statusText = isGlideReachable ? 'REACHABLE [GLIDE+PWR]' : 'REACHABLE [PWR]';
    } else if (isGlideReachable) {
      baseColor = '#ffaa00';
      statusText = 'GLIDE CONE ONLY';
    }

    // Runway icon
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(b.x - 7, b.y - 4);
    ctx.lineTo(b.x + 7, b.y + 4);
    ctx.stroke();

    // Base point marker
    ctx.fillStyle = baseColor;
    ctx.beginPath();
    ctx.arc(b.x, b.y, 3, 0, 2 * Math.PI);
    ctx.fill();

    // Base Label
    ctx.font = '9px Consolas, monospace';
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(b.name, b.x + 8, b.y - 3);
    ctx.fillStyle = baseColor;
    ctx.font = '7.5px Consolas, monospace';
    ctx.fillText(statusText, b.x + 8, b.y + 7);
  });

  // Draw UAV Position & Heading Vector
  ctx.fillStyle = '#00d2ff';
  ctx.shadowColor = '#00d2ff';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(uavX, uavY, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Heading vector line
  const headingRad = (time_s * 0.15) + Math.PI / 2;
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(uavX, uavY);
  ctx.lineTo(uavX + Math.cos(headingRad) * 16, uavY + Math.sin(headingRad) * 16);
  ctx.stroke();

  // UAV Callout Tag
  ctx.fillStyle = '#00d2ff';
  ctx.font = '9px Consolas, monospace';
  ctx.fillText('UAV-01 [TAPAS-BH-201]', uavX + 8, uavY - 8);
  ctx.fillStyle = '#8b9bb4';
  ctx.font = '8px Consolas, monospace';
  ctx.fillText(`${alt_ft.toFixed(0)} FT | ${(telem.airspeed_kts || 110).toFixed(0)} KTS`, uavX + 8, uavY + 2);
}

function setupUIEventListeners() {
  // ML Model Dossier View
  const btnMlDossier = document.getElementById('btn-view-ml-dossier');
  if (btnMlDossier) {
    btnMlDossier.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/ml/metrics');
        const metrics = await resp.json();
        if (metrics.performance_metrics) {
          const pm = metrics.performance_metrics;
          const rEl = document.getElementById('dos-roc-auc');
          if (rEl) rEl.innerText = pm.roc_auc.toFixed(4);
          const pEl = document.getElementById('dos-precision');
          if (pEl) pEl.innerText = `${pm.precision_pct.toFixed(2)}%`;
          const recEl = document.getElementById('dos-recall');
          if (recEl) recEl.innerText = `${pm.recall_pct.toFixed(2)}%`;
          const f1El = document.getElementById('dos-f1');
          if (f1El) f1El.innerText = `${pm.f1_score_pct.toFixed(2)}%`;
          const latEl = document.getElementById('dos-latency');
          if (latEl) latEl.innerText = `${pm.avg_inference_latency_ms.toFixed(2)} ms`;
        }
      } catch (e) {
        console.warn('Could not fetch ML metrics', e);
      }
      const overlay = document.getElementById('ml-dossier-modal-overlay');
      if (overlay) overlay.style.display = 'flex';
    });
  }

  const btnCloseMlDossier = document.getElementById('btn-close-ml-dossier');
  if (btnCloseMlDossier) {
    btnCloseMlDossier.addEventListener('click', () => {
      const overlay = document.getElementById('ml-dossier-modal-overlay');
      if (overlay) overlay.style.display = 'none';
    });
  }

  // Voice & Acoustic Annunciator Toggle
  const btnToggleAudio = document.getElementById('btn-toggle-audio');
  if (btnToggleAudio) {
    btnToggleAudio.addEventListener('click', () => {
      audioEnabled = !audioEnabled;
      if (audioEnabled) {
        getAudioContext();
        btnToggleAudio.innerText = '🔊 VOICE ANNUNCIATOR: ON';
        btnToggleAudio.style.color = 'var(--drdo-cyan)';
        btnToggleAudio.style.borderColor = 'var(--drdo-cyan)';
      } else {
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        btnToggleAudio.innerText = '🔇 VOICE ANNUNCIATOR: MUTED';
        btnToggleAudio.style.color = 'var(--text-muted)';
        btnToggleAudio.style.borderColor = 'var(--border-card)';
      }
    });
  }

  // Mission Profile selector
  const profileSelect = document.getElementById('select-mission-profile');
  profileSelect.addEventListener('change', (e) => {
    sendCommand({ action: 'SET_PROFILE', profile_key: e.target.value });
  });

  // Fault Injector trigger
  document.getElementById('btn-inject-fault').addEventListener('click', () => {
    const faultType = document.getElementById('select-fault-type').value;
    const cyl = parseInt(document.getElementById('select-fault-cyl').value, 10);
    sendCommand({
      action: 'INJECT_FAULT',
      fault_type: faultType,
      severity: 0.7,
      target_cylinder: cyl
    });
  });

  // Clear Faults
  document.getElementById('btn-clear-faults').addEventListener('click', () => {
    sendCommand({ action: 'CLEAR_ALL_FAULTS' });
  });

  // Throttle Override Slider
  const throttleSlider = document.getElementById('slider-throttle');
  throttleSlider.addEventListener('input', (e) => {
    document.getElementById('label-throttle-val').innerText = `${e.target.value}%`;
    sendCommand({
      action: 'THROTTLE_OVERRIDE',
      throttle_pct: parseFloat(e.target.value)
    });
  });

  // Report Generator button
  document.getElementById('btn-generate-report').addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/report/generate', { method: 'POST' });
      const report = await resp.json();
      openReportModal(report);
    } catch (err) {
      alert('Failed to generate DRDO mission health report: ' + err);
    }
  });

  // Close report modal
  document.getElementById('btn-close-modal').addEventListener('click', () => {
    document.getElementById('report-modal-overlay').style.display = 'none';
  });

  // Replay: Load Recording
  document.getElementById('btn-load-replay').addEventListener('click', async () => {
    const filename = document.getElementById('select-replay-mission').value;
    if (!filename) {
      alert('Please select a recorded flight session from the dropdown.');
      return;
    }
    try {
      const resp = await fetch('/api/replay/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      const data = await resp.json();
      if (resp.ok) {
        document.getElementById('btn-exit-replay').style.display = 'inline-block';
        document.getElementById('replay-controls').style.display = 'flex';
      } else {
        alert('Failed to load replay: ' + (data.detail || 'Unknown error'));
      }
    } catch (e) {
      alert('Network error loading replay: ' + e);
    }
  });

  // Replay: Exit Replay (Resume Live)
  document.getElementById('btn-exit-replay').addEventListener('click', async () => {
    try {
      await fetch('/api/replay/exit', { method: 'POST' });
      document.getElementById('btn-exit-replay').style.display = 'none';
      document.getElementById('replay-controls').style.display = 'none';
    } catch (e) {
      console.error('Error exiting replay', e);
    }
  });

  // Replay: Scrub Slider
  const scrubSlider = document.getElementById('replay-scrub-slider');
  scrubSlider.addEventListener('input', async (e) => {
    const frameIndex = parseInt(e.target.value, 10);
    try {
      await fetch('/api/replay/scrub', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: frameIndex })
      });
    } catch (err) {
      console.error('Error scrubbing replay', err);
    }
  });
}

async function populateRecordingsList() {
  try {
    const resp = await fetch('/api/recordings');
    const recordings = await resp.json();
    const select = document.getElementById('select-replay-mission');
    select.innerHTML = '<option value="">-- Select Recorded Flight Session --</option>';
    recordings.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.filename;
      opt.innerText = `${r.filename} (${r.size_kb} KB - ${r.created_at})`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load recordings list', err);
  }
}

function sendCommand(cmd) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(cmd));
  }
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function openReportModal(r) {
  document.getElementById('modal-report-id').innerText = r.report_id;
  document.getElementById('modal-report-body').innerHTML = `
    <div style="background: #0d131f; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
      <h3 style="color: #58a6ff; margin-bottom: 5px;">${r.organization}</h3>
      <div style="color: #8b949e; font-size: 12px;">Problem Statement: ${r.problem_statement_id} | Mission: ${r.mission_name} | Duration: ${r.mission_duration_minutes} min</div>
      <div style="margin-top: 10px; display: flex; gap: 20px;">
        <div><strong>Final Engine Health:</strong> <span style="color: #00ff88;">${r.final_engine_health_index}%</span></div>
        <div><strong>Mission Reliability:</strong> <span style="color: #38bdf8;">${r.mission_reliability_score}%</span></div>
        <div><strong>RUL Median (P50):</strong> <span style="color: #ffaa00;">${r.prognostics.remaining_useful_life_p50_hours} hrs</span></div>
      </div>
    </div>
    <h4 style="color: #79c0ff; margin-bottom: 8px;">Prescriptive Work Order Directives:</h4>
    <div style="background: #090d14; border: 1px solid #1f2c42; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 11px;">
      Primary Limiting Subsystem: ${r.prognostics.limiting_component}<br>
      Degradation Trend: ${r.prognostics.degradation_trend}<br>
      Confidence Interval (90%): [${r.prognostics.rul_confidence_interval_90pct[0]} - ${r.prognostics.rul_confidence_interval_90pct[1]}] hrs<br>
      Logged Anomalies: ${r.detected_fault_count}
    </div>
    <div style="margin-top: 15px; text-align: right;">
      <a href="${r.html_filepath.replace(/^.*[\\\/]/, '/reports/')}" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: bold; margin-right: 15px;">Download Printable HTML Report</a>
      <a href="${r.json_filepath.replace(/^.*[\\\/]/, '/reports/')}" target="_blank" style="color: #00ff88; text-decoration: underline;">View Raw JSON</a>
    </div>
  `;
  document.getElementById('report-modal-overlay').style.display = 'flex';
}
