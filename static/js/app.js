/**
 * Engineering Time Series Viewer — Core Application Module
 *
 * Provides:
 *   - API helpers (apiFetch, apiPost)
 *   - Path encoding (base64url)
 *   - GlobalNav: root → serial → step → file cascade + sidebar batch/signal tree
 *   - URL state persistence via sessionStorage
 *   - Loading overlay & toast notifications
 *   - Shared constants (SIGNAL_COLORS, formatNumber, etc.)
 */

// ============================================================================
// 1. CONFIGURATION & CONSTANTS
// ============================================================================

const APP_CONFIG = {
  API_TIMEOUT: 60000,
  DEFAULT_DECIMALS: 4,
  TOAST_DURATION: 4000,
  DEFAULT_DOWNSAMPLE: 5000
};

const SIGNAL_COLORS = [
  '#0077cc', '#e65100', '#2e7d32', '#c62828',
  '#6a1b9a', '#00838f', '#ef6c00', '#37474f'
];

const PLOT_CONFIG = {
  margin:   { l: 80, r: 20, t: 24, b: 56, autoexpand: true },
  marginSm: { l: 64, r: 16, t: 24, b: 52, autoexpand: true },
  axis:     { showgrid: true, gridwidth: 1, gridcolor: 'rgba(0,0,0,0.06)',
              zeroline: false, showline: true, linewidth: 1, linecolor: '#d1d5db' },
  legend:   { bgcolor: 'rgba(255,255,255,0.92)', bordercolor: '#d1d5db',
              borderwidth: 1, font: { size: 11, color: '#555770' } },
  font:     { family: 'system-ui, -apple-system, sans-serif', size: 12, color: '#555770' },
  bg: '#fff',
  paperBg: '#fff',
  CMP_COLORS: ['#ef4444', '#3b82f6', '#10b981']
};

// ============================================================================
// 2. API COMMUNICATION
// ============================================================================

/**
 * Fetch JSON from API with timeout and error extraction.
 * @param {string} url
 * @param {object} options  fetch() options
 * @returns {Promise<object>}
 */
async function apiFetch(url, options = {}) {
  const timeoutCtrl = new AbortController();
  const timeout = setTimeout(() => timeoutCtrl.abort(), APP_CONFIG.API_TIMEOUT);

  // Merge caller's abort signal with our timeout signal
  const callerSignal = options.signal || null;
  let mergedSignal;
  if (callerSignal && typeof AbortSignal.any === 'function') {
    mergedSignal = AbortSignal.any([callerSignal, timeoutCtrl.signal]);
  } else if (callerSignal) {
    // Polyfill: forward caller abort to timeout controller
    if (callerSignal.aborted) { timeoutCtrl.abort(); }
    else { callerSignal.addEventListener('abort', () => timeoutCtrl.abort(), { once: true }); }
    mergedSignal = timeoutCtrl.signal;
  } else {
    mergedSignal = timeoutCtrl.signal;
  }

  try {
    const fetchOpts = { ...options, signal: mergedSignal };
    const response = await fetch(url, fetchOpts);

    if (!response.ok) {
      let msg = `HTTP ${response.status}`;
      try { const d = await response.json(); if (d.error) msg = d.error; }
      catch (_) { /* ignore */ }
      throw new Error(msg);
    }

    return await response.json();
  } catch (err) {
    // Re-throw caller-initiated aborts as AbortError (not timeout message)
    if (callerSignal && callerSignal.aborted) {
      const e = new DOMException('The operation was aborted.', 'AbortError');
      throw e;
    }
    if (err.name === 'AbortError') throw new Error('Request timeout (exceeded ' + (APP_CONFIG.API_TIMEOUT / 1000) + ' s)');
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * POST JSON to API.
 * @param {string} url
 * @param {object} data  request body
 * @returns {Promise<object>}
 */
async function apiPost(url, data) {
  return apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
}

// ============================================================================
// 3. PATH ENCODING (base64url, no padding)
// ============================================================================

function encodePath(path) {
  return btoa(path).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function decodePath(encoded) {
  let s = encoded.replace(/-/g, '+').replace(/_/g, '/');
  s += '='.repeat((4 - (s.length % 4)) % 4);
  return atob(s);
}

// ============================================================================
// 4. LOADING OVERLAY & TOASTS
// ============================================================================

// ── Progress bar loading overlay ──────────────────────────────────────────
//
// Estimated durations per task.  For data-dependent tasks the caller can
// supply a `hint` object to showLoading() with sizing information so the
// estimate scales with the actual payload.
//
// hint = { signalCount, sampleCount, fileSize }   (all optional)
//
// Base estimates (ms) for small / default payloads:
const _PROGRESS_BASE = {
  'Restoring session\u2026':    800,
  'Loading steps\u2026':        300,
  'Loading files\u2026':        300,
  'Loading signal list\u2026':  500,
  'Loading signal\u2026':       2000,   // scaled by sampleCount
  'Computing statistics\u2026': 1000,   // scaled by sampleCount
};
const _PROGRESS_DEFAULT_MS = 1500;
let _progressTimer = null;
let _progressStart = 0;
let _progressEstMs = _PROGRESS_DEFAULT_MS;

/**
 * Estimate task duration (ms) based on task label and optional size hint.
 * @param {string} text    task label
 * @param {object} [hint]  { signalCount, sampleCount, fileSize }
 * @returns {number} estimated duration in ms
 */
function _estimateMs(text, hint) {
  const base = _PROGRESS_BASE[text] || _PROGRESS_DEFAULT_MS;
  if (!hint) return base;

  // Signal loading: ~2 s baseline + ~1 ms per 1 000 samples (network + decode)
  if (text === 'Loading signal\u2026' && hint.sampleCount) {
    return Math.max(base, 500 + hint.sampleCount / 1000);
  }
  // Statistics: similar scaling
  if (text === 'Computing statistics\u2026' && hint.sampleCount) {
    return Math.max(base, 400 + hint.sampleCount / 1500);
  }
  // Signal list: scales mildly with signal count
  if (text === 'Loading signal list\u2026' && hint.signalCount) {
    return Math.max(base, 300 + hint.signalCount * 3);
  }
  // File-size based (generic fallback)
  if (hint.fileSize) {
    return Math.max(base, 300 + hint.fileSize / (1024 * 50)); // ~1 s per 50 KB
  }
  return base;
}

/**
 * Show the loading overlay with a progress bar.
 * @param {string} text   label to display
 * @param {object} [hint] optional sizing hint for duration estimate
 */
function showLoading(text, hint) {
  if (text === undefined) text = 'Loading\u2026';
  const overlay = document.getElementById('loadingOverlay');
  const textEl  = document.getElementById('loadingText');
  const barEl   = document.getElementById('loadingBarFill');
  if (overlay) { overlay.classList.add('active'); }
  if (textEl)  { textEl.textContent = text; }
  if (barEl)   { barEl.style.transition = 'none'; barEl.style.width = '0%'; }

  // Start simulated progress
  _progressEstMs = _estimateMs(text, hint);
  _progressStart = Date.now();
  _clearProgressTimer();
  // Allow the 0% to paint, then start animating
  requestAnimationFrame(function () {
    if (barEl) { barEl.style.transition = 'width 0.3s ease'; }
    _progressTimer = setInterval(_tickProgress, 200);
  });
}

function hideLoading() {
  var overlay = document.getElementById('loadingOverlay');
  var barEl   = document.getElementById('loadingBarFill');
  _clearProgressTimer();
  // Snap to 100% briefly before hiding
  if (barEl) {
    barEl.style.transition = 'width 0.15s ease';
    barEl.style.width = '100%';
  }
  setTimeout(function () {
    if (overlay) { overlay.classList.remove('active'); }
    if (barEl)   { barEl.style.transition = 'none'; barEl.style.width = '0%'; }
  }, 180);
}

function _tickProgress() {
  var barEl = document.getElementById('loadingBarFill');
  if (!barEl) return;
  var elapsed = Date.now() - _progressStart;
  // Asymptotic curve: approaches 90% at estimated time, never reaches 100%
  var ratio = elapsed / _progressEstMs;
  var pct = Math.min(90, 90 * (1 - Math.exp(-2.5 * ratio)));
  barEl.style.width = pct.toFixed(1) + '%';
}

function _clearProgressTimer() {
  if (_progressTimer) { clearInterval(_progressTimer); _progressTimer = null; }
}

function showToast(message, type = 'info', duration = APP_CONFIG.TOAST_DURATION) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons  = { info: '\u2139', success: '\u2713', error: '\u2715', warning: '\u26A0' };
  const colors = { info: '#0077cc', success: '#2e7d32', error: '#c62828', warning: '#f57f17' };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span style="font-weight:bold;font-size:16px;color:${colors[type] || colors.info};flex-shrink:0">
      ${icons[type] || icons.info}
    </span>
    <span style="flex:1;color:var(--color-text-primary)">${escapeHtml(message)}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
  `;
  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => { if (toast.parentElement) toast.remove(); }, duration);
  }
}

// ============================================================================
// 5. UTILITY FUNCTIONS
// ============================================================================

function escapeHtml(text) {
  if (text == null) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

/** Escape HTML then convert ^… patterns to <sup> tags.
 *  Handles ^2, ^-1, ^{2.5}, ^{ab}, etc. */
function fmtHtml(text) {
  const safe = escapeHtml(text);
  return safe.replace(/\^{([^}]+)}/g, '<sup>$1</sup>')
             .replace(/\^(-?\w+)/g, '<sup>$1</sup>');
}

function formatNumber(num, decimals = APP_CONFIG.DEFAULT_DECIMALS) {
  if (num == null) return 'N/A';
  const abs = Math.abs(num);
  if (abs >= 1e6 || (abs < 1e-3 && abs !== 0)) return num.toExponential(decimals);
  return num.toFixed(decimals);
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + units[i];
}

function getSignalColor(index) {
  return SIGNAL_COLORS[index % SIGNAL_COLORS.length];
}

// ============================================================================
// 6. URL STATE PERSISTENCE
// ============================================================================

const URLState = {
  save() {
    sessionStorage.setItem('globalNavState', JSON.stringify({
      root: GlobalNav.root,
      serial: GlobalNav.serial,
      step: GlobalNav.step,
      filePath: GlobalNav.filePath,
      fileName: GlobalNav.fileName,
      fileSize: GlobalNav.fileSize
    }));
  },
  load() {
    try { return JSON.parse(sessionStorage.getItem('globalNavState')); }
    catch (_) { return null; }
  },
  saveSidebar() {
    const el = GlobalNav._sidebarEl;
    if (!el) return;
    const expanded = [];
    el.querySelectorAll('.batch-group').forEach((grp, i) => {
      const list = grp.querySelector('.batch-group-items');
      if (list && list.style.display !== 'none') expanded.push(i);
    });
    sessionStorage.setItem('sidebarState', JSON.stringify({
      expanded,
      scrollTop: el.scrollTop
    }));
  },
  restoreSidebar() {
    try {
      const saved = JSON.parse(sessionStorage.getItem('sidebarState'));
      if (!saved) return;
      const el = GlobalNav._sidebarEl;
      if (!el) return;
      const groups = el.querySelectorAll('.batch-group');
      groups.forEach((grp, i) => {
        const list = grp.querySelector('.batch-group-items');
        const header = grp.querySelector('.batch-group-header');
        if (!list || !header) return;
        const shouldExpand = saved.expanded.includes(i);
        list.style.display = shouldExpand ? 'block' : 'none';
        header.classList.toggle('collapsed', !shouldExpand);
      });
      if (typeof saved.scrollTop === 'number') {
        el.scrollTop = saved.scrollTop;
      }
    } catch (_) { /* ignore */ }
  }
};

// ============================================================================
// 7. FILTERABLE SELECT WIDGET
// ============================================================================

/**
 * FilterSelect — a lightweight combobox that turns a <div class="filter-select">
 * into a filterable dropdown.  Type to filter, arrow-keys + Enter to navigate,
 * Escape to close.  Fully replaces native <select> elements.
 *
 * Usage:
 *   const fs = new FilterSelect(containerEl, { onChange: (value, label) => {} });
 *   fs.setItems([{ value: 'a', label: 'Alpha' }]);
 *   fs.setValue('a');
 *   fs.setDisabled(true);
 */
class FilterSelect {
  constructor(containerEl, opts = {}) {
    this._el = containerEl;
    this._onChange = opts.onChange || null;
    this._placeholder = containerEl.dataset.placeholder || 'Select\u2026';
    this._items = [];       // [{ value, label }]
    this._value = '';       // current selected value
    this._hlIdx = -1;       // highlighted index in filtered list
    this._filtered = [];    // currently visible items after filter
    this._open = false;

    // Build DOM — input lives inside the container
    this._input = document.createElement('input');
    this._input.type = 'text';
    this._input.className = 'fs-input';
    this._input.placeholder = this._placeholder;
    this._input.autocomplete = 'off';
    this._input.spellcheck = false;
    this._el.appendChild(this._input);

    // Dropdown lives on <body> so it is never clipped by overflow:hidden
    this._dropdown = document.createElement('div');
    this._dropdown.className = 'fs-dropdown';
    document.body.appendChild(this._dropdown);

    // Events
    this._input.addEventListener('focus', () => this._openDropdown());
    this._input.addEventListener('input', () => this._onInput());
    this._input.addEventListener('keydown', (e) => this._onKeydown(e));

    // Close on outside click — check both container and portal dropdown
    document.addEventListener('mousedown', (e) => {
      if (!this._el.contains(e.target) && !this._dropdown.contains(e.target)) {
        this._closeDropdown(true);
      }
    });

    this.setDisabled(true);
  }

  /* ── Public API ─────────────────────────────────────────────────────── */

  /** Replace the list of items. Items: [{ value, label }] or [string]. */
  setItems(items) {
    this._items = items.map(it =>
      typeof it === 'string' ? { value: it, label: it } : it
    );
    this._filtered = this._items.slice();
    this._value = '';
    this._input.value = '';
    this._input.placeholder = this._placeholder;
  }

  /** Programmatically select a value (no onChange fired). */
  setValue(val) {
    const item = this._items.find(it => it.value === val);
    this._value = val;
    this._input.value = item ? item.label : '';
  }

  /** Get current selected value. */
  getValue() { return this._value; }

  /** Enable / disable the widget. */
  setDisabled(flag) {
    if (flag) {
      this._el.classList.add('fs-disabled');
      this._input.disabled = true;
      this._closeDropdown(false);
    } else {
      this._el.classList.remove('fs-disabled');
      this._input.disabled = false;
    }
  }

  /** Reset to empty placeholder state. */
  reset(placeholder) {
    if (placeholder) this._placeholder = placeholder;
    this._items = [];
    this._filtered = [];
    this._value = '';
    this._input.value = '';
    this._input.placeholder = this._placeholder;
    this._dropdown.innerHTML = '';
    this.setDisabled(true);
  }

  /* ── Internal ───────────────────────────────────────────────────────── */

  /** Position the portal dropdown directly below the input using fixed coords. */
  _positionDropdown() {
    const rect = this._input.getBoundingClientRect();
    this._dropdown.style.position = 'fixed';
    this._dropdown.style.top = rect.bottom + 'px';
    this._dropdown.style.left = rect.left + 'px';
    this._dropdown.style.width = rect.width + 'px';
  }

  _openDropdown() {
    if (this._open) return;
    this._open = true;
    // Select existing text so the user can start typing immediately
    this._input.select();
    this._filtered = this._items.slice();
    this._hlIdx = this._items.findIndex(it => it.value === this._value);
    this._renderDropdown();
    this._positionDropdown();
    this._dropdown.style.display = 'block';
    this._scrollToHighlighted();
  }

  _closeDropdown(restoreValue) {
    if (!this._open) return;
    this._open = false;
    this._dropdown.style.display = 'none';
    if (restoreValue) {
      const item = this._items.find(it => it.value === this._value);
      this._input.value = item ? item.label : '';
    }
  }

  _onInput() {
    const q = this._input.value.toLowerCase().trim();
    this._filtered = q
      ? this._items.filter(it => it.label.toLowerCase().includes(q))
      : this._items.slice();
    this._hlIdx = this._filtered.length > 0 ? 0 : -1;
    this._renderDropdown();
    this._positionDropdown();
  }

  _onKeydown(e) {
    if (!this._open) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter') {
        e.preventDefault();
        this._openDropdown();
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (this._hlIdx < this._filtered.length - 1) this._hlIdx++;
        this._updateHighlight();
        this._scrollToHighlighted();
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (this._hlIdx > 0) this._hlIdx--;
        this._updateHighlight();
        this._scrollToHighlighted();
        break;
      case 'Enter':
        e.preventDefault();
        if (this._hlIdx >= 0 && this._hlIdx < this._filtered.length) {
          this._selectItem(this._filtered[this._hlIdx]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        this._closeDropdown(true);
        this._input.blur();
        break;
      case 'Tab':
        this._closeDropdown(true);
        break;
    }
  }

  _selectItem(item) {
    const changed = this._value !== item.value;
    this._value = item.value;
    this._input.value = item.label;
    this._closeDropdown(false);
    this._input.blur();
    if (changed && this._onChange) this._onChange(item.value, item.label);
  }

  _renderDropdown() {
    this._dropdown.innerHTML = '';
    if (this._filtered.length === 0) {
      const el = document.createElement('div');
      el.className = 'fs-no-match';
      el.textContent = 'No matches';
      this._dropdown.appendChild(el);
      return;
    }
    this._filtered.forEach((item, i) => {
      const el = document.createElement('div');
      el.className = 'fs-option';
      if (item.value === this._value) el.classList.add('fs-selected');
      if (i === this._hlIdx) el.classList.add('fs-highlighted');
      el.textContent = item.label;
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();  // keep focus on input
        this._selectItem(item);
      });
      el.addEventListener('mouseenter', () => {
        this._hlIdx = i;
        this._updateHighlight();
      });
      this._dropdown.appendChild(el);
    });
  }

  _updateHighlight() {
    const children = this._dropdown.querySelectorAll('.fs-option');
    children.forEach((el, i) => {
      el.classList.toggle('fs-highlighted', i === this._hlIdx);
    });
  }

  _scrollToHighlighted() {
    const highlighted = this._dropdown.querySelector('.fs-highlighted');
    if (highlighted) {
      highlighted.scrollIntoView({ block: 'nearest' });
    }
  }
}

// ============================================================================
// 8. GLOBAL NAVIGATION
// ============================================================================

const GlobalNav = {
  // Current state
  root: null,
  serial: null,
  step: null,
  filePath: null,
  fileName: null,
  fileSize: null,

  // Page callbacks (set by each page's script)
  onSidebarSignalClick: null,   // (batch, idx, name, units)
  onFileSelected: null,         // ()
  onSelectionChange: null,      // () — called when root/serial/step/file changes

  // Internal
  _ready: null,
  _roots: [],
  _serials: {},
  _steps: {},
  _files: {},

  // FilterSelect widget instances (set in init)
  _rootFs: null,
  _serialFs: null,
  _stepFs: null,
  _fileFs: null,
  _sidebarEl: null,

  // ── Initialisation ──────────────────────────────────────────────────────

  async init() {
    const rootDiv   = document.getElementById('globalRoot');
    const serialDiv = document.getElementById('globalSerial');
    const stepDiv   = document.getElementById('globalStep');
    const fileDiv   = document.getElementById('globalFile');
    this._sidebarEl = document.getElementById('sidebarContent');

    if (!rootDiv || !serialDiv || !stepDiv || !fileDiv) {
      // Page without file selector (e.g. docs)
      return;
    }

    // Create FilterSelect widgets with cascade onChange handlers
    this._rootFs   = new FilterSelect(rootDiv,   { onChange: (v) => this.onRootChange(v) });
    this._serialFs = new FilterSelect(serialDiv, { onChange: (v) => this.onSerialChange(v) });
    this._stepFs   = new FilterSelect(stepDiv,   { onChange: (v) => this.onStepChange(v) });
    this._fileFs   = new FilterSelect(fileDiv,   { onChange: (v) => this.onFileChange(v) });

    // Persist sidebar scroll position on scroll (debounced)
    if (this._sidebarEl) {
      let _sbTimer = null;
      this._sidebarEl.addEventListener('scroll', () => {
        clearTimeout(_sbTimer);
        _sbTimer = setTimeout(() => URLState.saveSidebar(), 200);
      });
    }

    try {
      // Keep loading overlay visible for the ENTIRE init + restore cycle
      showLoading('Restoring session\u2026');

      // Fetch available roots
      const rootData = await apiFetch('/api/roots');
      this._roots = rootData.roots || [];

      this._rootFs.setItems(this._roots);
      this._rootFs.setDisabled(false);

      // Auto-select if only one root
      if (this._roots.length === 1) {
        this.root = this._roots[0];
        this._rootFs.setValue(this._roots[0]);
        // Pre-load serials for the single root
        const data = await apiFetch(`/api/roots/${encodeURIComponent(this.root)}/serials`);
        this._serials[this.root] = data.serials || [];
        this._serialFs.setItems(this._serials[this.root]);
        this._serialFs.setDisabled(false);
      }

      // Restore file-selector state (root/serial/step/file dropdowns + sidebar)
      const saved = URLState.load();
      if (saved) await this._restoreState(saved);

      hideLoading();
    } catch (err) {
      hideLoading();
      showToast(`Init failed: ${err.message}`, 'error');
    }
  },

  // ── State restoration ───────────────────────────────────────────────────

  async _restoreState(saved) {
    // Restore root
    if (!saved.root || !this._roots.includes(saved.root)) return;

    this.root = saved.root;
    this._rootFs.setValue(saved.root);

    // Load serials for this root (if not already loaded during auto-select)
    if (!this._serials[saved.root]) {
      const data = await apiFetch(`/api/roots/${encodeURIComponent(saved.root)}/serials`);
      this._serials[saved.root] = data.serials || [];
      this._serialFs.setItems(this._serials[saved.root]);
      this._serialFs.setDisabled(false);
    }

    if (!saved.serial || !this._serials[saved.root].includes(saved.serial)) return;

    this.serial = saved.serial;
    this._serialFs.setValue(saved.serial);

    // Load steps
    const stepsData = await apiFetch(
      `/api/roots/${encodeURIComponent(this.root)}/serials/${encodeURIComponent(saved.serial)}/steps`
    );
    this._steps[saved.serial] = stepsData.steps || [];
    this._stepFs.setItems(this._steps[saved.serial].map(s => String(s)));
    this._stepFs.setDisabled(false);

    // Steps come from API as integers but dropdown values are strings
    const savedStep = saved.step;
    const matchedStep = this._steps[saved.serial].find(s => String(s) === String(savedStep));
    if (savedStep == null || matchedStep == null) return;

    this.step = String(matchedStep);
    this._stepFs.setValue(String(matchedStep));

    // Load files
    const filesData = await apiFetch(
      `/api/roots/${encodeURIComponent(this.root)}/serials/${encodeURIComponent(saved.serial)}/steps/${encodeURIComponent(this.step)}/files`
    );
    this._files[this.step] = filesData.files || [];
    this._fileFs.setItems(this._files[this.step].map(f => ({
      value: f.path,
      label: `${f.filename} (${formatBytes(f.size)})`
    })));
    this._fileFs.setDisabled(false);

    if (!saved.filePath) return;
    const fileObj = this._files[this.step].find(f => f.path === saved.filePath);
    if (!fileObj) return;

    this.filePath = saved.filePath;
    this.fileName = saved.fileName || fileObj.filename;
    this.fileSize = saved.fileSize || fileObj.size;
    this._fileFs.setValue(saved.filePath);

    // Load sidebar — isolated so errors here don't break dropdown restoration
    try {
      await this._loadSidebar(false);
    } catch (e) {
      console.warn('Sidebar restore failed:', e);
    }

    // Re-save to ensure format consistency
    URLState.save();
  },

  /**
   * Clear all page-specific sessionStorage keys so that every page
   * starts fresh after a serial / step / file change — not just the
   * page the user happens to be on.
   */
  _clearAllPageState() {
    ['viewerState', 'analysisState', 'cmpState'].forEach(k => sessionStorage.removeItem(k));
  },

  // ── Dropdown handlers (called by FilterSelect onChange) ────────────────

  async onRootChange(val) {
    this.root = val || null;
    this._resetFrom('serial');
    this._clearAllPageState();
    if (this.onSelectionChange) this.onSelectionChange();
    if (!this.root) { URLState.save(); return; }

    try {
      showLoading('Loading serials\u2026');
      const data = await apiFetch(
        `/api/roots/${encodeURIComponent(this.root)}/serials`
      );
      this._serials[this.root] = data.serials || [];
      hideLoading();

      this._serialFs.setItems(this._serials[this.root]);
      this._serialFs.setDisabled(false);
      URLState.save();
    } catch (err) {
      hideLoading();
      showToast(`Failed to load serials: ${err.message}`, 'error');
    }
  },

  async onSerialChange(val) {
    this.serial = val || null;
    this._resetFrom('step');
    this._clearAllPageState();
    if (this.onSelectionChange) this.onSelectionChange();
    if (!this.serial) { URLState.save(); return; }

    try {
      showLoading('Loading steps\u2026');
      const data = await apiFetch(
        `/api/roots/${encodeURIComponent(this.root)}/serials/${encodeURIComponent(this.serial)}/steps`
      );
      this._steps[this.serial] = data.steps || [];
      hideLoading();

      this._stepFs.setItems(this._steps[this.serial].map(s => String(s)));
      this._stepFs.setDisabled(false);
      URLState.save();
    } catch (err) {
      hideLoading();
      showToast(`Failed to load steps: ${err.message}`, 'error');
    }
  },

  async onStepChange(val) {
    this.step = val || null;
    this._resetFrom('file');
    this._clearAllPageState();
    if (this.onSelectionChange) this.onSelectionChange();
    if (!this.step) { URLState.save(); return; }

    try {
      showLoading('Loading files\u2026');
      const data = await apiFetch(
        `/api/roots/${encodeURIComponent(this.root)}/serials/${encodeURIComponent(this.serial)}/steps/${encodeURIComponent(this.step)}/files`
      );
      this._files[this.step] = data.files || [];
      hideLoading();

      this._fileFs.setItems(this._files[this.step].map(f => ({
        value: f.path,
        label: `${f.filename} (${formatBytes(f.size)})`
      })));
      this._fileFs.setDisabled(false);
      URLState.save();
    } catch (err) {
      hideLoading();
      showToast(`Failed to load files: ${err.message}`, 'error');
    }
  },

  async onFileChange(selectedPath) {
    this._clearAllPageState();
    if (this.onSelectionChange) this.onSelectionChange();
    if (!selectedPath) {
      this.filePath = null; this.fileName = null; this.fileSize = null;
      this._clearSidebar();
      URLState.save();
      return;
    }

    const fileObj = (this._files[this.step] || []).find(f => f.path === selectedPath);
    this.filePath = selectedPath;
    this.fileName = fileObj ? fileObj.filename : '';
    this.fileSize = fileObj ? fileObj.size : 0;

    try {
      await this._loadSidebar();
      if (this.onFileSelected) this.onFileSelected();
      URLState.save();
    } catch (err) {
      showToast(`Failed to load sidebar: ${err.message}`, 'error');
    }
  },

  // ── Sidebar ─────────────────────────────────────────────────────────────

  async _loadSidebar(showOverlay = true) {
    if (!this._sidebarEl || !this.filePath) return;

    const encoded = encodePath(this.filePath);

    if (showOverlay) showLoading('Loading signal list\u2026', { fileSize: this.fileSize || 0 });
    let data;
    try {
      data = await apiFetch(`/api/files/${encoded}/batches`);
    } finally {
      if (showOverlay) hideLoading();
    }

    const batchNames = data.batches || [];
    if (batchNames.length === 0) { this._clearSidebar(); return; }

    this._sidebarEl.innerHTML = '';

    for (let bi = 0; bi < batchNames.length; bi++) {
      const batchName = batchNames[bi];

      // Fetch metadata for this batch (signal names, counts, units)
      let meta;
      try {
        meta = await apiFetch(`/api/files/${encoded}/batches/${encodeURIComponent(batchName)}/meta`);
      } catch (_) {
        meta = { signal_count: 0, signal_names: [], units: [] };
      }

      const group = this._buildBatchGroup(batchName, meta, false);
      this._sidebarEl.appendChild(group);
    }

    // Restore expanded/scroll state from previous page visit
    URLState.restoreSidebar();
  },

  _buildBatchGroup(batchName, meta, expanded) {
    const group = document.createElement('div');
    group.className = 'batch-group';

    // Header
    const header = document.createElement('div');
    header.className = 'batch-group-header' + (expanded ? '' : ' collapsed');
    header.innerHTML = `
      <span class="expand-arrow">\u25B6</span>
      <span style="flex:1">${escapeHtml(batchName)}</span>
      <span class="batch-badge">${meta.signal_count || 0}</span>
    `;

    // Signal list
    const list = document.createElement('div');
    list.className = 'batch-group-items';
    list.style.display = expanded ? 'block' : 'none';

    const names = meta.signal_names || [];
    const units = meta.units || [];
    const count = meta.signal_count || names.length;

    // Empty group — show badge "0" but no signal items
    if (count === 0) {
      group.appendChild(header);
      group.appendChild(list);
      return group;
    }

    // Build index array and sort alphabetically by signal name
    const sorted = Array.from({ length: count }, (_, i) => i)
      .sort((a, b) => {
        const na = (names[a] || '').toLowerCase();
        const nb = (names[b] || '').toLowerCase();
        return na < nb ? -1 : na > nb ? 1 : 0;
      });

    for (const i of sorted) {
      const name = names[i] || `signal_${String(i).padStart(3, '0')}`;
      const unit = units[i] || '';

      const item = document.createElement('div');
      item.className = 'signal-item';
      item.setAttribute('data-batch', batchName);
      item.setAttribute('data-idx', i);
      item.innerHTML = `
        <span class="signal-dot"></span>
        <span class="signal-name">${fmtHtml(name)}</span>
        <span class="signal-units">${fmtHtml(unit) || '\u2014'}</span>
      `;
      item.addEventListener('click', () => {
        if (this.onSidebarSignalClick) {
          this.onSidebarSignalClick(batchName, i, name, unit);
        }
      });
      list.appendChild(item);
    }

    // Toggle
    header.addEventListener('click', () => {
      const isOpen = list.style.display !== 'none';
      list.style.display = isOpen ? 'none' : 'block';
      header.classList.toggle('collapsed', isOpen);
      URLState.saveSidebar();
    });

    group.appendChild(header);
    group.appendChild(list);
    return group;
  },

  /**
   * Highlight active signals in the sidebar.
   * @param {Array<{batch:string, idx:number, color:string}>} activeSignals
   */
  updateSidebarHighlights(activeSignals) {
    if (!this._sidebarEl) return;

    // Clear all active states
    this._sidebarEl.querySelectorAll('.signal-item').forEach(el => {
      el.classList.remove('active');
      const dot = el.querySelector('.signal-dot');
      if (dot) dot.style.backgroundColor = '';
    });

    // Apply active state with color for each active signal
    (activeSignals || []).forEach(({ batch, idx, color }) => {
      const item = this._sidebarEl.querySelector(
        `.signal-item[data-batch="${CSS.escape(batch)}"][data-idx="${idx}"]`
      );
      if (item) {
        item.classList.add('active');
        const dot = item.querySelector('.signal-dot');
        if (dot) dot.style.backgroundColor = color;
      }
    });
  },

  _clearSidebar() {
    if (this._sidebarEl) {
      this._sidebarEl.innerHTML =
        '<div class="sidebar-empty">Select a file to browse signals</div>';
    }
  },

  // ── Signal URL builder ──────────────────────────────────────────────────

  /**
   * Build GET URL for a signal.
   * @param {string} batch      batch name
   * @param {number} idx        signal index
   * @param {number} downsample target points (0 = no downsample)
   * @returns {string}
   */
  signalUrl(batch, idx, downsample = 0) {
    const encoded = encodePath(this.filePath);
    let url = `/api/files/${encoded}/batches/${encodeURIComponent(batch)}/signals/${idx}`;
    if (downsample > 0) url += `?downsample=${downsample}`;
    return url;
  },

  // ── Helpers ─────────────────────────────────────────────────────────────

  _resetFrom(level) {
    if (level === 'serial') {
      this.serial = null;
      this._serialFs.reset('Select serial\u2026');
    }
    if (level === 'serial' || level === 'step') {
      this.step = null;
      this._stepFs.reset('Select step\u2026');
    }
    // Always reset file when resetting serial, step or file
    this.filePath = null;
    this.fileName = null;
    this.fileSize = null;
    this._fileFs.reset('Select file\u2026');
    this._clearSidebar();
  }
};

// ============================================================================
// 9. INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Hide loading overlay by default
  hideLoading();
  // Init GlobalNav (stores promise for pages to await)
  GlobalNav._ready = GlobalNav.init();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.toast').forEach(t => t.remove());
  }
});

// Safety net: if any unhandled error / rejection occurs while the loading
// overlay is visible, force-hide it so the UI doesn't freeze permanently.
window.addEventListener('unhandledrejection', () => { hideLoading(); });
window.addEventListener('error', () => { hideLoading(); });

// ============================================================================
// 10. DEBUG EXPORTS
// ============================================================================

window.APP = {
  showToast, showLoading, hideLoading,
  formatNumber, encodePath, decodePath, GlobalNav
};
