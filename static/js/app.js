/**
 * Engineering Time Series Viewer — Core Application Module
 *
 * Provides:
 *   - API helpers (apiFetch, apiPost)
 *   - Path encoding (base64url)
 *   - GlobalNav: serial → step → file cascade + sidebar batch/signal tree
 *   - URL state persistence via sessionStorage
 *   - Loading overlay & toast notifications
 *   - Shared constants (SIGNAL_COLORS, formatNumber, etc.)
 */

// ============================================================================
// 1. CONFIGURATION & CONSTANTS
// ============================================================================

const APP_CONFIG = {
  API_TIMEOUT: 30000,
  DEFAULT_DECIMALS: 4,
  TOAST_DURATION: 4000,
  DEFAULT_DOWNSAMPLE: 2000
};

const SIGNAL_COLORS = [
  '#0077cc', '#e65100', '#2e7d32', '#c62828',
  '#6a1b9a', '#00838f', '#ef6c00', '#37474f'
];

const PLOT_CONFIG = {
  margin:   { l: 56, r: 20, t: 24, b: 56, autoexpand: false },
  marginSm: { l: 48, r: 16, t: 24, b: 52, autoexpand: false },
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
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), APP_CONFIG.API_TIMEOUT);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });

    if (!response.ok) {
      let msg = `HTTP ${response.status}`;
      try { const d = await response.json(); if (d.error) msg = d.error; }
      catch (_) { /* ignore */ }
      throw new Error(msg);
    }

    return await response.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request timeout (exceeded 30 s)');
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

function showLoading(text = 'Loading\u2026') {
  const overlay = document.getElementById('loadingOverlay');
  const textEl  = document.getElementById('loadingText');
  if (overlay) { overlay.style.display = 'flex'; }
  if (textEl)  { textEl.textContent = text; }
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) { overlay.style.display = 'none'; }
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
  }
};

// ============================================================================
// 7. GLOBAL NAVIGATION
// ============================================================================

const GlobalNav = {
  // Current state
  serial: null,
  step: null,
  filePath: null,
  fileName: null,
  fileSize: null,

  // Page callbacks (set by each page's script)
  onSidebarSignalClick: null,   // (batch, idx, name, units)
  onFileSelected: null,         // ()

  // Internal
  _ready: null,
  _serials: [],
  _steps: {},
  _files: {},

  // DOM references (set in init)
  _serialEl: null,
  _stepEl: null,
  _fileEl: null,
  _sidebarEl: null,

  // ── Initialisation ──────────────────────────────────────────────────────

  async init() {
    this._serialEl  = document.getElementById('globalSerial');
    this._stepEl    = document.getElementById('globalStep');
    this._fileEl    = document.getElementById('globalFile');
    this._sidebarEl = document.getElementById('sidebarContent');

    if (!this._serialEl || !this._stepEl || !this._fileEl) {
      // Page without file selector (e.g. docs)
      return;
    }

    try {
      // Keep loading overlay visible for the ENTIRE init + restore cycle
      showLoading('Restoring session\u2026');

      const data = await apiFetch('/api/serials');
      this._serials = data.serials || [];

      this._serialEl.innerHTML = '<option value="">Select serial\u2026</option>';
      this._serials.forEach(s => {
        const o = document.createElement('option');
        o.value = s; o.textContent = s;
        this._serialEl.appendChild(o);
      });

      // Restore file-selector state (serial/step/file dropdowns + sidebar)
      // Pages call their own restorePageState() after this promise resolves
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
    if (!saved.serial || !this._serials.includes(saved.serial)) return;

    this.serial = saved.serial;
    this._serialEl.value = saved.serial;

    // Load steps
    const stepsData = await apiFetch(
      `/api/serials/${encodeURIComponent(saved.serial)}/steps`
    );
    this._steps[saved.serial] = stepsData.steps || [];
    this._populateSelect(this._stepEl, this._steps[saved.serial], 'Select step\u2026');

    // Steps come from API as integers but dropdown values are strings — use loose comparison
    const savedStep = saved.step;
    const matchedStep = this._steps[saved.serial].find(s => String(s) === String(savedStep));
    if (savedStep == null || matchedStep == null) return;

    this.step = String(matchedStep);
    this._stepEl.value = String(matchedStep);
    this._stepEl.disabled = false;

    // Load files — use this.step (normalized string) for both URL and cache key
    const filesData = await apiFetch(
      `/api/serials/${encodeURIComponent(saved.serial)}/steps/${encodeURIComponent(this.step)}/files`
    );
    this._files[this.step] = filesData.files || [];
    this._populateFileSelect(this._files[this.step]);

    if (!saved.filePath) return;
    const fileObj = this._files[this.step].find(f => f.path === saved.filePath);
    if (!fileObj) return;

    this.filePath = saved.filePath;
    this.fileName = saved.fileName || fileObj.filename;
    this.fileSize = saved.fileSize || fileObj.size;
    this._fileEl.value = saved.filePath;
    this._fileEl.disabled = false;

    // Load sidebar — isolated so errors here don't break dropdown restoration
    try {
      await this._loadSidebar(false);
    } catch (e) {
      console.warn('Sidebar restore failed:', e);
    }

    // Re-save to ensure format consistency
    URLState.save();
  },

  // ── Dropdown handlers (called from inline onchange in base.html) ──────

  async onSerialChange() {
    this.serial = this._serialEl.value;
    this._resetFrom('step');
    if (!this.serial) { URLState.save(); return; }

    try {
      showLoading('Loading steps\u2026');
      const data = await apiFetch(
        `/api/serials/${encodeURIComponent(this.serial)}/steps`
      );
      this._steps[this.serial] = data.steps || [];
      hideLoading();

      this._populateSelect(this._stepEl, this._steps[this.serial], 'Select step\u2026');
      this._stepEl.disabled = false;
      URLState.save();
    } catch (err) {
      hideLoading();
      showToast(`Failed to load steps: ${err.message}`, 'error');
    }
  },

  async onStepChange() {
    this.step = this._stepEl.value;
    this._resetFrom('file');
    if (!this.step) { URLState.save(); return; }

    try {
      showLoading('Loading files\u2026');
      const data = await apiFetch(
        `/api/serials/${encodeURIComponent(this.serial)}/steps/${encodeURIComponent(this.step)}/files`
      );
      this._files[this.step] = data.files || [];
      hideLoading();

      this._populateFileSelect(this._files[this.step]);
      this._fileEl.disabled = false;
      URLState.save();
    } catch (err) {
      hideLoading();
      showToast(`Failed to load files: ${err.message}`, 'error');
    }
  },

  async onFileChange() {
    const selectedPath = this._fileEl.value;
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

    if (showOverlay) showLoading('Loading signal list\u2026');
    const data = await apiFetch(`/api/files/${encoded}/batches`);
    if (showOverlay) hideLoading();

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

      const expanded = (bi === 0);
      const group = this._buildBatchGroup(batchName, meta, expanded);
      this._sidebarEl.appendChild(group);
    }
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
    for (let i = 0; i < (meta.signal_count || names.length); i++) {
      const name = names[i] || `signal_${String(i).padStart(3, '0')}`;
      const unit = units[i] || '';

      const item = document.createElement('div');
      item.className = 'signal-item';
      item.setAttribute('data-batch', batchName);
      item.setAttribute('data-idx', i);
      item.innerHTML = `
        <span class="signal-dot"></span>
        <span class="signal-name">${escapeHtml(name)}</span>
        <span class="signal-units">${escapeHtml(unit) || '\u2014'}</span>
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

  _populateSelect(el, items, placeholder) {
    el.innerHTML = `<option value="">${placeholder}</option>`;
    items.forEach(item => {
      const o = document.createElement('option');
      o.value = item; o.textContent = item;
      el.appendChild(o);
    });
    el.disabled = false;
  },

  _populateFileSelect(files) {
    this._fileEl.innerHTML = '<option value="">Select file\u2026</option>';
    files.forEach(f => {
      const o = document.createElement('option');
      o.value = f.path;
      o.textContent = `${f.filename} (${formatBytes(f.size)})`;
      this._fileEl.appendChild(o);
    });
    this._fileEl.disabled = false;
  },

  _resetFrom(level) {
    if (level === 'step') {
      this.step = null;
      this._stepEl.value = '';
      this._stepEl.innerHTML = '<option value="">Select step\u2026</option>';
      this._stepEl.disabled = true;
    }
    // Always reset file when resetting step or file
    this.filePath = null;
    this.fileName = null;
    this.fileSize = null;
    this._fileEl.value = '';
    this._fileEl.innerHTML = '<option value="">Select file\u2026</option>';
    this._fileEl.disabled = true;
    this._clearSidebar();
  }
};

// ============================================================================
// 8. INITIALISATION
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

// ============================================================================
// 9. DEBUG EXPORTS
// ============================================================================

window.APP = {
  showToast, showLoading, hideLoading,
  formatNumber, encodePath, decodePath, GlobalNav
};
