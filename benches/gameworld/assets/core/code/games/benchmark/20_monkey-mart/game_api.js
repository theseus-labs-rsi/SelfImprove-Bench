(function(){
  const GAME_ID = '20_monkey-mart';
  const MONEY_KEYS = ['money', 'cash', 'coins', 'coin', 'gold', 'wallet', 'bank', 'currency', 'funds'];
  const MONEY_KEY_PATTERN = MONEY_KEYS.map(function(key) {
    return key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }).join('|');
  const KEY_VALUE_REGEX = new RegExp('(?:^|[^a-z0-9_])((?:' + MONEY_KEY_PATTERN + '))\\s*[:=]\\s*(-?\\d+(?:\\.\\d+)?)', 'ig');
  const BRACKET_KEY_REGEX = new RegExp('\\[\\s*[\'"]?((?:' + MONEY_KEY_PATTERN + '))[\'"]?\\s*\\]\\s*[:=]?\\s*(-?\\d+(?:\\.\\d+)?)', 'ig');
  const STARTING_MONEY = 35;
  const MIN_PLAUSIBLE_MONEY_BEFORE_SOURCE_LOCK = 10;
  const MIN_SCAN_INTERVAL_MS = 100;
  const MAX_BYTES_TO_SCAN = 200000;
  // Optional: fill with known-good candidate ids to force faster detection.
  // Format: "<sourcePath>::<fieldPath>", example:
  // "localStorage:monkeymart_save::player.wallet.money"
  const PREFERRED_MONEY_CANDIDATE_IDS = [
    'localStorage:monkeymart_config::coins'
  ];
  const capabilities = {
    supports_seed: true,
    supports_level_select: false,
    supports_difficulty: false,
    supports_inplace_reset: false,
    supports_reload_reset: true,
    supports_pause_detection: false,
    supports_menu_detection: false,
    provides_actionable_flag: true
  };
  const session = {
    seed: 42,
    difficulty: null,
    episodeStartMs: Date.now()
  };
  const runtime = {
    resetCount: 0,
    gameplayStartMs: null,
    lastResetMethod: null
  };

  let lastMoney = null;
  let moneyBaseline = null;
  let totalMoneyEarned = 0;
  let trackedMoneySource = null;
  let trackedMoneyCandidateId = null;
  const sourceObservations = Object.create(null);
  let lastReportedSource = null;
  let lastReportedLocked = null;
  let lastScanMs = 0;
  let cachedMoneyInfo = {
    value: null,
    sourcePath: null,
    scannedFiles: 0,
    scanStatus: 'unavailable',
    candidateSummaries: []
  };

  function resetMoneyTracking(){
    lastMoney = null;
    moneyBaseline = null;
    totalMoneyEarned = 0;
    trackedMoneySource = null;
    trackedMoneyCandidateId = null;
    lastReportedSource = null;
    lastReportedLocked = null;
    lastScanMs = 0;
    cachedMoneyInfo = {
      value: null,
      sourcePath: null,
      scannedFiles: 0,
      scanStatus: 'unavailable',
      candidateSummaries: []
    };
    Object.keys(sourceObservations).forEach(function(key){
      delete sourceObservations[key];
    });
  }

  function normalizeSeed(value){
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    return (numeric >>> 0);
  }

  function normalizeOptions(options){
    const opts = options || {};
    return {
      seed: normalizeSeed(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : (typeof opts.level === 'number' && Number.isFinite(opts.level))
            ? Math.trunc(opts.level)
            : String(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty)
    };
  }

  function getCurrentSeed(){
    if (typeof window !== 'undefined' && typeof window.__getDeterministicSeed === 'function') {
      const seed = safeNumber(window.__getDeterministicSeed());
      return seed === null ? 42 : (Math.trunc(seed) >>> 0);
    }
    return 42;
  }

  function applySeed(seed){
    if (typeof window === 'undefined') return null;
    if (typeof window.__setDeterministicSeed === 'function') {
      const applied = safeNumber(window.__setDeterministicSeed(seed));
      return applied === null ? null : (Math.trunc(applied) >>> 0);
    }
    if (typeof window.__resetRandom === 'function') {
      const applied = safeNumber(window.__resetRandom(seed));
      return applied === null ? null : (Math.trunc(applied) >>> 0);
    }
    return null;
  }

  function beginEpisode(options){
    const accepted = normalizeOptions(options);
    const notes = [];
    let appliedSeed = accepted.seed;

    if (accepted.level !== null) notes.push('level_not_supported');
    if (accepted.difficulty !== null) notes.push('difficulty_not_supported');

    if (accepted.seed !== null) {
      appliedSeed = applySeed(accepted.seed);
      if (appliedSeed === null) {
        notes.push('seed_not_supported');
      }
    } else {
      appliedSeed = applySeed(getCurrentSeed());
      if (appliedSeed === null) {
        appliedSeed = getCurrentSeed();
      }
    }

    session.seed = appliedSeed === null ? getCurrentSeed() : appliedSeed;
    session.difficulty = null;
    session.episodeStartMs = Date.now();
    runtime.resetCount += 1;
    runtime.gameplayStartMs = null;
    resetMoneyTracking();
    return {
      accepted: accepted,
      applied: {
        seed: session.seed,
        level: null,
        difficulty: null
      },
      notes: notes
    };
  }

  function safeNumber(value){
    return (typeof value === 'number' && Number.isFinite(value)) ? value : null;
  }

  function getModule(){
    if (typeof window === 'undefined') return null;
    return window.Module || null;
  }

  function getCanvasInfo(){
    if (typeof document === 'undefined') return { width: null, height: null };
    const canvas = document.getElementById('canvas');
    if (!canvas) return { width: null, height: null };
    return { width: safeNumber(canvas.width), height: safeNumber(canvas.height) };
  }

  function getFs(){
    if (typeof window === 'undefined') return null;
    if (window.Module && window.Module.FS) return window.Module.FS;
    if (window.FS) return window.FS;
    return null;
  }

  function getPersistentRoot(){
    if (typeof window !== 'undefined' && window.DMSYS && typeof window.DMSYS.GetUserPersistentDataRoot === 'function') {
      return window.DMSYS.GetUserPersistentDataRoot();
    }
    return '/data';
  }

  function isDir(fs, mode){
    if (fs && typeof fs.isDir === 'function') return fs.isDir(mode);
    return (mode & 0x4000) === 0x4000;
  }

  function listFiles(fs, dir, depth, out){
    if (!fs || depth < 0) return;
    let entries = [];
    try {
      entries = fs.readdir(dir);
    } catch (e) {
      return;
    }
    entries.forEach(function(name){
      if (name === '.' || name === '..') return;
      const path = dir.replace(/\/$/, '') + '/' + name;
      let stat = null;
      try {
        stat = fs.stat(path);
      } catch (e) {
        return;
      }
      if (stat && isDir(fs, stat.mode)) {
        listFiles(fs, path, depth - 1, out);
      } else {
        out.push(path);
      }
    });
  }

  function binaryToString(data){
    if (!data || typeof data.length !== 'number') return null;
    const limit = Math.min(data.length, MAX_BYTES_TO_SCAN);
    let result = '';
    for (let i = 0; i < limit; i += 1) {
      result += String.fromCharCode(data[i]);
    }
    return result;
  }

  function readFileAsText(fs, path){
    try {
      return fs.readFile(path, { encoding: 'utf8' });
    } catch (e) {}
    try {
      const data = fs.readFile(path);
      return binaryToString(data);
    } catch (e) {}
    return null;
  }

  function normalizeMoneyValue(value){
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  function keyLooksMoneyRelated(key){
    return getKeyHintScore(key) > 0;
  }

  function getKeyHintScore(key){
    if (!key) return 0;
    const lower = String(key).toLowerCase();
    let score = 0;

    if (lower === 'balance') score += 90;
    if (MONEY_KEYS.indexOf(lower) !== -1) score += 90;

    MONEY_KEYS.forEach(function(token){
      if (lower.indexOf(token) !== -1) score += 18;
    });

    if (lower.indexOf('balance') !== -1) score += 25;
    if (lower.indexOf('earn') !== -1 || lower.indexOf('income') !== -1 || lower.indexOf('profit') !== -1) score -= 10;
    if (lower.indexOf('total') !== -1) score -= 8;
    if (lower.indexOf('ad') !== -1 || lower.indexOf('reward') !== -1 || lower.indexOf('bonus') !== -1) score -= 18;
    if (lower.indexOf('tutorial') !== -1 || lower.indexOf('guide') !== -1 || lower.indexOf('step') !== -1) score -= 15;
    if (lower.indexOf('config') !== -1 || lower.indexOf('setting') !== -1 || lower.indexOf('option') !== -1) score -= 10;
    if (lower.indexOf('price') !== -1 || lower.indexOf('cost') !== -1 || lower.indexOf('unlock') !== -1) score -= 10;
    if (lower === 'coins' || lower === 'coin' || lower.endsWith('.coins') || lower.endsWith('.coin')) score -= 60;
    if (lower === 'cash2' || lower.endsWith('.cash2')) score -= 35;
    if (lower === 'cash' || lower.endsWith('.cash')) score += 14;
    if (lower === 'money' || lower.endsWith('.money') || lower === 'balance' || lower.endsWith('.balance')) score += 20;

    return score;
  }

  function isWeakMoneyField(keyPath){
    if (!keyPath) return false;
    const lower = String(keyPath).toLowerCase();
    return (
      lower === 'coin' ||
      lower === 'coins' ||
      lower.endsWith('.coin') ||
      lower.endsWith('.coins') ||
      lower === 'cash2' ||
      lower.endsWith('.cash2')
    );
  }

  function isStrongMoneyField(keyPath){
    if (!keyPath) return false;
    const lower = String(keyPath).toLowerCase();
    return (
      lower === 'money' ||
      lower === 'cash' ||
      lower === 'wallet' ||
      lower === 'balance' ||
      lower.endsWith('.money') ||
      lower.endsWith('.cash') ||
      lower.endsWith('.wallet') ||
      lower.endsWith('.balance') ||
      lower.endsWith('.currency') ||
      lower.endsWith('.funds')
    );
  }

  function sourceLooksMoneySpecific(sourcePath){
    return getSourceHintScore(sourcePath) >= 12;
  }

  function getSourceHintScore(sourcePath){
    if (!sourcePath) return 0;
    const lower = String(sourcePath).toLowerCase();
    let score = 0;

    if (
      lower.indexOf('money') !== -1 ||
      lower.indexOf('cash') !== -1 ||
      lower.indexOf('wallet') !== -1 ||
      lower.indexOf('currency') !== -1 ||
      lower.indexOf('fund') !== -1 ||
      lower.indexOf('bank') !== -1 ||
      lower.indexOf('coin') !== -1 ||
      lower.indexOf('gold') !== -1 ||
      lower.indexOf('balance') !== -1
    ) {
      score += 20;
    }
    if (lower.indexOf('save') !== -1 || lower.indexOf('profile') !== -1 || lower.indexOf('persist') !== -1) score += 8;
    if (lower.indexOf('config') !== -1 || lower.indexOf('setting') !== -1 || lower.indexOf('option') !== -1) score -= 15;
    if (lower.indexOf('analytics') !== -1 || lower.indexOf('telemetry') !== -1) score -= 20;

    return score;
  }

  function candidateIdFromParts(sourcePath, keyPath){
    const source = sourcePath || '(unknown-source)';
    const key = keyPath || '(root)';
    return source + '::' + key;
  }

  function getCandidateId(candidate){
    if (!candidate) return null;
    return candidateIdFromParts(candidate.sourcePath, candidate.keyPath);
  }

  function getSourceObservation(candidateId){
    if (!candidateId) return null;
    if (!sourceObservations[candidateId]) {
      sourceObservations[candidateId] = {
        seen: 0,
        changes: 0,
        lastValue: null,
        minValue: null,
        maxValue: null
      };
    }
    return sourceObservations[candidateId];
  }

  function recordSourceObservation(candidate){
    const candidateId = getCandidateId(candidate);
    if (!candidateId) return;
    const obs = getSourceObservation(candidateId);
    if (!obs) return;
    obs.seen += 1;
    if (typeof obs.lastValue === 'number' && candidate.value !== obs.lastValue) {
      obs.changes += 1;
    }
    obs.lastValue = candidate.value;
    obs.minValue = (obs.minValue === null) ? candidate.value : Math.min(obs.minValue, candidate.value);
    obs.maxValue = (obs.maxValue === null) ? candidate.value : Math.max(obs.maxValue, candidate.value);
  }

  function isLockReady(candidate){
    const candidateId = getCandidateId(candidate);
    if (!candidateId) return false;
    const obs = getSourceObservation(candidateId);
    if (!obs) return false;
    const hint = (candidate && typeof candidate.hintScore === 'number') ? candidate.hintScore : 0;
    const value = candidate && typeof candidate.value === 'number' ? candidate.value : null;
    const weakField = isWeakMoneyField(candidate ? candidate.keyPath : null);
    const strongField = isStrongMoneyField(candidate ? candidate.keyPath : null);

    if (weakField && value !== null && value < (STARTING_MONEY - 5)) return false;
    if (strongField && obs.changes >= 1 && hint >= 10) return true;
    if (obs.changes >= 2 && hint >= 10) return true;
    if (obs.seen >= 4 && hint >= 70) return true;
    return false;
  }

  function shouldUseCandidateValue(candidate){
    if (!candidate || typeof candidate.value !== 'number' || !Number.isFinite(candidate.value)) return false;
    const candidateId = getCandidateId(candidate);
    const obs = getSourceObservation(candidateId) || { seen: 0, changes: 0 };
    const hint = (candidate && typeof candidate.hintScore === 'number') ? candidate.hintScore : 0;
    const weakField = isWeakMoneyField(candidate.keyPath);
    const strongField = isStrongMoneyField(candidate.keyPath);

    if (weakField && candidate.value < (STARTING_MONEY - 5)) return false;
    if (candidate.value >= (STARTING_MONEY - 5)) return true;
    if (trackedMoneyCandidateId && candidateId === trackedMoneyCandidateId) return true;
    if (strongField && obs.changes >= 1 && candidate.value >= MIN_PLAUSIBLE_MONEY_BEFORE_SOURCE_LOCK) return true;
    if (hint >= 85 && obs.changes >= 1) return true;

    return false;
  }

  function collectMoneyCandidatesFromObject(obj, sourcePath, sourceType, out, visited, parentPath){
    if (!obj || typeof obj !== 'object') return;
    const seen = visited || new Set();
    if (seen.has(obj)) return;
    seen.add(obj);

    const basePath = parentPath || '';
    if (Array.isArray(obj)) {
      for (let i = 0; i < obj.length; i += 1) {
        const child = obj[i];
        const childPath = basePath ? (basePath + '[' + i + ']') : ('[' + i + ']');
        if (child && typeof child === 'object') {
          collectMoneyCandidatesFromObject(child, sourcePath, sourceType, out, seen, childPath);
        }
      }
      return;
    }

    Object.keys(obj).forEach(function(key){
      const value = obj[key];
      const keyPath = basePath ? (basePath + '.' + key) : key;
      const numeric = normalizeMoneyValue(value);
      if (numeric !== null) {
        const hintScore = getKeyHintScore(key) + getKeyHintScore(keyPath) + getSourceHintScore(sourcePath);
        if (hintScore > 0 || numeric >= (STARTING_MONEY - 5)) {
          out.push({
            value: numeric,
            sourcePath: sourcePath,
            sourceType: sourceType,
            keyPath: keyPath,
            hintScore: hintScore
          });
        }
      }
      if (value && typeof value === 'object') {
        collectMoneyCandidatesFromObject(value, sourcePath, sourceType, out, seen, keyPath);
      }
    });
  }

  function collectMoneyCandidatesFromText(text, sourcePath, sourceType, out){
    if (!text || typeof text !== 'string') return;
    let match;

    KEY_VALUE_REGEX.lastIndex = 0;
    while ((match = KEY_VALUE_REGEX.exec(text)) !== null) {
      const key = match[1];
      const numeric = normalizeMoneyValue(match[2]);
      if (numeric === null) continue;
      out.push({
        value: numeric,
        sourcePath: sourcePath,
        sourceType: sourceType,
        keyPath: 'text.' + key,
        hintScore: getKeyHintScore(key) + getSourceHintScore(sourcePath)
      });
    }

    BRACKET_KEY_REGEX.lastIndex = 0;
    while ((match = BRACKET_KEY_REGEX.exec(text)) !== null) {
      const key = match[1];
      const numeric = normalizeMoneyValue(match[2]);
      if (numeric === null) continue;
      out.push({
        value: numeric,
        sourcePath: sourcePath,
        sourceType: sourceType,
        keyPath: 'text[' + key + ']',
        hintScore: getKeyHintScore(key) + getSourceHintScore(sourcePath)
      });
    }
  }

  function collectCandidatesFromSerializedValue(value, sourcePath, sourceType, out){
    if (typeof value !== 'string' || !value) return;
    const trimmed = value.trim();
    if (trimmed && (trimmed[0] === '{' || trimmed[0] === '[')) {
      try {
        const parsed = JSON.parse(trimmed);
        collectMoneyCandidatesFromObject(parsed, sourcePath, sourceType, out, new Set(), '');
      } catch (e) {}
    }
    collectMoneyCandidatesFromText(value, sourcePath, sourceType, out);
  }

  function scanWebStorageMoney(){
    if (typeof window === 'undefined') {
      return {
        candidates: [],
        scannedFiles: 0,
        scanStatus: 'storage_unavailable'
      };
    }

    const storageSpecs = [
      { name: 'localStorage', storage: null },
      { name: 'sessionStorage', storage: null }
    ];

    try { storageSpecs[0].storage = window.localStorage; } catch (e) {}
    try { storageSpecs[1].storage = window.sessionStorage; } catch (e) {}

    const candidates = [];
    let scannedEntries = 0;

    storageSpecs.forEach(function(spec){
      const storage = spec.storage;
      if (!storage || typeof storage.length !== 'number') return;
      for (let i = 0; i < storage.length; i += 1) {
        let key = null;
        let value = null;
        try {
          key = storage.key(i);
          value = key !== null ? storage.getItem(key) : null;
        } catch (e) {
          continue;
        }
        scannedEntries += 1;
        if (!key || typeof value !== 'string') continue;

        const sourcePath = spec.name + ':' + key;
        collectCandidatesFromSerializedValue(value, sourcePath, 'storage', candidates);

        if (keyLooksMoneyRelated(key)) {
          const directNumeric = normalizeMoneyValue(value);
          if (directNumeric !== null) {
            candidates.push({
              value: directNumeric,
              sourcePath: sourcePath,
              sourceType: 'storage',
              keyPath: '(value)',
              hintScore: getKeyHintScore(key) + getSourceHintScore(sourcePath)
            });
          }
        }
      }
    });

    return {
      candidates: candidates,
      scannedFiles: scannedEntries,
      scanStatus: scannedEntries ? 'storage_scanned' : 'storage_empty'
    };
  }

  function scanFileSystemMoney(fs, root){
    const files = [];
    listFiles(fs, root, 3, files);
    const candidates = [];

    files.forEach(function(path){
      const text = readFileAsText(fs, path);
      if (!text) return;
      collectCandidatesFromSerializedValue(text, path, 'fs', candidates);
    });

    return {
      candidates: candidates,
      scannedFiles: files.length,
      scanStatus: files.length ? 'scanned' : 'no_files'
    };
  }

  function scoreCandidate(candidate, referenceMoney){
    const candidateId = getCandidateId(candidate);
    const obs = getSourceObservation(candidateId) || { seen: 0, changes: 0 };
    const hint = (candidate && typeof candidate.hintScore === 'number') ? candidate.hintScore : 0;
    const distance = Math.abs(candidate.value - referenceMoney);
    let score = 0;

    score += hint;
    score += getSourceHintScore(candidate.sourcePath);
    score += Math.min(obs.changes, 6) * 7;
    score += Math.min(obs.seen, 10) * 0.5;
    if (candidate.sourceType === 'fs') score += 2;
    if (candidate.sourceType === 'storage') score += 1;

    score += Math.max(0, 25 - (distance * 1.8));
    if (candidateId && PREFERRED_MONEY_CANDIDATE_IDS.indexOf(candidateId) !== -1) score += 200;

    if (candidate.value < MIN_PLAUSIBLE_MONEY_BEFORE_SOURCE_LOCK && hint < 40) score -= 30;
    if (candidate.sourcePath && String(candidate.sourcePath).toLowerCase().indexOf('config') !== -1 && hint < 70) score -= 20;
    if (candidate.keyPath) {
      const pathLower = String(candidate.keyPath).toLowerCase();
      if (pathLower.indexOf('ad') !== -1 || pathLower.indexOf('reward') !== -1 || pathLower.indexOf('bonus') !== -1) score -= 18;
      if (pathLower.indexOf('total') !== -1 || pathLower.indexOf('earned') !== -1 || pathLower.indexOf('income') !== -1 || pathLower.indexOf('profit') !== -1) score -= 10;
      if (pathLower.indexOf('tutorial') !== -1 || pathLower.indexOf('step') !== -1) score -= 15;
    }

    return score;
  }

  function selectMoneyCandidate(candidates){
    if (!Array.isArray(candidates) || candidates.length === 0) return null;

    const referenceMoney = (lastMoney !== null && typeof lastMoney === 'number') ? lastMoney : STARTING_MONEY;
    let candidatePool = candidates.filter(function(candidate){
      return candidate && typeof candidate.value === 'number' && Number.isFinite(candidate.value);
    });

    if (PREFERRED_MONEY_CANDIDATE_IDS.length > 0) {
      const preferred = candidatePool.find(function(candidate){
        const candidateId = getCandidateId(candidate);
        return candidateId && PREFERRED_MONEY_CANDIDATE_IDS.indexOf(candidateId) !== -1;
      });
      if (preferred) {
        recordSourceObservation(preferred);
        return preferred;
      }
    }

    candidatePool.forEach(recordSourceObservation);

    if (trackedMoneyCandidateId) {
      const trackedCandidate = candidatePool.find(function(candidate){
        return getCandidateId(candidate) === trackedMoneyCandidateId;
      }) || null;

      if (trackedCandidate) {
        let bestAlternative = null;
        let bestAlternativeScore = -Number.MAX_VALUE;
        candidatePool.forEach(function(candidate){
          const candidateId = getCandidateId(candidate);
          if (candidateId === trackedMoneyCandidateId) return;
          const candidateScore = scoreCandidate(candidate, referenceMoney);
          if (candidateScore > bestAlternativeScore) {
            bestAlternative = candidate;
            bestAlternativeScore = candidateScore;
          }
        });

        const trackedScore = scoreCandidate(trackedCandidate, referenceMoney) + 12;
        if (bestAlternative && bestAlternativeScore > trackedScore) return bestAlternative;
        return trackedCandidate;
      }
    }

    // Before source lock, filter obviously wrong tiny counters.
    candidatePool = candidatePool.filter(function(candidate){
      const hint = (candidate && typeof candidate.hintScore === 'number') ? candidate.hintScore : 0;
      if (hint >= 15) return true;
      if (sourceLooksMoneySpecific(candidate.sourcePath)) return true;
      if (candidate.value >= (STARTING_MONEY - 5)) return true;
      return false;
    });
    if (candidatePool.length === 0) return null;

    let best = null;
    let bestScore = -Number.MAX_VALUE;
    let bestDistance = Number.POSITIVE_INFINITY;
    candidatePool.forEach(function(candidate){
      const score = scoreCandidate(candidate, referenceMoney);
      const distance = Math.abs(candidate.value - referenceMoney);
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
        bestDistance = distance;
        return;
      }
      if (score === bestScore && distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    });

    return best;
  }

  function buildCandidateSummaries(candidates, selectedCandidateId, limit){
    if (!Array.isArray(candidates) || candidates.length === 0) return [];
    const maxItems = typeof limit === 'number' && limit > 0 ? limit : 8;
    const referenceMoney = (lastMoney !== null && typeof lastMoney === 'number') ? lastMoney : STARTING_MONEY;
    const scored = [];

    candidates.forEach(function(candidate){
      if (!candidate || typeof candidate.value !== 'number' || !Number.isFinite(candidate.value)) return;
      const candidateId = getCandidateId(candidate);
      const obs = getSourceObservation(candidateId) || { seen: 0, changes: 0 };
      const distance = Math.abs(candidate.value - referenceMoney);
      const priority = scoreCandidate(candidate, referenceMoney) - distance * 0.2;
      scored.push({
        candidateId: candidateId,
        sourcePath: candidate.sourcePath,
        keyPath: candidate.keyPath || null,
        value: candidate.value,
        sourceType: candidate.sourceType,
        seen: obs.seen || 0,
        changes: obs.changes || 0,
        selected: !!(selectedCandidateId && candidateId === selectedCandidateId),
        priority: priority
      });
    });

    scored.sort(function(a, b){
      if (b.priority !== a.priority) return b.priority - a.priority;
      if (a.value !== b.value) return b.value - a.value;
      return String(a.sourcePath).localeCompare(String(b.sourcePath));
    });

    return scored.slice(0, maxItems).map(function(item){
      return (
        item.sourcePath +
        (item.keyPath ? ('::' + item.keyPath) : '') +
        '=' + item.value +
        ' [' + item.sourceType +
        ',seen:' + item.seen +
        ',chg:' + item.changes +
        (item.selected ? ',selected' : '') +
        ']'
      );
    });
  }

  function maybeLogMoneyDetection(info){
    if (typeof console === 'undefined' || !console || typeof console.info !== 'function') return;
    const source = info && info.sourcePath ? info.sourcePath : null;
    const field = info && info.keyPath ? info.keyPath : null;
    const candidateId = (source || field) ? candidateIdFromParts(source, field) : null;
    const locked = !!(trackedMoneyCandidateId && candidateId && candidateId === trackedMoneyCandidateId);
    if (candidateId === lastReportedSource && locked === lastReportedLocked) return;
    lastReportedSource = candidateId;
    lastReportedLocked = locked;
    try {
      console.info('[GameAPI][MonkeyMartMoney]', {
        source: source,
        field: field,
        locked: locked,
        candidateId: candidateId,
        value: info ? info.value : null,
        scanStatus: info ? info.scanStatus : null,
        candidates: info && Array.isArray(info.candidateSummaries) ? info.candidateSummaries : []
      });
    } catch (e) {}
  }

  function scanPersistentMoney(){
    const now = Date.now();
    if (now - lastScanMs < MIN_SCAN_INTERVAL_MS) return cachedMoneyInfo;
    lastScanMs = now;

    const storageMoney = scanWebStorageMoney();
    let fileMoney = {
      candidates: [],
      scannedFiles: 0,
      scanStatus: 'fs_unavailable'
    };
    const fs = getFs();
    if (fs && typeof fs.readdir === 'function') {
      const root = getPersistentRoot();
      fileMoney = scanFileSystemMoney(fs, root);
    }

    const allCandidates = []
      .concat(storageMoney.candidates || [])
      .concat(fileMoney.candidates || []);
    const selectedCandidate = selectMoneyCandidate(allCandidates);
    const selectedCandidateId = getCandidateId(selectedCandidate);
    const canLock = !!(selectedCandidate && isLockReady(selectedCandidate));
    if (canLock) {
      trackedMoneyCandidateId = selectedCandidateId;
      trackedMoneySource = selectedCandidate.sourcePath || null;
    }
    const isSelectedLocked = !!(
      selectedCandidate &&
      trackedMoneyCandidateId &&
      selectedCandidateId &&
      trackedMoneyCandidateId === selectedCandidateId
    );

    cachedMoneyInfo = {
      value: selectedCandidate ? selectedCandidate.value : null,
      sourcePath: selectedCandidate ? selectedCandidate.sourcePath : null,
      keyPath: selectedCandidate ? (selectedCandidate.keyPath || null) : null,
      candidateId: selectedCandidateId,
      scannedFiles: (storageMoney.scannedFiles || 0) + (fileMoney.scannedFiles || 0),
      candidateSummaries: buildCandidateSummaries(
        allCandidates,
        selectedCandidateId,
        12
      ),
      scanStatus: selectedCandidate
        ? (selectedCandidate.sourceType + (isSelectedLocked ? '_selected_locked' : '_selected_unlocked'))
        : (storageMoney.scanStatus + '+' + fileMoney.scanStatus)
    };
    maybeLogMoneyDetection(cachedMoneyInfo);
    return cachedMoneyInfo;
  }

  function updateMoneyTotals(currentMoney){
    if (currentMoney === null || typeof currentMoney !== 'number') return;
    if (lastMoney === null) {
      lastMoney = currentMoney;
      if (moneyBaseline === null) moneyBaseline = currentMoney;
      return;
    }
    if (currentMoney > lastMoney) {
      const delta = currentMoney - lastMoney;
      totalMoneyEarned += delta;
    }
    lastMoney = currentMoney;
  }

  function buildLoadingState(nowMs){
    const canvasInfo = getCanvasInfo();
    return {
      schemaVersion: '2.0',
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: nowMs,
      gameTimeMs: null,
      status: 'loading',
      is_actionable: false,
      terminal: {
        isTerminal: false,
        outcome: null,
        reason: null
      },
      game_state: {
        score: null,
        level: null,
        player: null,
        environment: null,
        completion_progress: null,
        money: null,
        money_total_earned: null,
        money_scan_status: 'unavailable'
      },
      metrics: {
        primary_score: null,
        attempts: runtime.resetCount
      },
      debug: {
        ready: false,
        canvas_width: canvasInfo.width,
        canvas_height: canvasInfo.height,
        money_source_path: null,
        money_field_path: null,
        money_candidate_id: null,
        money_source_locked: false,
        money_scanned_files: 0,
        last_reset_method: runtime.lastResetMethod
      },
      raw: null
    };
  }

  window.gameAPI = {
    version: '2.0',
    capabilities: capabilities,
    init: async function(config){
      const episode = beginEpisode(config || {});
      runtime.lastResetMethod = 'init';
      return {
        ok: true,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes
      };
    },
    getState: function(){
      const nowMs = Date.now();
      const module = getModule();
      if (!module) {
        return buildLoadingState(nowMs);
      }
      const canvasInfo = getCanvasInfo();
      const isRunning = module ? !!module.calledRun : false;
      const expectedDownloads = module ? module.expectedDataFileDownloads : null;
      const finishedDownloads = module ? module.finishedDataFileDownloads : null;
      const downloadsReady = (typeof expectedDownloads === 'number' && typeof finishedDownloads === 'number')
        ? finishedDownloads >= expectedDownloads
        : true;
      const ready = module ? (!!module.calledRun && downloadsReady) : false;
      if (!ready) {
        return buildLoadingState(nowMs);
      }

      const baseStatus = isRunning ? 'playing' : 'menu';
      if (baseStatus === 'playing' && runtime.gameplayStartMs === null) {
        runtime.gameplayStartMs = nowMs;
      }

      const moneyInfo = scanPersistentMoney();
      let currentMoney = safeNumber(moneyInfo.value);
      if (currentMoney === null && lastMoney !== null) {
        currentMoney = lastMoney;
      }
      if (currentMoney === null && ready) {
        currentMoney = STARTING_MONEY;
      }
      updateMoneyTotals(currentMoney);

      return {
        schemaVersion: '2.0',
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: nowMs,
        gameTimeMs: runtime.gameplayStartMs === null ? null : Math.max(0, nowMs - runtime.gameplayStartMs),
        status: baseStatus,
        is_actionable: baseStatus === 'playing',
        terminal: {
          isTerminal: false,
          outcome: null,
          reason: null
        },
        game_state: {
          score: currentMoney,
          level: null,
          player: null,
          environment: null,
          completion_progress: null,
          money: currentMoney,
          money_total_earned: totalMoneyEarned,
          money_scan_status: moneyInfo.scanStatus
        },
        metrics: {
          primary_score: currentMoney,
          attempts: runtime.resetCount
        },
        debug: {
          ready: true,
          canvas_width: canvasInfo.width,
          canvas_height: canvasInfo.height,
          money_source_path: moneyInfo.sourcePath || null,
          money_field_path: moneyInfo.keyPath || null,
          money_candidate_id: moneyInfo.candidateId || null,
          money_source_locked: !!trackedMoneyCandidateId,
          money_scanned_files: moneyInfo.scannedFiles,
          last_reset_method: runtime.lastResetMethod
        },
        raw: null
      };
    },
    reset: async function(options){
      const episode = beginEpisode(options || {});

      if (typeof window !== 'undefined' && window.location && window.location.reload){
        runtime.lastResetMethod = 'reload';
        window.location.reload();
        return {
          ok: true,
          method: 'reload',
          accepted: episode.accepted,
          applied: episode.applied,
          notes: episode.notes
        };
      }
      runtime.lastResetMethod = 'unsupported';
      return {
        ok: false,
        method: 'unsupported',
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.concat(['no_reset_method_available'])
      };
    }
  };
})();
