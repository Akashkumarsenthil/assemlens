/* Browser UI for the AssemLens checkpoint flow. No model runs in this file. */
(() => {
  'use strict';

  const PREVIEW_PRODUCT = {
    id: 'fyd-jeep-v1',
    name: 'FYD take-apart jeep',
    description: 'A sample product for reviewing the interface.',
    steps: [{
      id: 'step_01',
      title: 'Visible state check',
      instruction: 'Compare the current assembly with an approved reference image.',
      criteria: 'Confirm the visible parts and their relationship.',
      referenceUrl: null,
    }],
  };
  const SAMPLE_RESULTS = {
    matches: { assessment: 'matches', evidence: 'The visible parts appear to match the reference.', nextAction: 'Take a second clear view to confirm.' },
    mismatch: { assessment: 'mismatch', evidence: 'A visible part appears out of place.', nextAction: 'Adjust the part, then check again.' },
    uncertain: { assessment: 'uncertain', evidence: 'The relevant connection is not clear in this view.', nextAction: 'Move closer and show the connection from another angle.' },
  };
  const state = {
    mode: localStorage.getItem('assemlens-mode') === 'api' ? 'api' : 'preview',
    apiBase: localStorage.getItem('assemlens-api-base') || '',
    product: null, stepIndex: 0, sessionId: null,
    stream: null, streamTarget: null, qrLoop: null,
    imageBlob: null, imageUrl: null, referenceBlob: null, referenceObjectUrl: null,
    liveLab: false, modelEnabled: false, busy: false,
  };
  const el = (id) => document.getElementById(id);
  const show = (id) => {
    stopCamera();
    for (const screen of document.querySelectorAll('.screen')) screen.classList.toggle('active', screen.id === `${id}-screen`);
    window.scrollTo({ top: 0, behavior: 'auto' });
    location.hash = id;
  };
  const setMessage = (id, message) => { el(id).textContent = message; };
  const escapeProductId = (value) => /^[a-z0-9][a-z0-9_-]{0,63}$/i.test(value);

  function updateMode() {
    el('mode-chip').textContent = state.liveLab ? 'LIVE NANO MODEL' : state.mode === 'preview' ? 'PREVIEW MODE' : 'NANO API';
    el('mode-chip').classList.toggle('live', state.mode === 'api');
    el('preview-button').hidden = state.mode === 'api';
    el('live-button').hidden = !state.modelEnabled;
    el('settings-form').elements.mode.value = state.mode;
    el('api-base').value = state.apiBase;
  }

  function apiUrl(path) {
    const base = state.apiBase || location.origin;
    return new URL(path, base.endsWith('/') ? base : `${base}/`).toString();
  }

  async function request(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    if (!response.ok) {
      let detail = '';
      try { detail = (await response.json()).detail || ''; } catch (_) { /* An HTTP status is enough. */ }
      throw new Error(typeof detail === 'string' && detail ? detail : `Server returned ${response.status}.`);
    }
    return response.json();
  }

  function productIdFromQR(raw) {
    const value = raw.trim();
    if (escapeProductId(value)) return value;
    try {
      const url = new URL(value);
      if (!['http:', 'https:'].includes(url.protocol)) return null;
      const match = url.pathname.match(/^\/p\/([a-z0-9_-]+)\/?$/i);
      return match && escapeProductId(match[1]) ? match[1] : null;
    } catch (_) { return null; }
  }

  function validateProduct(product) {
    if (!product || typeof product.id !== 'string' || !escapeProductId(product.id) ||
        typeof product.name !== 'string' || !Array.isArray(product.steps) || product.steps.length === 0) {
      throw new Error('This product package is incomplete.');
    }
    for (const step of product.steps) {
      if (!step || typeof step.id !== 'string' || !escapeProductId(step.id) ||
          typeof step.title !== 'string' || typeof step.instruction !== 'string' ||
          typeof step.criteria !== 'string') throw new Error('A product step is incomplete.');
    }
    return product;
  }

  async function loadProduct(id) {
    if (!escapeProductId(id)) throw new Error('Enter a valid product ID using letters, numbers, dashes, or underscores.');
    const product = state.mode === 'preview'
      ? (id === PREVIEW_PRODUCT.id ? PREVIEW_PRODUCT : null)
      : await request(`api/products/${encodeURIComponent(id)}`);
    if (!product) throw new Error('Preview mode includes only fyd-jeep-v1. Switch to Nano API for other products.');
    state.liveLab = false;
    clearReference();
    el('lab-panel').hidden = true;
    el('step-card').hidden = false;
    el('reference-upload-button').hidden = true;
    el('analyze-button').firstChild.textContent = 'Check this view ';
    updateMode();
    state.product = validateProduct(product);
    state.stepIndex = 0;
    state.sessionId = null;
    clearImage();
    el('result-card').hidden = true;
    renderProduct();
    show('work');
  }

  function clearReference() {
    if (state.referenceObjectUrl) URL.revokeObjectURL(state.referenceObjectUrl);
    state.referenceObjectUrl = null;
    state.referenceBlob = null;
    el('reference-input').value = '';
  }

  function selectReference(blob) {
    if (!blob || !['image/jpeg', 'image/png', 'image/webp'].includes(blob.type)) throw new Error('Choose a JPEG, PNG, or WebP reference.');
    if (blob.size > 10 * 1024 * 1024) throw new Error('Choose a reference smaller than 10 MB.');
    clearReference();
    state.referenceBlob = blob;
    state.referenceObjectUrl = URL.createObjectURL(blob);
    const frame = el('reference-image');
    frame.replaceChildren();
    const image = document.createElement('img');
    image.src = state.referenceObjectUrl;
    image.alt = 'Your reference photo';
    frame.append(image);
    el('reference-title').textContent = 'Your reference photo';
    el('reference-caption').textContent = 'This photo is compared with the current view on the Nano.';
    el('reference-upload-button').firstChild.textContent = 'Change reference photo ';
  }

  function startLiveComparison() {
    state.liveLab = true;
    state.mode = 'api';
    localStorage.setItem('assemlens-mode', 'api');
    state.product = null;
    state.sessionId = null;
    clearImage();
    clearReference();
    updateMode();
    el('step-card').hidden = true;
    el('lab-panel').hidden = false;
    el('reference-upload-button').hidden = false;
    el('work-title').textContent = 'Live visual comparison';
    el('product-description').textContent = 'Compare a reference and a current photo using the Nano GPU.';
    el('product-kicker').textContent = 'EXPERIMENTAL MODEL TEST';
    el('step-pill').textContent = 'LIVE MODEL';
    el('reference-image').textContent = '◎';
    el('reference-title').textContent = 'Reference photo needed';
    el('reference-caption').textContent = 'Choose a clear image of the intended visible state.';
    el('reference-upload-button').firstChild.textContent = 'Choose reference photo ';
    el('lab-instruction').value = '';
    el('lab-criteria').value = '';
    el('analyze-button').firstChild.textContent = 'Compare views ';
    el('result-card').hidden = true;
    show('work');
  }

  function referenceUrl(value) {
    if (!value || typeof value !== 'string') return null;
    try {
      const url = new URL(value, apiUrl(''));
      return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null;
    } catch (_) { return null; }
  }

  function renderProduct() {
    const product = state.product;
    const step = product.steps[state.stepIndex];
    el('work-title').textContent = product.name;
    el('product-description').textContent = product.description || 'One visible checkpoint at a time.';
    el('product-kicker').textContent = state.mode === 'preview' ? 'UI PREVIEW / PRODUCT CHECK' : 'PRODUCT CHECK';
    el('step-pill').textContent = `STEP ${String(state.stepIndex + 1).padStart(2, '0')} / ${String(product.steps.length).padStart(2, '0')}`;
    el('step-title').textContent = step.title;
    el('step-instruction').textContent = step.instruction;
    el('step-criteria').lastElementChild.textContent = step.criteria;
    el('reference-title').textContent = step.referenceUrl ? 'Approved step photo' : 'Reference photo pending';
    el('reference-caption').textContent = step.referenceUrl ? 'Use the same step when comparing views.' : 'Add a reviewed reference in the product package.';
    const frame = el('reference-image');
    frame.replaceChildren();
    const url = referenceUrl(step.referenceUrl);
    if (url) {
      const img = document.createElement('img');
      img.src = url;
      img.alt = `Reference for ${step.title}`;
      img.onerror = () => { frame.textContent = '◎'; };
      frame.append(img);
    } else frame.textContent = '◎';
  }

  function stopCamera() {
    if (state.qrLoop) cancelAnimationFrame(state.qrLoop);
    state.qrLoop = null;
    if (state.stream) for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
    state.streamTarget = null;
    for (const video of [el('qr-video'), el('capture-video')]) { video.pause(); video.srcObject = null; video.classList.remove('active'); }
    el('scanner-placeholder').hidden = false;
    el('capture-placeholder').hidden = Boolean(state.imageBlob);
    el('live-tag').hidden = true;
    el('capture-button').disabled = true;
    el('camera-button').textContent = 'Start camera';
    el('start-scanner').textContent = 'Start QR scanner';
  }

  async function startCamera(target) {
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs HTTPS or localhost. Use a photo or manual product ID.');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } });
    state.stream = stream;
    state.streamTarget = target;
    const video = el(target === 'qr' ? 'qr-video' : 'capture-video');
    video.srcObject = stream;
    await video.play();
    video.classList.add('active');
    if (target === 'qr') {
      el('scanner-placeholder').hidden = true;
      el('start-scanner').textContent = 'Stop QR scanner';
    } else {
      clearImage();
      el('capture-placeholder').hidden = true;
      el('live-tag').hidden = false;
      el('capture-button').disabled = false;
      el('camera-button').textContent = 'Stop camera';
    }
  }

  async function startQrScanner() {
    if (!('BarcodeDetector' in window)) throw new Error('This browser cannot scan QR codes. Enter the product ID below.');
    const formats = await BarcodeDetector.getSupportedFormats();
    if (!formats.includes('qr_code')) throw new Error('This browser cannot scan QR codes. Enter the product ID below.');
    await startCamera('qr');
    const detector = new BarcodeDetector({ formats: ['qr_code'] });
    let last = 0;
    const scan = async (time) => {
      if (state.streamTarget !== 'qr') return;
      state.qrLoop = requestAnimationFrame(scan);
      if (time - last < 350 || el('qr-video').readyState < 2) return;
      last = time;
      try {
        const codes = await detector.detect(el('qr-video'));
        if (!codes.length || state.streamTarget !== 'qr') return;
        const id = productIdFromQR(codes[0].rawValue);
        if (!id) { setMessage('scanner-message', 'That QR code is not an AssemLens product ID.'); return; }
        stopCamera();
        await loadProduct(id);
      } catch (error) { setMessage('scanner-message', error.message || 'Could not read that code.'); }
    };
    state.qrLoop = requestAnimationFrame(scan);
  }

  function clearImage() {
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
    state.imageBlob = null;
    el('captured-image').hidden = true;
    el('captured-image').removeAttribute('src');
    el('analyze-button').disabled = true;
  }

  function selectImage(blob) {
    if (!blob || !blob.type.startsWith('image/')) throw new Error('Choose an image file.');
    if (blob.size > 10 * 1024 * 1024) throw new Error('Choose an image smaller than 10 MB.');
    stopCamera();
    clearImage();
    state.imageBlob = blob;
    state.imageUrl = URL.createObjectURL(blob);
    const image = el('captured-image');
    image.src = state.imageUrl;
    image.hidden = false;
    el('capture-placeholder').hidden = true;
    el('analyze-button').disabled = false;
    el('result-card').hidden = true;
    el('review-stage').classList.remove('reviewed');
    setMessage('camera-message', 'View ready. Check it when the relevant parts are clearly visible.');
  }

  async function captureFrame() {
    const video = el('capture-video');
    if (!video.videoWidth || !video.videoHeight) throw new Error('Wait for the camera image, then capture again.');
    const scale = Math.min(1, 1280 / video.videoWidth);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .86));
    if (!blob) throw new Error('Could not capture this frame.');
    selectImage(blob);
  }

  function showResult(result, simulated = false) {
    const assessment = ['matches', 'mismatch', 'uncertain'].includes(result.assessment) ? result.assessment : 'error';
    const titles = { matches: 'This view matches', mismatch: 'Something looks different', uncertain: 'We need a clearer view', error: 'Could not check this view' };
    const icons = { matches: '✓', mismatch: '!', uncertain: '?', error: '!' };
    const card = el('result-card');
    card.className = `result-card ${assessment}`;
    el('result-icon').textContent = icons[assessment];
    el('result-kicker').textContent = simulated ? 'SIMULATED RESULT' : assessment === 'error' ? 'CHECK FAILED' : result.experimental ? 'EXPERIMENTAL MODEL RESULT' : 'VIEW CHECKED';
    el('result-title').textContent = titles[assessment];
    el('result-evidence').textContent = result.evidence || '';
    el('result-action').textContent = result.nextAction || 'Check another view.';
    const latency = Number.isFinite(result.latencyMs) ? ` · ${Math.round(result.latencyMs)} ms` : '';
    el('result-meta').textContent = `${simulated ? 'UI preview · no AI analysis' : result.experimental ? 'Nano GPU · experimental comparison' : 'Nano API result'}${latency}${result.readyToAdvance ? ' · Ready for human confirmation' : ''}`;
    card.hidden = false;
    el('review-stage').classList.add('reviewed');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function analyze() {
    if (!state.imageBlob || state.busy) return;
    if (state.liveLab) {
      if (!state.referenceBlob) { el('reference-caption').textContent = 'Choose a reference photo first.'; return; }
      if (!el('lab-instruction').value.trim() || !el('lab-criteria').value.trim()) {
        setMessage('camera-message', 'Describe the goal and visible conditions before comparing.'); return;
      }
    } else if (state.mode === 'preview') { el('preview-dialog').showModal(); return; }
    state.busy = true;
    el('analyze-button').disabled = true;
    el('analyze-button').firstChild.textContent = 'Checking view ';
    try {
      if (state.liveLab) {
        const form = new FormData();
        form.append('reference', state.referenceBlob, 'reference.jpg');
        form.append('image', state.imageBlob, 'capture.jpg');
        form.append('instruction', el('lab-instruction').value.trim());
        form.append('criteria', el('lab-criteria').value.trim());
        const result = await request('api/compare', { method: 'POST', body: form });
        showResult(result);
        return;
      }
      if (!state.sessionId) {
        const session = await request('api/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ productId: state.product.id }) });
        if (!session || typeof session.id !== 'string') throw new Error('The server did not return a session ID.');
        state.sessionId = session.id;
      }
      const form = new FormData();
      form.append('stepId', state.product.steps[state.stepIndex].id);
      form.append('image', state.imageBlob, 'capture.jpg');
      const result = await request(`api/sessions/${encodeURIComponent(state.sessionId)}/observations`, { method: 'POST', body: form });
      showResult(result);
    } catch (error) {
      showResult({ assessment: 'error', evidence: error.message || 'The server is unavailable.', nextAction: 'Check the connection and try again.' });
    } finally {
      state.busy = false;
      el('analyze-button').disabled = false;
      el('analyze-button').firstChild.textContent = state.liveLab ? 'Compare views ' : 'Check this view ';
    }
  }

  document.querySelector('.brand').addEventListener('click', (event) => { event.preventDefault(); show('home'); });
  el('scan-button').addEventListener('click', () => show('scan'));
  el('preview-button').addEventListener('click', () => loadProduct(PREVIEW_PRODUCT.id));
  el('live-button').addEventListener('click', startLiveComparison);
  for (const button of document.querySelectorAll('[data-back]')) button.addEventListener('click', () => show(button.dataset.back));
  el('product-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    el('product-error').hidden = true;
    try { await loadProduct(el('product-id').value.trim()); }
    catch (error) { el('product-error').textContent = error.message; el('product-error').hidden = false; }
  });
  el('start-scanner').addEventListener('click', async () => {
    if (state.streamTarget === 'qr') { stopCamera(); return; }
    try { await startQrScanner(); setMessage('scanner-message', 'Hold a product QR code inside the frame.'); }
    catch (error) { stopCamera(); setMessage('scanner-message', error.message); }
  });
  el('camera-button').addEventListener('click', async () => {
    if (state.streamTarget === 'capture') { stopCamera(); return; }
    try { await startCamera('capture'); setMessage('camera-message', 'Move the relevant parts into frame, then capture.'); }
    catch (error) { stopCamera(); setMessage('camera-message', error.message); }
  });
  el('capture-button').addEventListener('click', async () => {
    try { await captureFrame(); } catch (error) { setMessage('camera-message', error.message); }
  });
  el('reference-input').addEventListener('change', (event) => {
    try { if (event.target.files[0]) selectReference(event.target.files[0]); }
    catch (error) { el('reference-caption').textContent = error.message; }
  });
  el('photo-input').addEventListener('change', (event) => {
    try { if (event.target.files[0]) selectImage(event.target.files[0]); }
    catch (error) { setMessage('camera-message', error.message); }
    event.target.value = '';
  });
  el('analyze-button').addEventListener('click', analyze);
  el('check-again-button').addEventListener('click', () => {
    clearImage(); el('result-card').hidden = true; el('review-stage').classList.remove('reviewed'); el('capture-placeholder').hidden = false;
    el('capture-stage').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  el('settings-button').addEventListener('click', () => el('settings-dialog').showModal());
  el('settings-form').addEventListener('submit', (event) => {
    if (event.submitter?.value !== 'save') return;
    event.preventDefault();
    const base = el('api-base').value.trim();
    if (base) {
      try { if (!['http:', 'https:'].includes(new URL(base).protocol)) throw new Error(); }
      catch (_) { el('api-base').setCustomValidity('Use an http or https URL.'); el('api-base').reportValidity(); return; }
    }
    el('api-base').setCustomValidity('');
    state.mode = el('settings-form').elements.mode.value;
    state.apiBase = base;
    localStorage.setItem('assemlens-mode', state.mode);
    localStorage.setItem('assemlens-api-base', base);
    state.product = null; state.sessionId = null; state.liveLab = false; clearImage(); clearReference();
    updateMode();
    el('settings-dialog').close();
    show('home');
  });
  el('api-base').addEventListener('input', () => el('api-base').setCustomValidity(''));
  for (const button of document.querySelectorAll('[data-preview-result]')) button.addEventListener('click', () => {
    el('preview-dialog').close();
    showResult(SAMPLE_RESULTS[button.dataset.previewResult], true);
  });
  el('cancel-preview').addEventListener('click', () => el('preview-dialog').close());
  window.addEventListener('pagehide', stopCamera);
  updateMode();
  request('api/health').then((health) => {
    state.modelEnabled = health.inferenceEnabled === true;
    const apiOption = document.querySelector('input[name="mode"][value="api"]');
    apiOption.disabled = !state.modelEnabled;
    el('api-mode-note').textContent = state.modelEnabled
      ? 'Use reviewed products and model responses'
      : 'Live inference is disabled on this server';
    if (!state.modelEnabled && state.mode === 'api') {
      state.mode = 'preview';
      localStorage.setItem('assemlens-mode', 'preview');
    }
    updateMode();
  }).catch(() => {
    if (state.mode === 'api') { state.mode = 'preview'; updateMode(); }
  });
})();
