/* A clearly simulated step-coach prototype. It never calls an API or runs image analysis. */
(() => {
  'use strict';
  const steps = [
    { title: 'Attach the frame', instruction: 'Place the red frame squarely on the black chassis.', criteria: 'The frame sits fully on the chassis with no visible gap.', fix: 'Seat the frame squarely, then check again.' },
    { title: 'Attach the spare wheel', instruction: 'Mount the spare wheel on the rear panel.', criteria: 'The spare wheel is centered on the back of the jeep.', fix: 'Move the spare wheel to the center mounting point.' },
    { title: 'Attach the front wheels', instruction: 'Attach both front wheels to the front axle.', criteria: 'Both front wheels are visible below the front arches.', fix: 'Check that both front wheels are attached.' },
    { title: 'Attach the back wheels', instruction: 'Attach both back wheels to finish the jeep.', criteria: 'Both back wheels are attached and the jeep sits level.', fix: 'Attach the remaining back wheel and check the stance.' },
  ];
  const el = (id) => document.getElementById(id);
  const state = { index: 0, auto: false, phase: 'WORKING', count: 0, incorrect: 0,
    deadline: 0, message: 'Build this step, then press Done.', hint: '',
    stream: null, photoUrl: null, history: [], last: null };
  const required = () => state.auto ? 3 : 2;
  const active = () => document.getElementById('coach-screen').classList.contains('active');

  function record(text) {
    state.history.unshift({ time: new Date().toLocaleTimeString(), text });
    state.history.length = Math.min(state.history.length, 30);
  }
  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    const video = el('coach-video');
    video.pause(); video.srcObject = null; video.hidden = true;
    el('coach-camera-button').textContent = 'Start camera';
    el('coach-camera-status').textContent = 'CAMERA OFF';
    if (el('coach-upload-preview').hidden) el('coach-camera-empty').hidden = false;
  }
  async function listCameras(activeId) {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === 'videoinput');
      const select = el('coach-camera-select');
      select.replaceChildren();
      cameras.forEach((camera, index) => {
        const option = document.createElement('option');
        option.value = camera.deviceId;
        option.textContent = camera.label || `Camera ${index + 1}`;
        select.append(option);
      });
      if (activeId) select.value = activeId;
      el('coach-camera-select').disabled = cameras.length === 0;
    } catch (_) { /* A camera list is optional in preview. */ }
  }
  async function startCamera(deviceId) {
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) {
      el('coach-camera-message').textContent = 'Camera access needs HTTPS or localhost. You can still explore the simulated checks.';
      return;
    }
    try {
      const video = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } };
      state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      const preview = el('coach-video');
      preview.srcObject = state.stream;
      await preview.play();
      preview.hidden = false;
      el('coach-camera-empty').hidden = true;
      el('coach-upload-preview').hidden = true;
      el('coach-camera-button').textContent = 'Stop camera';
      el('coach-camera-status').textContent = 'LIVE PREVIEW';
      el('coach-camera-message').textContent = 'The camera stays in your browser. Results below remain simulated.';
      await listCameras(state.stream.getVideoTracks()[0]?.getSettings().deviceId);
    } catch (error) {
      stopCamera();
      el('coach-camera-message').textContent = `Camera unavailable: ${error.message || 'try a different camera or choose a photo.'}`;
    }
  }
  function render() {
    const complete = state.index >= steps.length;
    el('coach-active-view').hidden = complete;
    el('coach-complete').hidden = !complete;
    if (complete) { stopCamera(); return; }
    const step = steps[state.index];
    el('coach-step-count').textContent = `STEP ${String(state.index + 1).padStart(2, '0')} / ${String(steps.length).padStart(2, '0')}`;
    el('coach-step-title').textContent = step.title;
    el('coach-instruction').textContent = step.instruction;
    el('coach-criteria').textContent = step.criteria;
    const badge = el('coach-state-badge');
    badge.textContent = state.auto && state.phase === 'WORKING' ? 'WATCHING' : state.phase.replace('_', ' ');
    badge.dataset.state = state.phase;
    el('coach-status-label').textContent = state.auto ? 'AUTO PREVIEW' : 'MANUAL PREVIEW';
    el('coach-status-message').textContent = state.message;
    el('coach-status-hint').textContent = state.hint;
    el('coach-confirmation-dots').textContent = Array.from({ length: required() }, (_, i) => i < state.count ? '●' : '○').join(' ');
    el('coach-confirmation-text').textContent = `${state.count} of ${required()} simulated confirmations`;
    el('coach-auto').classList.toggle('selected', state.auto);
    el('coach-manual').classList.toggle('selected', !state.auto);
    el('coach-done').firstChild.textContent = state.auto ? 'Check now ' : state.phase === 'CHECKING' ? 'Checking… ' : state.phase === 'WORKING' ? 'Done with this step ' : 'Fixed it — check again ';
    el('coach-done').disabled = !state.auto && state.phase === 'CHECKING';
    const canSimulate = state.auto || state.phase === 'CHECKING';
    document.querySelectorAll('[data-coach-result]').forEach((button) => { button.disabled = !canSimulate; });
    el('coach-step-list').replaceChildren(...steps.map((item, index) => {
      const row = document.createElement('li');
      row.textContent = `${String(index + 1).padStart(2, '0')}  ${item.title}`;
      row.className = index < state.index ? 'done' : index === state.index ? 'current' : '';
      return row;
    }));
    const last = state.last;
    el('coach-reason-classifier').textContent = last?.classifier || '—';
    el('coach-reason-agent').textContent = last?.agent || '—';
    el('coach-reason-rule').textContent = last?.rule || 'Waiting for a simulated check';
    el('coach-reason-evidence').textContent = last?.evidence || 'The model reasoning panel will show how an actual check reached its result.';
    el('coach-history-list').replaceChildren(...state.history.map((entry) => {
      const row = document.createElement('li'); row.textContent = `${entry.time} · ${entry.text}`; return row;
    }));
  }
  function changeMode(auto) {
    if (state.auto === auto) return;
    state.auto = auto; state.phase = 'WORKING'; state.count = 0; state.incorrect = 0; state.deadline = 0;
    state.message = auto ? 'Watching the step. Simulate an observation below.' : 'Build this step, then press Done.';
    state.hint = auto ? 'Live auto checking will be connected after the coach backend is integrated.' : '';
    record(`Switched to ${auto ? 'auto' : 'manual'} preview; confirmation count reset.`);
    render();
  }
  function done() {
    if (state.auto) { state.message = 'Choose a simulated observation below.'; render(); return; }
    if (state.phase === 'CHECKING') return;
    state.phase = 'CHECKING'; state.count = 0; state.incorrect = 0;
    state.deadline = Date.now() + 20000;
    state.message = 'Checking… choose a simulated observation below.';
    state.hint = 'A live check would time out after 20 seconds.';
    record(`Started manual check for step ${state.index + 1}.`);
    render();
  }
  function simulate(kind) {
    if (!(state.auto || state.phase === 'CHECKING') || state.index >= steps.length) return;
    const step = steps[state.index];
    if (kind === 'match') {
      if (state.auto) state.phase = 'WORKING';
      state.count += 1; state.incorrect = 0;
      state.last = { classifier: 'correct · simulated', agent: 'matches · simulated',
        rule: `${state.count} / ${required()} confirmations`, evidence: `The visible condition would be checked against a reviewed photo: ${step.criteria}` };
      record(`Step ${state.index + 1}: simulated match ${state.count}/${required()}.`);
      if (state.count >= required()) {
        state.index += 1; state.phase = 'WORKING'; state.count = 0; state.deadline = 0;
        state.message = state.index < steps.length ? `Step verified. Next: ${steps[state.index].title}.` : 'All preview steps complete.';
        state.hint = '';
        record(`Step ${state.index} verified in the preview.`);
      } else {
        state.message = 'Looks right. One more different view is needed.';
        state.hint = 'This preview treats each click as a new simulated observation.';
      }
    } else if (kind === 'mismatch') {
      state.count = 0; state.incorrect += 1;
      const alarm = !state.auto || state.incorrect >= 2;
      state.phase = alarm ? 'NEEDS_FIX' : 'WORKING';
      state.deadline = 0;
      state.message = alarm ? 'A visible part needs correction.' : 'Possible mismatch. Check another view.';
      state.hint = alarm ? step.fix : 'Auto preview waits for a second mismatch before raising an alarm.';
      state.last = { classifier: 'incorrect_method · simulated', agent: 'skipped',
        rule: alarm ? 'Stop and correct the part' : 'First mismatch: wait for confirmation', evidence: step.fix };
      record(`Step ${state.index + 1}: simulated mismatch${alarm ? ', fix needed' : ', awaiting confirmation'}.`);
    } else {
      state.count = 0; state.incorrect = 0; state.phase = 'WORKING'; state.deadline = 0;
      state.message = 'The view is unclear. Try another angle.';
      state.hint = 'Unclear observations never verify a step.';
      state.last = { classifier: 'unclear_view · simulated', agent: 'skipped',
        rule: 'No confirmation counted', evidence: 'The relevant connection was not visible.' };
      record(`Step ${state.index + 1}: simulated unclear view.`);
    }
    render();
  }
  function reset() {
    state.index = 0; state.phase = 'WORKING'; state.count = 0; state.incorrect = 0; state.deadline = 0;
    state.message = state.auto ? 'Watching the step. Simulate an observation below.' : 'Build this step, then press Done.';
    state.hint = ''; state.last = null; state.history = [];
    record('Started a new simulated build.'); render();
  }
  function open() {
    document.querySelectorAll('.screen').forEach((screen) => screen.classList.toggle('active', screen.id === 'coach-screen'));
    location.hash = 'coach'; window.scrollTo({ top: 0, behavior: 'auto' }); render();
  }
  function close() {
    stopCamera();
    document.querySelectorAll('.screen').forEach((screen) => screen.classList.toggle('active', screen.id === 'home-screen'));
    location.hash = 'home'; window.scrollTo({ top: 0, behavior: 'auto' });
  }
  el('coach-preview-button').addEventListener('click', open);
  el('coach-back').addEventListener('click', close);
  el('coach-auto').addEventListener('click', () => changeMode(true));
  el('coach-manual').addEventListener('click', () => changeMode(false));
  el('coach-done').addEventListener('click', done);
  el('coach-restart').addEventListener('click', reset);
  el('coach-build-again').addEventListener('click', reset);
  document.querySelectorAll('[data-coach-result]').forEach((button) => button.addEventListener('click', () => simulate(button.dataset.coachResult)));
  el('coach-camera-button').addEventListener('click', () => state.stream ? stopCamera() : startCamera());
  el('coach-camera-select').addEventListener('change', (event) => startCamera(event.target.value));
  el('coach-photo-input').addEventListener('change', (event) => {
    const file = event.target.files[0]; event.target.value = '';
    if (!file || !file.type.startsWith('image/')) return;
    stopCamera();
    if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
    state.photoUrl = URL.createObjectURL(file);
    el('coach-upload-preview').src = state.photoUrl;
    el('coach-upload-preview').hidden = false;
    el('coach-camera-empty').hidden = true;
    el('coach-camera-message').textContent = 'Photo selected locally. No image analysis runs in this preview.';
  });
  window.addEventListener('hashchange', () => { if (!active()) stopCamera(); });
  window.addEventListener('pagehide', stopCamera);
  setInterval(() => {
    if (!active() || state.auto || state.phase !== 'CHECKING') return;
    const seconds = Math.max(0, Math.ceil((state.deadline - Date.now()) / 1000));
    el('coach-countdown').textContent = `${seconds}s left`;
    if (seconds === 0) {
      state.phase = 'WORKING'; state.count = 0; state.deadline = 0;
      state.message = 'Could not verify this step. Press Done to try again.';
      state.hint = ''; record(`Step ${state.index + 1}: simulated check timed out.`); render();
    }
  }, 500);
  reset();
})();
