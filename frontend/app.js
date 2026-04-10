/**
 * ISL Translator — Frontend v3.0 (Premium Edition)
 * Features: Hand Skeleton, Sparkline, Voice Commands, Analytics,
 *           Sign Gallery, Model Modal, Tour, Sound FX, PDF Export, Typewriter
 */

// ============ STATE ============
let ws = null, videoStream = null, mpCamera = null;
let isTranslating = false, isMirrored = false, isVoiceActive = false, swapHands = false;
let recognition = null;
let currentWords = [], translationHistory = [];
let signFrequency = {}, confidenceHistory = [];
let sessionSignCount = 0, translationCount = 0;
let sessionStartTime = null;
let fpsFrames = 0, fpsLast = performance.now();
let allSigns = [];
let tourStep = 0;
let toastTimeout = null;
let audioCtx = null;
let holisticInstance = null;

// ============ HAND SKELETON CONNECTIONS ============
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
];

// ============ DOM ============
const video = document.getElementById('videoFeed');
const canvas = document.getElementById('canvasCapture');
const ctx = canvas.getContext('2d');
const skeletonCanvas = document.getElementById('skeletonCanvas');
const skelCtx = skeletonCanvas.getContext('2d');
const sparkCanvas = document.getElementById('sparklineCanvas');
const sparkCtx = sparkCanvas.getContext('2d');

const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const statusBadge = document.getElementById('statusBadge');
const handIndicator = document.getElementById('handIndicator');
const bufferFill = document.getElementById('bufferFill');
const bufferLabel = document.getElementById('bufferLabel');
const predictionOverlay = document.getElementById('predictionOverlay');
const predLabel = document.getElementById('predLabel');
const predConf = document.getElementById('predConf');
const signWord = document.getElementById('signWord');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const wordsDisplay = document.getElementById('wordsDisplay');
const sentenceText = document.getElementById('sentenceText');
const historyList = document.getElementById('historyList');
const tickerScroll = document.getElementById('tickerScroll');
const cameraIntro = document.getElementById('cameraIntro');
const wordCount = document.getElementById('wordCount');
const historyCount = document.getElementById('historyCount');
const tipText = document.getElementById('tipText');
const statSessions = document.getElementById('statSessions');
const ringFill = document.getElementById('ringFill');
const ringText = document.getElementById('ringText');
const RING_TOTAL = 2 * Math.PI * 22; // circumference

// ============ TIPS ============
const TIPS = [
    "Make sure your hand is clearly visible in good lighting.",
    "Hold each sign steady for ~1 second until the ring fills completely.",
    "Position your hand in the center frame for optimal detection.",
    "Avoid busy backgrounds — solid colors work best with MediaPipe.",
    "Use the ✨ Translate button to convert signs into a natural sentence.",
    "Press Space to speak the translation aloud using TTS.",
    "The model recognizes 318+ ISL signs — hover the ticker to browse!",
    "Say 'translate' with Voice Commands enabled for hands-free use.",
    "Click 📊 after your session to see detailed analytics.",
    "Click ℹ️ on the sign card to see the LSTM model architecture.",
];
let tipIdx = 0;
tipText.style.transition = 'opacity 0.4s ease';
setInterval(() => {
    tipIdx = (tipIdx + 1) % TIPS.length;
    tipText.style.opacity = '0';
    setTimeout(() => { tipText.textContent = TIPS[tipIdx]; tipText.style.opacity = '1'; }, 420);
}, 7000);

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    fetchSigns();
    checkHealth();
    startFPSCounter();
    drawSparkline();
    resizeSkeletonCanvas();
    window.addEventListener('resize', resizeSkeletonCanvas);
    // Show onboarding tour first time
    if (!localStorage.getItem('isl_tour_done')) {
        setTimeout(startTour, 1200);
    }
    // First-time keyboard hint
    if (!localStorage.getItem('isl_hints')) {
        setTimeout(() => {
            showToast('⌨️ Shortcuts: S=Start  X=Stop  T=Translate  Space=Speak  M=Mirror');
            localStorage.setItem('isl_hints', '1');
        }, 3000);
    }
});

// ============ SKELETON CANVAS RESIZE ============
function resizeSkeletonCanvas() {
    skeletonCanvas.width = skeletonCanvas.offsetWidth;
    skeletonCanvas.height = skeletonCanvas.offsetHeight;
}

// ============ FPS COUNTER ============
function startFPSCounter() {
    setInterval(() => {
        const now = performance.now();
        const fps = Math.round(fpsFrames / ((now - fpsLast) / 1000));
        document.getElementById('fpsValue').textContent = isTranslating ? `${fps} FPS` : '-- FPS';
        fpsFrames = 0;
        fpsLast = now;
    }, 1000);
}

// ============ THEME ============
function initTheme() {
    const t = localStorage.getItem('isl_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', t);
    updateThemeIcons(t);
}
function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('isl_theme', next);
    updateThemeIcons(next);
}
function updateThemeIcons(t) {
    const l = document.querySelector('.theme-icon-light');
    const d = document.querySelector('.theme-icon-dark');
    if (l && d) { l.style.display = t === 'dark' ? 'inline' : 'none'; d.style.display = t === 'dark' ? 'none' : 'inline'; }
}

// ============ MIRROR ============
function toggleMirror() {
    isMirrored = !isMirrored;
    document.getElementById('cameraWrapper').classList.toggle('mirrored', isMirrored);
    showToast(isMirrored ? '🔄 Mirror: Normal' : '🔄 Mirror: Flipped');
}

// ============ API ============
async function fetchSigns() {
    try {
        const res = await fetch('/api/signs');
        const data = await res.json();
        allSigns = data.signs;
        populateTicker(allSigns);
        const el = document.getElementById('signsCountBadge');
        if (el) el.textContent = `⚡ ${data.count}+ Signs`;
        const se = document.getElementById('statSigns');
        if (se) se.textContent = data.count + '+';
        document.getElementById('galleryTotal').textContent = data.count + '+';
        buildGalleryGrid(allSigns);
    } catch(e) { console.warn('fetchSigns:', e); }
}

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.status === 'online') setStatus('online', `${data.signs_count} Signs · Ready`);
    } catch { setStatus('offline', 'Server Offline'); }
}

function setStatus(state, text) {
    statusBadge.className = `status-badge ${state}`;
    statusBadge.querySelector('.status-text').textContent = text;
}

function populateTicker(signs) {
    const all = [...signs, ...signs, ...signs];
    tickerScroll.innerHTML = all.map(s => `<span class="ticker-item">${s.replace(/_/g,' ')}</span>`).join('');
}

// ============ CAMERA ============
async function initCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
        video.srcObject = videoStream;
        return true;
    } catch { showToast('❌ Camera access denied!', true); return false; }
}
function stopCamera() {
    if (mpCamera) { mpCamera.stop(); mpCamera = null; }
    if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
}

// ============ MEDIAPIPE BROWSER-SIDE ============
function initHolistic() {
    // Use cached instance if already created
    if (!holisticInstance) {
        holisticInstance = new Holistic({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`
        });
        holisticInstance.setOptions({
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });
        holisticInstance.onResults(onHolisticResults);
    }

    mpCamera = new Camera(video, {
        onFrame: async () => {
            if (isTranslating && holisticInstance) {
                await holisticInstance.send({ image: video });
            }
        },
        width: 640,
        height: 480,
    });
    mpCamera.start();
}

function onHolisticResults(results) {
    if (!isTranslating) return;
    fpsFrames++;

    // Draw skeleton directly from JS MediaPipe results
    drawSkeletonFromHolistic(results);

    // Update hand indicator
    const handsDetected = !!(results.leftHandLandmarks || results.rightHandLandmarks);
    if (handsDetected) {
        handIndicator.className = 'hand-indicator active';
        handIndicator.innerHTML = '<span class="hand-pulse"></span><span>Hands Detected ✋</span>';
    } else {
        handIndicator.className = 'hand-indicator';
        handIndicator.innerHTML = '<span class="hand-pulse"></span><span>No Hands Detected</span>';
    }

    // Send 258 keypoints to backend (same format as Python extract_keypoints)
    if (ws && ws.readyState === WebSocket.OPEN) {
        const keypoints = extractKeypoints(results);
        ws.send(JSON.stringify({
            type: 'landmarks',
            data: keypoints,
            hands_detected: handsDetected
        }));
    }
}

function extractKeypoints(results) {
    // Pose: 33 landmarks x 4 (x, y, z, visibility) = 132
    const pose = results.poseLandmarks
        ? results.poseLandmarks.flatMap(lm => [lm.x, lm.y, lm.z, lm.visibility || 0])
        : new Array(33 * 4).fill(0);

    // swapHands mode swaps left/right labels —
    // fixes cases where browser MediaPipe labels hands opposite to the training dataset
    const leftSrc  = swapHands ? results.rightHandLandmarks : results.leftHandLandmarks;
    const rightSrc = swapHands ? results.leftHandLandmarks  : results.rightHandLandmarks;

    const lh = leftSrc
        ? leftSrc.flatMap(lm => [lm.x, lm.y, lm.z])
        : new Array(21 * 3).fill(0);
    const rh = rightSrc
        ? rightSrc.flatMap(lm => [lm.x, lm.y, lm.z])
        : new Array(21 * 3).fill(0);

    return [...pose, ...lh, ...rh]; // Total = 258
}

function drawSkeletonFromHolistic(results) {
    const W = skeletonCanvas.width, H = skeletonCanvas.height;
    skelCtx.clearRect(0, 0, W, H);

    const hands = [];
    if (results.leftHandLandmarks)  hands.push({ pts: results.leftHandLandmarks,  color: '#00e5a0' });
    if (results.rightHandLandmarks) hands.push({ pts: results.rightHandLandmarks, color: '#a29bfe' });

    hands.forEach(({ pts, color }) => {
        const mapped = pts.map(lm => ({
            x: isMirrored ? lm.x * W : (1 - lm.x) * W,
            y: lm.y * H
        }));
        // Connections
        HAND_CONNECTIONS.forEach(([a, b]) => {
            skelCtx.beginPath();
            skelCtx.strokeStyle = color;
            skelCtx.lineWidth = 2;
            skelCtx.globalAlpha = 0.75;
            skelCtx.moveTo(mapped[a].x, mapped[a].y);
            skelCtx.lineTo(mapped[b].x, mapped[b].y);
            skelCtx.stroke();
        });
        // Joints
        mapped.forEach((pt, i) => {
            skelCtx.beginPath();
            skelCtx.globalAlpha = 0.9;
            skelCtx.fillStyle = i === 0 ? '#ffffff' : color;
            skelCtx.arc(pt.x, pt.y, i === 0 ? 5 : 3, 0, Math.PI * 2);
            skelCtx.fill();
        });
        skelCtx.globalAlpha = 1;
    });
}

// ============ WEBSOCKET ============
function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/translate`);
    ws.onopen = () => setStatus('online', 'Translating Live');
    ws.onmessage = e => handlePrediction(JSON.parse(e.data));
    ws.onerror = () => setStatus('offline', 'Connection Error');
    ws.onclose = () => { if (isTranslating) { setStatus('offline', 'Reconnecting...'); setTimeout(connectWebSocket, 2000); } };
}

// ============ HANDLE PREDICTIONS ============
let lastSignWord = '', lastSoundSign = '';

function handlePrediction(data) {
    if (data.type === 'prediction') {
        // Hand indicator
        if (data.hands_detected) {
            handIndicator.className = 'hand-indicator active';
            handIndicator.innerHTML = '<span class="hand-pulse"></span><span>Hands Detected ✋</span>';
        } else {
            handIndicator.className = 'hand-indicator';
            handIndicator.innerHTML = '<span class="hand-pulse"></span><span>No Hands Detected</span>';
        }

        // Buffer + Ring
        const prog = Math.min((data.buffer_progress / data.buffer_needed), 1);
        bufferFill.style.width = `${prog * 100}%`;
        if (bufferLabel) bufferLabel.textContent = `${data.buffer_progress} / ${data.buffer_needed}`;
        // Update SVG ring
        const offset = RING_TOTAL * (1 - prog);
        if (ringFill) { ringFill.style.strokeDashoffset = offset; ringFill.setAttribute('stroke-dasharray', `${RING_TOTAL} ${RING_TOTAL}`); }
        if (ringText) ringText.textContent = `${Math.round(prog * 100)}%`;

        // Prediction
        if (data.current_sign) {
            const conf = Math.round((data.confidence || 0) * 100);
            const name = data.current_sign.replace(/_/g, ' ');

            predictionOverlay.classList.add('visible');
            predLabel.textContent = name;
            predConf.textContent = `${conf}%`;

            if (name !== lastSignWord) {
                signWord.classList.remove('pop');
                void signWord.offsetWidth;
                signWord.classList.add('pop');
                lastSignWord = name;
            }
            signWord.textContent = name;
            confidenceBar.style.width = `${conf}%`;
            confidenceBar.className = `confidence-bar ${conf < 40 ? 'low' : conf < 70 ? 'medium' : 'high'}`;
            confidenceText.textContent = `${conf}%`;

            // Sparkline update
            confidenceHistory.push(conf);
            if (confidenceHistory.length > 30) confidenceHistory.shift();
            drawSparkline();

            // Card glow
            if (conf > 55) {
                document.getElementById('cardSign').classList.add('detected');
                setTimeout(() => document.getElementById('cardSign').classList.remove('detected'), 700);
            }

            // Sound on new detection
            if (conf > 70 && name !== lastSoundSign) {
                playDetectionSound();
                lastSoundSign = name;
            }

            // Top-3 alternatives panel
            const top3Container = document.getElementById('top3Container');
            const top3List = document.getElementById('top3List');
            if (data.top3 && data.top3.length > 0 && top3Container && top3List) {
                top3Container.style.display = 'block';
                top3List.innerHTML = data.top3.map((item, i) => `
                    <div class="top3-item rank-${i + 1}">
                        <span class="top3-rank">#${i + 1}</span>
                        <span class="top3-sign">${item.sign.replace(/_/g, ' ')}</span>
                        <div class="top3-bar-wrap"><div class="top3-bar" style="width:${Math.min(item.confidence, 100)}%"></div></div>
                        <span class="top3-pct">${item.confidence}%</span>
                    </div>`).join('');
            }
        }  // end if(data.current_sign)

        // Draw Hand Skeleton
        if (data.landmarks) drawSkeleton(data.landmarks);
        else clearSkeleton();

        // Word sequence
        if (data.sentence && data.sentence.length > 0) {
            if (data.sentence.length > currentWords.length) {
                sessionSignCount++;
                if (statSessions) statSessions.textContent = sessionSignCount;
                const newSign = data.sentence[data.sentence.length - 1];
                signFrequency[newSign] = (signFrequency[newSign] || 0) + 1;
            }
            currentWords = data.sentence;
            renderWordChips(data.sentence);
            wordCount.textContent = `${data.sentence.length} word${data.sentence.length > 1 ? 's' : ''}`;
        }

    } else if (data.type === 'nlp_result') {
        typewriterEffect(data.sentence);
        addToHistory(data.sentence);
        translationCount++;
        showToast('✅ AI translation complete!');

    } else if (data.type === 'cleared') {
        resetUI();
    }
}

// ============ HAND SKELETON DRAWING ============
function drawSkeleton(landmarks) {
    const W = skeletonCanvas.width, H = skeletonCanvas.height;
    skelCtx.clearRect(0, 0, W, H);

    const hands = [];
    if (landmarks.left) hands.push({ pts: landmarks.left, color: '#00e5a0' });
    if (landmarks.right) hands.push({ pts: landmarks.right, color: '#a29bfe' });

    hands.forEach(({ pts, color }) => {
        // Flip x if not mirrored (landmarks are in original frame coords)
        const mapped = pts.map(([x, y]) => ({
            x: isMirrored ? x * W : (1 - x) * W,
            y: y * H
        }));

        // Draw connections
        skelCtx.strokeStyle = color;
        skelCtx.lineWidth = 2;
        skelCtx.globalAlpha = 0.75;
        HAND_CONNECTIONS.forEach(([a, b]) => {
            skelCtx.beginPath();
            skelCtx.moveTo(mapped[a].x, mapped[a].y);
            skelCtx.lineTo(mapped[b].x, mapped[b].y);
            skelCtx.stroke();
        });

        // Draw joints
        mapped.forEach((pt, i) => {
            skelCtx.beginPath();
            skelCtx.globalAlpha = 0.9;
            skelCtx.fillStyle = i === 0 ? '#ffffff' : color;
            skelCtx.arc(pt.x, pt.y, i === 0 ? 5 : 3, 0, Math.PI * 2);
            skelCtx.fill();
        });
        skelCtx.globalAlpha = 1;
    });
}

function clearSkeleton() { skelCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height); }

// ============ SPARKLINE ============
function drawSparkline() {
    const W = sparkCanvas.offsetWidth || sparkCanvas.parentElement?.offsetWidth || 300;
    sparkCanvas.width = W;
    const H = sparkCanvas.height;
    sparkCtx.clearRect(0, 0, W, H);

    if (confidenceHistory.length < 2) {
        sparkCtx.fillStyle = 'rgba(255,255,255,0.04)';
        sparkCtx.beginPath();
        sparkCtx.roundRect ? sparkCtx.roundRect(0, 0, W, H, 6) : sparkCtx.rect(0, 0, W, H);
        sparkCtx.fill();
        return;
    }

    const pts = confidenceHistory;
    const step = W / (pts.length - 1);
    const pad = 4;

    // Background
    sparkCtx.fillStyle = 'rgba(255,255,255,0.03)';
    sparkCtx.fillRect(0, 0, W, H);

    // Gradient fill under line
    const grad = sparkCtx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(108,92,231,0.3)');
    grad.addColorStop(1, 'rgba(108,92,231,0.02)');

    sparkCtx.beginPath();
    pts.forEach((v, i) => {
        const x = i * step;
        const y = H - pad - ((v / 100) * (H - 2 * pad));
        i === 0 ? sparkCtx.moveTo(x, y) : sparkCtx.lineTo(x, y);
    });
    const lastX = (pts.length - 1) * step;
    const lastY = H - pad - ((pts[pts.length - 1] / 100) * (H - 2 * pad));
    sparkCtx.lineTo(lastX, H);
    sparkCtx.lineTo(0, H);
    sparkCtx.closePath();
    sparkCtx.fillStyle = grad;
    sparkCtx.fill();

    // Line
    sparkCtx.beginPath();
    pts.forEach((v, i) => {
        const x = i * step;
        const y = H - pad - ((v / 100) * (H - 2 * pad));
        i === 0 ? sparkCtx.moveTo(x, y) : sparkCtx.lineTo(x, y);
    });
    const lineGrad = sparkCtx.createLinearGradient(0, 0, W, 0);
    lineGrad.addColorStop(0, '#6c5ce7');
    lineGrad.addColorStop(0.5, '#a29bfe');
    lineGrad.addColorStop(1, '#00cec9');
    sparkCtx.strokeStyle = lineGrad;
    sparkCtx.lineWidth = 2;
    sparkCtx.lineJoin = 'round';
    sparkCtx.stroke();

    // Last dot
    sparkCtx.beginPath();
    sparkCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    sparkCtx.fillStyle = '#00e5a0';
    sparkCtx.fill();
}

// ============ SOUND EFFECT ============
function playDetectionSound() {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain); gain.connect(audioCtx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.1);
        gain.gain.setValueAtTime(0.07, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.22);
        osc.start(audioCtx.currentTime);
        osc.stop(audioCtx.currentTime + 0.22);
    } catch(e) { /* AudioContext may need user interaction first */ }
}

// ============ USER ACTIONS ============
async function startTranslation() {
    if (!await initCamera()) return;
    isTranslating = true;
    sessionStartTime = Date.now();
    connectWebSocket();
    initHolistic();  // Uses MediaPipe JS in the browser — no Python dependency!
    btnStart.style.display = 'none';
    btnStop.style.display = 'inline-flex';
    document.getElementById('cameraWrapper').classList.add('active');
    cameraIntro.classList.add('hidden');
    showToast('▶ Translation started!');
}

function stopTranslation() {
    isTranslating = false;
    stopCamera();
    if (ws) { ws.close(); ws = null; }
    btnStart.style.display = 'inline-flex';
    btnStop.style.display = 'none';
    document.getElementById('cameraWrapper').classList.remove('active');
    predictionOverlay.classList.remove('visible');
    cameraIntro.classList.remove('hidden');
    clearSkeleton();
    setStatus('online', 'Translation Paused');
    showToast('⏹ Translation stopped.');
}

function clearSentence() {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'clear' }));
    currentWords = []; resetUI(); showToast('🗑 Cleared.');
}

function resetUI() {
    signWord.textContent = '—';
    confidenceBar.style.width = '0%'; confidenceBar.className = 'confidence-bar';
    confidenceText.textContent = '0%';
    wordsDisplay.innerHTML = '<span class="placeholder-text">Signs will appear here as you gesture...</span>';
    sentenceText.textContent = 'Waiting for signs...'; sentenceText.classList.remove('active');
    bufferFill.style.width = '0%';
    if (bufferLabel) bufferLabel.textContent = '0 / 30';
    if (ringFill) { ringFill.style.strokeDashoffset = RING_TOTAL; }
    if (ringText) ringText.textContent = '0%';
    wordCount.textContent = '0 words';
    predictionOverlay.classList.remove('visible');
    clearSkeleton();
}

function renderWordChips(words) {
    wordsDisplay.innerHTML = words.map((w, i) =>
        `<span class="word-chip" style="animation-delay:${i * 0.04}s">${w.replace(/_/g, ' ')}</span>`
    ).join('');
}

// ============ TYPEWRITER EFFECT ============
function typewriterEffect(text) {
    sentenceText.textContent = '';
    sentenceText.classList.remove('active');
    let i = 0;
    const interval = setInterval(() => {
        sentenceText.textContent += text[i];
        i++;
        if (i >= text.length) {
            clearInterval(interval);
            sentenceText.classList.add('active');
        }
    }, 28);
}

// ============ AI TRANSLATE ============
async function translateWithAI() {
    if (currentWords.length === 0) { showToast('⚠️ No signs detected yet!', true); return; }
    sentenceText.textContent = '✨ AI is thinking...';
    sentenceText.classList.remove('active');
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'nlp', words: currentWords }));
    } else {
        try {
            const res = await fetch('/api/nlp', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ words: currentWords }) });
            const data = await res.json();
            typewriterEffect(data.sentence);
            addToHistory(data.sentence);
            translationCount++;
            showToast('✅ Translation complete!');
        } catch { sentenceText.textContent = 'Error. Is server running?'; showToast('❌ Server unreachable.', true); }
    }
}

// ============ SPEAK ============
async function speakSentence() {
    const text = sentenceText.textContent;
    if (!text || ['Waiting for signs...', '✨ AI is thinking...'].includes(text)) return;
    const speakEl = document.getElementById('speakText');
    const waves = document.getElementById('audioWaves');
    speakEl.textContent = 'Speaking...'; waves.style.display = 'flex';
    const done = () => { speakEl.textContent = 'Speak Aloud'; waves.style.display = 'none'; };
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.9; u.pitch = 1.0; u.volume = 1.0;
        u.onend = done; u.onerror = done;
        speechSynthesis.speak(u);
        showToast('🔊 Speaking...');
    } else {
        try { await fetch('/api/speak', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) }); } catch(e) {}
        setTimeout(done, 2500);
    }
}

// ============ COPY ============
function copySentence() {
    const text = sentenceText.textContent;
    if (!text || text === 'Waiting for signs...') return;
    navigator.clipboard.writeText(text).then(() => showToast('📋 Copied!')).catch(() => showToast('❌ Copy failed', true));
}

// ============ HISTORY ============
function addToHistory(sentence) {
    if (!sentence || sentence === 'Waiting for signs...') return;
    const timeStr = new Date().toLocaleTimeString('en-IN', { hour:'2-digit', minute:'2-digit' });
    translationHistory.unshift({ sentence, time: timeStr });
    if (translationHistory.length > 25) translationHistory.pop();
    renderHistory();
}
function renderHistory() {
    historyCount.textContent = translationHistory.length;
    if (translationHistory.length === 0) { historyList.innerHTML = '<span class="placeholder-text">Translations will be logged here...</span>'; return; }
    historyList.innerHTML = translationHistory.map((item, i) => `
        <div class="history-item" style="animation-delay:${i*0.03}s">
            <span class="history-sentence">${item.sentence}</span>
            <span class="history-time">${item.time}</span>
        </div>`).join('');
}

// ============ EXPORT TXT ============
function exportTranscript() {
    if (translationHistory.length === 0) { showToast('⚠️ No history yet!', true); return; }
    let c = `ISL Translator — Session Transcript\n${'='.repeat(36)}\n`;
    c += `Date: ${new Date().toLocaleDateString('en-IN')}\nSigns: ${sessionSignCount}  Translations: ${translationHistory.length}\n\n`;
    [...translationHistory].reverse().forEach(i => c += `[${i.time}] ${i.sentence}\n`);
    const url = URL.createObjectURL(new Blob([c], { type:'text/plain' }));
    const a = Object.assign(document.createElement('a'), { href:url, download:`ISL_${new Date().toISOString().slice(0,10)}.txt` });
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    showToast('📥 Transcript exported!');
}

// ============ EXPORT PDF ============
function exportPDF() {
    if (translationHistory.length === 0) { showToast('⚠️ No history yet!', true); return; }
    if (typeof window.jspdf === 'undefined') { showToast('⚠️ PDF library loading...', true); return; }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(20);
    doc.setTextColor(108, 92, 231);
    doc.text('ISL Translator — Session Transcript', 14, 20);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(120, 120, 130);
    doc.text(`Date: ${new Date().toLocaleDateString('en-IN')}   Signs: ${sessionSignCount}   Translations: ${translationHistory.length}`, 14, 30);
    doc.setDrawColor(108, 92, 231);
    doc.line(14, 34, 196, 34);
    let y = 44;
    doc.setTextColor(30, 30, 50);
    [...translationHistory].reverse().forEach(item => {
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(8);
        doc.setTextColor(108, 92, 231);
        doc.text(`[${item.time}]`, 14, y);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(11);
        doc.setTextColor(30, 30, 50);
        const lines = doc.splitTextToSize(item.sentence, 165);
        doc.text(lines, 38, y);
        y += lines.length * 7 + 4;
        if (y > 270) { doc.addPage(); y = 20; }
    });
    doc.save(`ISL_Transcript_${new Date().toISOString().slice(0, 10)}.pdf`);
    showToast('🗒️ PDF exported!');
}

// ============ SIGN GALLERY ============
function buildGalleryGrid(signs) {
    const grid = document.getElementById('galleryGrid');
    grid.innerHTML = signs.map(s => `<span class="gallery-chip">${s.replace(/_/g, ' ')}</span>`).join('');
    document.getElementById('searchCount').textContent = `${signs.length} signs`;
}

function filterGallery() {
    const q = document.getElementById('gallerySearch').value.toLowerCase().trim();
    const chips = document.querySelectorAll('.gallery-chip');
    let shown = 0;
    chips.forEach(c => {
        const match = c.textContent.toLowerCase().includes(q);
        c.classList.toggle('hidden', !match);
        if (match) shown++;
    });
    document.getElementById('searchCount').textContent = `${shown} signs`;
}

function openGalleryModal() {
    document.getElementById('galleryModal').classList.add('active');
    if (allSigns.length === 0) fetchSigns();
}

// ============ MODEL MODAL ============
function openModelModal() { document.getElementById('modelModal').classList.add('active'); }

// ============ ANALYTICS MODAL ============
function openAnalyticsModal() {
    const modal = document.getElementById('analyticsModal');
    // Duration
    const elapsed = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
    document.getElementById('aDuration').textContent = `${Math.floor(elapsed/60)}:${String(elapsed%60).padStart(2,'0')}`;
    document.getElementById('aSignCount').textContent = sessionSignCount;
    document.getElementById('aTranslations').textContent = translationCount;
    // Top sign
    const sorted = Object.entries(signFrequency).sort((a,b) => b[1]-a[1]);
    document.getElementById('aTopSign').textContent = sorted[0] ? sorted[0][0].replace(/_/g,' ') : '—';
    // Signs per minute
    const mins = elapsed / 60 || 1;
    document.getElementById('aSPM').textContent = (sessionSignCount / mins).toFixed(1);
    // Avg confidence
    const avg = confidenceHistory.length ? Math.round(confidenceHistory.reduce((a,b)=>a+b,0)/confidenceHistory.length) : 0;
    document.getElementById('aAvgConf').textContent = `${avg}%`;
    // Bar chart (top 5)
    const chartDiv = document.getElementById('analyticsChart');
    const top5 = sorted.slice(0, 5);
    const maxCount = top5[0]?.[1] || 1;
    chartDiv.innerHTML = top5.length ? top5.map(([sign, count]) => `
        <div class="analytics-bar-row">
            <span class="analytics-bar-label">${sign.replace(/_/g,' ')}</span>
            <div class="analytics-bar-track">
                <div class="analytics-bar-fill" style="width:${(count/maxCount)*100}%"></div>
            </div>
            <span class="analytics-bar-count">${count}</span>
        </div>`).join('') : '<span style="color:var(--text-muted);font-size:0.8rem">No signs detected yet.</span>';
    modal.classList.add('active');
}

// ============ MODAL HELPERS ============
function closeModal(id) { document.getElementById(id).classList.remove('active'); }
function handleModalClick(e, id) { if (e.target === document.getElementById(id)) closeModal(id); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') { document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); endTour(); } });

// ============ VOICE COMMANDS ============
function toggleVoiceCommand() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) { showToast('❌ Voice not supported in this browser', true); return; }
    if (isVoiceActive) {
        recognition.stop(); isVoiceActive = false;
        document.getElementById('voiceStatus').style.display = 'none';
        document.getElementById('btnVoice').classList.remove('btn-primary');
        showToast('🎤 Voice commands OFF');
    } else {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous = true; recognition.interimResults = false; recognition.lang = 'en-US';
        recognition.onresult = e => {
            const t = e.results[e.results.length-1][0].transcript.toLowerCase().trim();
            handleVoiceCmd(t);
        };
        recognition.onend = () => { if (isVoiceActive) recognition.start(); };
        recognition.start(); isVoiceActive = true;
        document.getElementById('voiceStatus').style.display = 'flex';
        document.getElementById('btnVoice').classList.add('btn-primary');
        showToast('🎤 Voice ON — say "start", "stop", "translate"');
    }
}

function handleVoiceCmd(text) {
    const el = document.getElementById('voiceStatusText');
    el.textContent = `Heard: "${text}"`;
    if (text.includes('start')) startTranslation();
    else if (text.includes('stop') || text.includes('pause')) stopTranslation();
    else if (text.includes('clear') || text.includes('reset')) clearSentence();
    else if (text.includes('translate') || text.includes('convert')) translateWithAI();
    else if (text.includes('speak') || text.includes('read')) speakSentence();
    else if (text.includes('export') || text.includes('save')) exportTranscript();
    else if (text.includes('gallery') || text.includes('signs')) openGalleryModal();
    setTimeout(() => { el.textContent = 'Listening... say "start", "stop", "translate"'; }, 2000);
}

// ============ ONBOARDING TOUR ============
const TOUR_STEPS = [
    { icon:'👋', title:'Welcome to ISL Translator!', desc:'An AI-powered real-time Indian Sign Language translator. Let\'s take a 30-second tour!' },
    { icon:'📸', title:'Camera Feed', desc:'Your webcam feed appears here. The AI detects your hands using MediaPipe and draws a live skeleton overlay.' },
    { icon:'⚡', title:'Start Translating', desc:'Click "Start Translation" or press S. Hold each ISL sign steady — the ring fills as the buffer loads.' },
    { icon:'🧠', title:'AI Translation', desc:'Signs build up as word chips. Click ✨ Translate for AI-powered natural sentence construction, then 🔊 Speak.' },
    { icon:'🎉', title:'You\'re Ready!', desc:'Explore 318+ signs in the 📚 Gallery, check 📊 Analytics after your session, and use 🎤 Voice Commands!' },
];

function startTour() {
    tourStep = 0;
    document.getElementById('tourOverlay').style.display = 'flex';
    renderTourStep();
}

function renderTourStep() {
    const step = TOUR_STEPS[tourStep];
    document.getElementById('tourIcon').textContent = step.icon;
    document.getElementById('tourTitle').textContent = step.title;
    document.getElementById('tourDesc').textContent = step.desc;
    document.getElementById('tourNextBtn').textContent = tourStep === TOUR_STEPS.length - 1 ? 'Finish 🚀' : 'Next →';
    // Dots
    document.getElementById('tourDots').innerHTML = TOUR_STEPS.map((_, i) =>
        `<span class="tour-dot ${i === tourStep ? 'active' : ''}"></span>`).join('');
}

function nextTourStep() {
    tourStep++;
    if (tourStep >= TOUR_STEPS.length) { endTour(); return; }
    renderTourStep();
}

function endTour() {
    document.getElementById('tourOverlay').style.display = 'none';
    localStorage.setItem('isl_tour_done', '1');
}

// ============ TOAST ============
function showToast(msg, isError = false) {
    const toast = document.getElementById('toast');
    const toastMsg = document.getElementById('toastMsg');
    if (toastTimeout) clearTimeout(toastTimeout);
    toastMsg.textContent = msg;
    toast.style.borderColor = isError ? 'rgba(255,118,117,0.3)' : 'rgba(108,92,231,0.3)';
    toast.classList.add('show');
    toastTimeout = setTimeout(() => toast.classList.remove('show'), 3200);
}

// ============ KEYBOARD SHORTCUTS ============
document.addEventListener('keydown', e => {
    if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === 's' && !isTranslating) startTranslation();
    else if (key === 'x' && isTranslating) stopTranslation();
    else if (key === 'c') clearSentence();
    else if (key === 't') translateWithAI();
    else if (key === ' ') { e.preventDefault(); speakSentence(); }
    else if (key === 'm') toggleMirror();
    else if (key === 'g') openGalleryModal();
    else if (key === 'a') openAnalyticsModal();
});
