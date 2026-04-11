const DEFAULT_API = 'https://rwolfe26-phishing-detector.hf.space';

const apiInput   = document.getElementById('api-url');
const saveBtn    = document.getElementById('save-btn');
const feedback   = document.getElementById('feedback');
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// ── Load saved URL and ping the API ──────────────────────────────────────────
chrome.storage.sync.get('apiUrl', async ({ apiUrl = DEFAULT_API }) => {
  apiInput.value = apiUrl;
  await pingApi(apiUrl);
});

// ── Save ──────────────────────────────────────────────────────────────────────
saveBtn.addEventListener('click', async () => {
  const url = apiInput.value.trim().replace(/\/$/, '') || DEFAULT_API;
  apiInput.value = url;

  await chrome.storage.sync.set({ apiUrl: url });
  feedback.textContent = 'Saved.';
  feedback.className   = 'feedback ok';
  setTimeout(() => { feedback.textContent = ''; }, 2000);

  await pingApi(url);
});

// ── Ping health endpoint ──────────────────────────────────────────────────────
async function pingApi(baseUrl) {
  statusDot.className  = 'status-dot';
  statusText.textContent = 'Checking API…';

  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(6000) });
    if (res.ok) {
      const data = await res.json();
      if (data.model_loaded) {
        setStatus(true, 'API online · model loaded');
      } else {
        setStatus(false, 'API online · model not loaded');
      }
    } else {
      setStatus(false, `API error ${res.status}`);
    }
  } catch {
    setStatus(false, 'Cannot reach API');
  }
}

function setStatus(online, msg) {
  statusDot.className    = online ? 'status-dot' : 'status-dot offline';
  statusText.textContent = msg;
}
