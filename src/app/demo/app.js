const quickQueries = [
  "vegan iced espresso under 15g sugar",
  "hot brewed coffee with strong caffeine under $4.0",
  "caffeine free herbal tea without milk",
  "fruity refresher under 25g sugar",
  "blended drink with no caffeine"
];

const els = {
  serviceBadge: document.getElementById("serviceBadge"),
  presetBadge: document.getElementById("presetBadge"),
  queryInput: document.getElementById("queryInput"),
  topKInput: document.getElementById("topKInput"),
  explainTopNInput: document.getElementById("explainTopNInput"),
  explainToggle: document.getElementById("explainToggle"),
  recommendBtn: document.getElementById("recommendBtn"),
  resetBtn: document.getElementById("resetBtn"),
  emptyState: document.getElementById("emptyState"),
  resultBody: document.getElementById("resultBody"),
  errorBox: document.getElementById("errorBox"),
  constraintsJson: document.getElementById("constraintsJson"),
  recommendationList: document.getElementById("recommendationList"),
  rawJson: document.getElementById("rawJson"),
  candidateCount: document.getElementById("candidateCount"),
  returnedCount: document.getElementById("returnedCount"),
  latencyLabel: document.getElementById("latencyLabel"),
  quickQueries: document.getElementById("quickQueries")
};

function setBadge(el, text, type) {
  el.textContent = text;
  el.className = `badge ${type}`;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.remove("hidden");
}

function clearError() {
  els.errorBox.textContent = "";
  els.errorBox.classList.add("hidden");
}

function resetOutput() {
  els.resultBody.classList.add("hidden");
  els.emptyState.classList.remove("hidden");
  clearError();
  els.constraintsJson.textContent = "";
  els.rawJson.textContent = "";
  els.recommendationList.innerHTML = "";
  els.candidateCount.textContent = "-";
  els.returnedCount.textContent = "-";
  els.latencyLabel.textContent = "Latency: -";
}

function renderRecommendations(data, includeExplanations) {
  const items = Array.isArray(data.recommendations) ? data.recommendations : [];
  const explanationMap = new Map();

  if (includeExplanations && Array.isArray(data.top_explanations)) {
    for (const item of data.top_explanations) {
      explanationMap.set(item.product_id, item);
    }
  }

  els.recommendationList.innerHTML = items
    .map((item) => {
      const explanation = explanationMap.get(item.product_id);
      const reasons = explanation && Array.isArray(explanation.reasons)
        ? `<ul class="reason-list">${explanation.reasons
            .slice(0, 4)
            .map((reason) => `<li>${escapeHtml(reason)}</li>`)
            .join("")}</ul>`
        : "";

      return `
        <li class="recommendation-item">
          <div class="item-head">
            <p class="product-name">${escapeHtml(item.name)} (${escapeHtml(item.product_id)})</p>
            <span class="rank-pill">Rank ${item.rank}</span>
          </div>
          <p class="product-meta">
            ${escapeHtml(item.category)} • ${escapeHtml(item.temperature)} • $${Number(item.price).toFixed(2)}
          </p>
          ${reasons}
        </li>
      `;
    })
    .join("");
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) {
      throw new Error(`health request failed: ${res.status}`);
    }

    const data = await res.json();
    const mode = data.mode === "online" ? "Online" : "Offline";
    const stateType = data.mode === "online" ? "good" : "warn";

    setBadge(els.serviceBadge, `Service: ${mode}`, stateType);
    setBadge(els.presetBadge, `Preset: ${data.model_preset || "-"}`, "neutral");
  } catch (err) {
    setBadge(els.serviceBadge, "Service unavailable", "warn");
    setBadge(els.presetBadge, "Preset: -", "neutral");
  }
}

async function runRecommendation() {
  clearError();

  const queryText = els.queryInput.value.trim();
  if (!queryText) {
    showError("Please enter a query before submitting.");
    return;
  }

  const topK = Math.max(1, Math.min(25, Number(els.topKInput.value) || 10));
  const explanationTopN = Math.max(1, Math.min(10, Number(els.explainTopNInput.value) || 3));
  const includeExplanations = Boolean(els.explainToggle.checked);

  els.recommendBtn.disabled = true;
  els.recommendBtn.textContent = "Running...";

  const payload = {
    query_text: queryText,
    top_k: topK,
    include_explanations: includeExplanations,
    explanation_top_n: explanationTopN
  };

  const start = performance.now();

  try {
    const res = await fetch("/v1/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const elapsed = Math.round(performance.now() - start);
    els.latencyLabel.textContent = `Latency: ${elapsed} ms`;

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `request failed: ${res.status}`);
    }

    const data = await res.json();

    els.emptyState.classList.add("hidden");
    els.resultBody.classList.remove("hidden");

    els.candidateCount.textContent = data.candidate_count ?? "-";
    els.returnedCount.textContent = Array.isArray(data.product_ids) ? data.product_ids.length : 0;
    els.constraintsJson.textContent = pretty(data.constraints || {});
    els.rawJson.textContent = pretty(data);

    renderRecommendations(data, includeExplanations);
  } catch (err) {
    showError(`Request failed: ${err.message}`);
  } finally {
    els.recommendBtn.disabled = false;
    els.recommendBtn.textContent = "Run Recommendation";
  }
}

function mountQuickQueries() {
  els.quickQueries.innerHTML = quickQueries
    .map((query) => `<button class="quick-chip" type="button">${escapeHtml(query)}</button>`)
    .join("");

  Array.from(els.quickQueries.querySelectorAll(".quick-chip")).forEach((btn, index) => {
    btn.addEventListener("click", () => {
      els.queryInput.value = quickQueries[index];
      els.queryInput.focus();
    });
  });
}

function setupEvents() {
  els.recommendBtn.addEventListener("click", runRecommendation);
  els.resetBtn.addEventListener("click", () => {
    els.queryInput.value = "";
    resetOutput();
  });

  els.queryInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      runRecommendation();
    }
  });
}

function init() {
  mountQuickQueries();
  setupEvents();
  resetOutput();
  loadHealth();
}

init();
