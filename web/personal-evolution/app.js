const state = {
  date: "",
  review: null,
  candidates: [],
  memories: [],
};

const nodes = {};

function todayIsoDate() {
  return new Date().toISOString().slice(0, 10);
}

function normalizeList(payload, keys) {
  if (Array.isArray(payload)) {
    return payload;
  }

  for (const key of keys) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }

  return [];
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

function text(value, fallback = "Untitled") {
  return value == null || value === "" ? fallback : String(value);
}

function createEl(tagName, className, content) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (content !== undefined) {
    element.textContent = content;
  }
  return element;
}

function actionRequest() {
  return {
    method: "POST",
    body: JSON.stringify({}),
  };
}

function renderReview() {
  const review = state.review || {};
  const items = [
    ["Date", text(review.date, state.date || todayIsoDate())],
    ["Events", normalizeList(review.events, []).length],
    ["Evidence", normalizeList(review.evidence, []).length],
    ["Candidates", normalizeList(review.candidates, []).length],
  ];

  const cards = items.map(([label, value]) => {
    const card = createEl("div", "summary-item");
    card.append(
      createEl("span", "meta", label),
      createEl("strong", "", text(value)),
    );
    return card;
  });

  nodes.reviewSummary.replaceChildren(...cards);
}

function renderEmpty(container, message) {
  container.replaceChildren(createEl("p", "empty", message));
}

function renderCandidateCard(candidate) {
  const id = text(candidate.candidate_id, "");
  const card = createEl("article", "list-item");
  const body = createEl("div");
  const actions = createEl("div", "item-actions");
  const approveButton = createEl("button", "primary", "Approve");
  const rejectButton = createEl("button", "", "Reject");

  approveButton.type = "button";
  rejectButton.type = "button";
  approveButton.addEventListener("click", () => approveCandidate(id));
  rejectButton.addEventListener("click", () => rejectCandidate(id));

  body.append(
    createEl("span", "tag", text(candidate.memory_type, "candidate")),
    createEl("h3", "", text(candidate.claim)),
    createEl("p", "", text(candidate.rationale || candidate.detail || candidate.content, "No detail provided.")),
    createEl("p", "meta", `Status: ${text(candidate.status, "unknown")}`),
  );
  actions.append(approveButton, rejectButton);
  card.append(body, actions);
  return card;
}

function renderMemoryCard(memory) {
  const id = text(memory.memory_id, "");
  const card = createEl("article", "list-item");
  const body = createEl("div");
  const actions = createEl("div", "item-actions");
  const revokeButton = createEl("button", "danger", "Revoke");

  revokeButton.type = "button";
  revokeButton.addEventListener("click", () => revokeMemory(id));

  body.append(
    createEl("span", "tag", text(memory.memory_type, "memory")),
    createEl("h3", "", text(memory.title || memory.summary || memory.content)),
    createEl("p", "", text(memory.note || memory.detail || memory.approved_at, "No detail provided.")),
    createEl("p", "meta", `Status: ${text(memory.status, "unknown")}`),
  );
  actions.append(revokeButton);
  card.append(body, actions);
  return card;
}

function renderCandidates() {
  nodes.candidateCount.textContent = `${state.candidates.length} candidates`;

  if (state.candidates.length === 0) {
    renderEmpty(nodes.candidateList, "No candidate learnings are waiting.");
    return;
  }

  nodes.candidateList.replaceChildren(...state.candidates.map(renderCandidateCard));
}

function renderMemories() {
  nodes.memoryCount.textContent = `${state.memories.length} memories`;

  if (state.memories.length === 0) {
    renderEmpty(nodes.memoryList, "No active memories are recorded.");
    return;
  }

  nodes.memoryList.replaceChildren(...state.memories.map(renderMemoryCard));
}

function renderAll() {
  renderReview();
  renderCandidates();
  renderMemories();
}

async function loadReview() {
  state.date = nodes.reviewDate.value || todayIsoDate();
  state.review = await requestJson(`/personal-evolution/review/${state.date}`);
  renderReview();
}

async function loadCandidates() {
  const payload = await requestJson("/personal-evolution/candidates");
  state.candidates = normalizeList(payload, ["candidates", "items", "results"]);
  renderCandidates();
  renderReview();
}

async function loadMemories() {
  const payload = await requestJson("/personal-evolution/memories");
  state.memories = normalizeList(payload, ["memories", "items", "results"]);
  renderMemories();
}

async function approveCandidate(candidateId) {
  await requestJson(`/personal-evolution/candidates/${candidateId}/approve`, actionRequest());
  await loadCandidates();
  await loadMemories();
}

async function rejectCandidate(candidateId) {
  await requestJson(`/personal-evolution/candidates/${candidateId}/reject`, actionRequest());
  await loadCandidates();
}

async function revokeMemory(memoryId) {
  await requestJson(`/personal-evolution/memories/${memoryId}/revoke`, actionRequest());
  await loadMemories();
}

async function bootstrap() {
  nodes.reviewDate = document.querySelector("#review-date");
  nodes.reviewSummary = document.querySelector("#review-summary");
  nodes.refreshReview = document.querySelector("#refresh-review");
  nodes.candidateCount = document.querySelector("#candidate-count");
  nodes.candidateList = document.querySelector("#candidate-list");
  nodes.memoryCount = document.querySelector("#memory-count");
  nodes.memoryList = document.querySelector("#memory-list");

  nodes.reviewDate.value = todayIsoDate();
  nodes.reviewDate.addEventListener("change", loadReview);
  nodes.refreshReview.addEventListener("click", loadReview);

  renderAll();

  try {
    await Promise.all([loadReview(), loadCandidates(), loadMemories()]);
  } catch (error) {
    renderEmpty(nodes.reviewSummary, `Unable to load review data: ${error.message}`);
  }
}

window.approveCandidate = approveCandidate;
window.rejectCandidate = rejectCandidate;
window.revokeMemory = revokeMemory;

document.addEventListener("DOMContentLoaded", bootstrap);
