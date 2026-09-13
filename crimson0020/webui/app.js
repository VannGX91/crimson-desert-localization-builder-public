"use strict";

const fieldNames = [
  "source", "font_source", "json", "unpack_output", "repack_output",
  "auto_old", "auto_new", "auto_work", "auto_translate_input", "auto_output",
  "auto_merge_input", "auto_merge_output",
  "font_unpack_output", "font_repack_input", "font_repack_output",
  "install_source", "game"
];

const DEFAULT_TRANSLATION_PROVIDER = "ai_router_free";
const TRANSLATION_PROVIDER_LABELS = Object.freeze({
  ai_router_free: "AI Router (Free)",
  google_free: "Google-free",
  "bing-web-free": "Bing-free",
  "alibaba-web-free": "Alibaba-free",
  "modernmt-web-free": "ModernMT-free",
  "youdao-web-free": "YouDao-free",
  "openai-compatible": "Custom (OpenAI-compatible)"
});
const TRANSLATION_PROVIDER_KEYS = Object.freeze(Object.keys(TRANSLATION_PROVIDER_LABELS));
const TRANSLATION_ROUTE_PROVIDER_KEYS = Object.freeze({
  "google-web-free": "google_free"
});

let currentState = null;
let pollTimer = null;
let toastTimer = null;
let knowledgePage = 1;
let knowledgeLastPage = 1;
const knowledgePageSize = 25;
let knowledgePathMode = null;
let knowledgePathTrigger = null;
let customConfigDirty = false;

const CUSTOM_CONFIG_INPUT_IDS = Object.freeze([
  "translationBaseUrl",
  "translationModel",
  "translationApiKey",
  "translationBatchItems",
  "translationBatchCharacters",
  "translationConcurrency",
  "translationTimeout"
]);

const byId = (id) => document.getElementById(id);
const all = (selector) => Array.from(document.querySelectorAll(selector));

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { toast.hidden = true; }, 4200);
}

function activatePage(key) {
  all("[data-page]").forEach((element) => {
    const active = element.dataset.page === key;
    element.classList.toggle("active", active);
    if (element.classList.contains("nav-item")) {
      element.setAttribute("aria-current", active ? "page" : "false");
    }
  });
  const page = byId(key);
  if (!page) return;
  const nav = document.querySelector(`.nav-item[data-page="${key}"] b`);
  byId("contextTitle").textContent = nav ? nav.textContent : key;
  byId("contextMeta").textContent = page.dataset.context || "";
}

function applyTheme(key, themes) {
  document.documentElement.dataset.theme = key;
  const selected = themes.find((theme) => theme.key === key);
  byId("themeName").textContent = selected ? selected.name : key;
  all("[data-theme-choice]").forEach((button) => {
    const active = button.dataset.themeChoice === key;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", String(active));
  });
}

function renderMonitor(entries) {
  const log = byId("monitorLog");
  const fragment = document.createDocumentFragment();
  entries.forEach((entry) => {
    const line = document.createElement("div");
    line.className = "log-line";
    line.textContent = entry;
    fragment.appendChild(line);
  });
  log.replaceChildren(fragment);
  log.scrollTop = log.scrollHeight;
}

function renderStatus(status) {
  const card = document.querySelector(".status-card");
  card.dataset.level = status.level;
  byId("statusIcon").textContent = status.icon;
  byId("statusTitle").textContent = status.title;
  byId("statusText").textContent = status.body;
  const details = byId("statusDetails");
  const detail = status.detail || "";
  details.hidden = !detail;
  byId("statusDetail").textContent = detail;
}

function renderFields(fields) {
  fieldNames.forEach((name) => {
    const element = byId(`field-${name}`);
    if (element) {
      element.value = fields[name] || "";
      element.title = fields[name] || "";
    }
  });
  byId("field-uninstall-game").value = fields.game || "";
  byId("field-uninstall-game").title = fields.game || "";
}

function renderLanguages(state) {
  const select = byId("languageSelect");
  if (select.options.length !== state.languages.length) {
    const options = state.languages.map((language) => {
      const option = document.createElement("option");
      option.value = language;
      option.textContent = language;
      return option;
    });
    select.replaceChildren(...options);
  }
  select.value = state.language;
}

function renderTranslationProviders(state) {
  const select = byId("translationProvider");
  const providers = state.translation_providers.filter(
    (provider) => TRANSLATION_PROVIDER_KEYS.includes(provider.key)
  );
  if (select.options.length !== providers.length) {
    const options = providers.map((provider) => {
      const option = document.createElement("option");
      option.value = provider.key;
      option.textContent = TRANSLATION_PROVIDER_LABELS[provider.key];
      return option;
    });
    select.replaceChildren(...options);
  }
  const providerKey = TRANSLATION_PROVIDER_KEYS.includes(state.translation_provider)
    ? state.translation_provider
    : DEFAULT_TRANSLATION_PROVIDER;
  select.value = providerKey;
  const selected = providers.find(
    (provider) => provider.key === providerKey
  );
  byId("providerDescription").textContent = selected ? selected.description : "";
  renderTranslationConfig(state, providerKey);
}

function renderTranslationConfig(state, providerKey) {
  const panel = byId("translationCustomConfig");
  const isCustom = providerKey === "openai-compatible";
  panel.hidden = !isCustom;
  if (!isCustom) return;

  const config = state.translation_config || {};
  const baseUrl = byId("translationBaseUrl");
  const model = byId("translationModel");
  const apiKey = byId("translationApiKey");
  const batchItems = byId("translationBatchItems");
  const batchCharacters = byId("translationBatchCharacters");
  const concurrency = byId("translationConcurrency");
  const timeout = byId("translationTimeout");
  if (!customConfigDirty) {
    baseUrl.value = config.base_url || "";
    model.value = config.model || "";
    batchItems.value = config.max_batch_items || 24;
    batchCharacters.value = config.max_batch_characters || 4000;
    concurrency.value = config.concurrency || 1;
    timeout.value = config.timeout_seconds || 120;
  }
  apiKey.placeholder = config.api_key_configured
    ? "Đã cấu hình — để trống để giữ key hiện tại"
    : "Nhập API key; localhost có thể để trống";
  const keyStatus = config.api_key_configured
    ? "Start dùng đúng cấu hình đang hiển thị; API key hiện đã có trong runtime. File config chỉ thay đổi khi bạn bấm Lưu config."
    : "Start dùng đúng cấu hình đang hiển thị. API key chưa cấu hình; localhost có thể không cần key.";
  if (!customConfigDirty) byId("translationConfigStatus").textContent = keyStatus;
}

function readCustomConfigDraft() {
  return {
    base_url: byId("translationBaseUrl").value,
    model: byId("translationModel").value,
    api_key: byId("translationApiKey").value,
    max_batch_items: byId("translationBatchItems").value,
    max_batch_characters: byId("translationBatchCharacters").value,
    concurrency: byId("translationConcurrency").value,
    timeout_seconds: byId("translationTimeout").value
  };
}

function setCustomConfigFeedback(message) {
  byId("translationConfigStatus").textContent = message;
  showToast(message);
}

function setTranslationMetric(id, value, fallback = "0") {
  const element = byId(id);
  element.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
}

function translationRouteLabel(routeKey) {
  const providerKey = TRANSLATION_ROUTE_PROVIDER_KEYS[routeKey] || routeKey;
  return TRANSLATION_PROVIDER_LABELS[providerKey] || routeKey;
}

function renderTranslationJob(job, ready, busy) {
  job = job || {};
  const checkpointReuse = Number(job.resumed || 0);
  const tmReuse = Number(job.tm_reused || job.knowledge_reused || 0);
  const newlyTranslated = Math.max(0, Number(job.translated || 0));
  setTranslationMetric("translationTotal", job.total);
  setTranslationMetric("translationSaved", job.saved);
  setTranslationMetric("translationCheckpointReuse", checkpointReuse);
  setTranslationMetric("translationTmReuse", tmReuse);
  setTranslationMetric("translationNewlyTranslated", newlyTranslated);
  setTranslationMetric("translationPending", job.pending);
  setTranslationMetric("translationCurrentRoute", translationRouteLabel(job.current_route), "—");
  setTranslationMetric("translationCurrentModel", job.current_model, "—");
  setTranslationMetric("translationFallbacks", job.fallbacks);
  const cooldown = Number(job.cooldown_seconds || 0);
  setTranslationMetric("translationCooldown", cooldown > 0 ? `${cooldown.toFixed(1)} s` : "—", "—");
  setTranslationMetric("translationFailureReason", job.failure_summary, "—");

  const button = byId("autoTranslateAction");
  const jobState = job.state || "idle";
  if (jobState === "running" || jobState === "pausing") {
    button.firstChild.textContent = "Pause safely";
  } else if (["paused", "completed_with_pending"].includes(jobState)) {
    button.firstChild.textContent = "Resume translation";
  } else {
    button.firstChild.textContent = "Start translation";
  }
  button.disabled = !ready || jobState === "pausing" || (busy && jobState !== "running");
  byId("translationConnectionStatus").textContent = `Trạng thái: ${jobState}.`;
}

function renderTranslationOffline(error) {
  const detail = error && (error.message || String(error));
  byId("translationConnectionStatus").textContent = detail
    ? `Không thể đọc trạng thái dịch: ${detail}`
    : "Không thể đọc trạng thái dịch.";
}

function renderKnowledgeItems(items) {
  const container = byId("knowledgeResults");
  const fragment = document.createDocumentFragment();
  (items || []).forEach((item) => {
    const card = document.createElement("article");
    card.className = "knowledge-item";

    const source = document.createElement("p");
    source.className = "knowledge-source";
    source.textContent = item.source_text || "";

    const targetLabel = document.createElement("label");
    targetLabel.textContent = "Bản dịch";
    const target = document.createElement("textarea");
    target.rows = 2;
    target.value = item.target_text || "";
    targetLabel.appendChild(target);

    const actions = document.createElement("div");
    actions.className = "knowledge-item-actions";
    const correct = document.createElement("button");
    correct.type = "button";
    correct.className = "ghost";
    correct.textContent = "Lưu sửa";
    correct.addEventListener("click", async () => {
      applyResponse(await window.pywebview.api.correct_translation_memory(item.segment_id, target.value));
      await loadTranslationKnowledge();
    });
    const disable = document.createElement("button");
    disable.type = "button";
    disable.className = "ghost";
    disable.textContent = "Tắt mục này";
    disable.addEventListener("click", async () => {
      applyResponse(await window.pywebview.api.disable_translation_memory(item.segment_id));
      await loadTranslationKnowledge();
    });
    actions.append(correct, disable);
    card.append(source, targetLabel, actions);
    fragment.appendChild(card);
  });
  if (!fragment.childNodes.length) {
    const empty = document.createElement("p");
    empty.className = "knowledge-empty";
    empty.textContent = "Không có mục phù hợp.";
    fragment.appendChild(empty);
  }
  container.replaceChildren(fragment);
}

function syncKnowledgePaginationDisabled(busy) {
  byId("knowledgePrev").disabled = busy || knowledgePage <= 1;
  byId("knowledgeNext").disabled = busy || knowledgePage >= knowledgeLastPage;
}

function renderTranslationKnowledge(knowledge) {
  knowledge = knowledge || {};
  const totals = knowledge.totals || {};
  const segments = Number(totals.segments || 0);
  const terms = Number(totals.terms || 0);
  byId("knowledgeSummary").textContent = `Đã ghi nhớ ${segments} bản dịch · ${terms} thuật ngữ.`;
  if (Number.isInteger(knowledge.page)) knowledgePage = knowledge.page;
  renderKnowledgeItems(knowledge.items || []);
  const totalCount = Number(knowledge.total_count || 0);
  knowledgeLastPage = Math.max(1, Math.ceil(totalCount / knowledgePageSize));
  byId("knowledgePageStatus").textContent = `Trang ${knowledgePage} / ${knowledgeLastPage}`;
  syncKnowledgePaginationDisabled(Boolean(currentState && currentState.busy));
}

function hasPaginatedTranslationKnowledge(knowledge) {
  return Boolean(
    knowledge && (
      Array.isArray(knowledge.items)
      || Number.isInteger(knowledge.page)
      || Object.prototype.hasOwnProperty.call(knowledge, "total_count")
    )
  );
}

async function loadTranslationKnowledge() {
  const query = byId("knowledgeSearch").value.trim();
  const response = query
    ? await window.pywebview.api.search_translation_knowledge(query, knowledgePage, knowledgePageSize)
    : await window.pywebview.api.get_translation_knowledge(knowledgePage, knowledgePageSize);
  applyResponse(response);
  if (response.knowledge) renderTranslationKnowledge(response.knowledge);
}

function renderTranslationReadiness(state) {
  const ready = Boolean(state.translation_input_ready);
  const message = byId("translationWorkReadiness");
  message.dataset.ready = String(ready);
  message.textContent = ready
    ? "File cần dịch đã sẵn sàng. Bạn có thể bắt đầu hoặc tiếp tục."
    : "Chọn file cần dịch hoặc hoàn tất bước 1 trước khi dịch.";
  renderTranslationJob(state.translation_job, ready, state.busy);

  const mergeReady = Boolean(state.translation_merge_input_ready);
  const structureValid = Boolean(state.translation_structure_valid);
  const fixAvailable = Boolean(state.translation_fix_available);
  const pendingCount = Number(state.translation_pending_count || 0);
  const mergeMessage = byId("translationMergeReadiness");
  const validateButton = byId("validateTranslationAction");
  const fixButton = byId("fixTranslationAction");
  const mergeButton = byId("mergeTranslationAction");
  mergeMessage.dataset.ready = String(structureValid);
  if (!mergeReady) {
    mergeMessage.textContent = "Chọn file câu đã dịch để kiểm tra và ghép.";
  } else if (structureValid) {
    mergeMessage.textContent = "Cấu trúc hợp lệ và đã đủ bản dịch. Có thể ghép vào JSON chính.";
  } else if (pendingCount > 0 && !fixAvailable) {
    mergeMessage.textContent = `Cấu trúc ổn, còn ${pendingCount} câu chưa dịch. Hãy đổi dịch vụ và tiếp tục bước 2.`;
  } else if (fixAvailable) {
    mergeMessage.textContent = "Cấu trúc chưa hợp lệ. Hãy dùng Fix cấu trúc rồi kiểm tra lại.";
  } else {
    mergeMessage.textContent = "Hãy kiểm tra cấu trúc trước khi ghép.";
  }
  validateButton.disabled = state.busy || !mergeReady;
  fixButton.disabled = state.busy || !mergeReady;
  mergeButton.disabled = state.busy || !structureValid;
}

function setBusy(busy) {
  document.documentElement.classList.toggle("busy", busy);
  all("button, select, input:not([readonly])").forEach((control) => {
    control.disabled = busy;
  });
  syncKnowledgePaginationDisabled(busy);
}

function applyState(state) {
  currentState = state;
  byId("versionLabel").textContent = `v${state.version}`;
  activatePage(state.active_page);
  applyTheme(state.theme, state.themes);
  renderFields(state.fields);
  renderLanguages(state);
  setBusy(state.busy);
  renderTranslationProviders(state);
  renderTranslationReadiness(state);
  const knowledge = state.translation_knowledge;
  if (hasPaginatedTranslationKnowledge(state.translation_knowledge)) {
    renderTranslationKnowledge(knowledge);
  }
  renderStatus(state.status);
  renderMonitor(state.monitor);
}

function applyResponse(response) {
  if (response.state) applyState(response.state);
  if (!response.ok && response.error) showToast(response.error.message);
}

async function pollState() {
  try {
    const state = await window.pywebview.api.get_state();
    applyState(state);
  } catch (error) {
    renderTranslationOffline(error);
    showToast(`Không thể đọc trạng thái: ${error.message || error}`);
  } finally {
    const delay = currentState && currentState.busy ? 250 : 900;
    pollTimer = window.setTimeout(pollState, delay);
  }
}

function themeOptions() {
  return all("[data-theme-choice]");
}

function focusThemeOption(index) {
  const options = themeOptions();
  if (!options.length) return;
  const normalized = (index + options.length) % options.length;
  options[normalized].focus();
}

function openThemeMenu(preferLast = false) {
  byId("themeSwitcher").classList.add("open");
  byId("themeTrigger").setAttribute("aria-expanded", "true");
  const options = themeOptions();
  const active = options.findIndex((option) => option.classList.contains("active"));
  const target = preferLast ? options.length - 1 : Math.max(active, 0);
  window.requestAnimationFrame(() => focusThemeOption(target));
}

function closeThemeMenu(returnFocus = false) {
  byId("themeSwitcher").classList.remove("open");
  byId("themeTrigger").setAttribute("aria-expanded", "false");
  if (returnFocus) byId("themeTrigger").focus();
}

function openKnowledgePathDialog(mode, trigger) {
  const knowledgePathDialog = byId("knowledgePathDialog");
  const knowledgePathInput = byId("knowledgePathInput");
  const isImport = mode === "import";
  knowledgePathMode = isImport ? "import" : "export";
  knowledgePathTrigger = trigger;
  byId("knowledgePathDialogTitle").textContent = isImport
    ? "Import Translation Knowledge"
    : "Export Translation Knowledge";
  byId("knowledgePathDialogHelp").textContent = isImport
    ? "Nhập đường dẫn tới file Translation Knowledge dạng JSON cần import."
    : "Nhập đường dẫn file JSON sẽ lưu Translation Knowledge.";
  byId("knowledgePathConfirm").textContent = isImport ? "Import" : "Export";
  byId("knowledgePathError").textContent = "";
  knowledgePathInput.value = isImport ? "" : "translation-knowledge.json";
  knowledgePathDialog.showModal();
  window.requestAnimationFrame(() => {
    knowledgePathInput.focus();
    knowledgePathInput.select();
  });
}

function closeKnowledgePathDialog(restoreFocus = true) {
  const knowledgePathDialog = byId("knowledgePathDialog");
  if (knowledgePathDialog.open) knowledgePathDialog.close();
  if (restoreFocus && knowledgePathTrigger) knowledgePathTrigger.focus();
  knowledgePathMode = null;
  knowledgePathTrigger = null;
}

async function confirmKnowledgePathDialog() {
  const path = byId("knowledgePathInput").value.trim();
  const error = byId("knowledgePathError");
  if (!path) {
    error.textContent = "Hãy nhập đường dẫn file JSON.";
    byId("knowledgePathInput").focus();
    return;
  }
  if (!path.toLowerCase().endsWith(".json")) {
    error.textContent = "Translation Knowledge phải dùng file .json.";
    byId("knowledgePathInput").focus();
    return;
  }

  const mode = knowledgePathMode;
  closeKnowledgePathDialog();
  if (mode === "import") {
    applyResponse(await window.pywebview.api.import_translation_knowledge(path));
    knowledgePage = 1;
    await loadTranslationKnowledge();
  } else if (mode === "export") {
    applyResponse(await window.pywebview.api.export_translation_knowledge(path));
  }
}

function bindInteractions() {
  byId("themeTrigger").addEventListener("click", () => {
    const switcher = byId("themeSwitcher");
    if (switcher.classList.contains("open")) closeThemeMenu();
    else openThemeMenu();
  });
  byId("themeTrigger").addEventListener("keydown", (event) => {
    if (["ArrowDown", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      openThemeMenu();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      openThemeMenu(true);
    }
  });
  byId("themeMenu").addEventListener("keydown", (event) => {
    const options = themeOptions();
    const current = Math.max(options.indexOf(document.activeElement), 0);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusThemeOption(current + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      focusThemeOption(current - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusThemeOption(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusThemeOption(options.length - 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeThemeMenu(true);
    }
  });

  document.addEventListener("click", (event) => {
    if (!byId("themeSwitcher").contains(event.target)) closeThemeMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeThemeMenu();
  });

  all(".nav-item[data-page]").forEach((button) => {
    button.addEventListener("click", async () => {
      activatePage(button.dataset.page);
      applyResponse(await window.pywebview.api.set_page(button.dataset.page));
    });
  });

  all("[data-theme-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      closeThemeMenu(true);
      applyResponse(await window.pywebview.api.set_theme(button.dataset.themeChoice));
    });
  });

  byId("languageSelect").addEventListener("change", async (event) => {
    applyResponse(await window.pywebview.api.set_language(event.target.value));
  });

  byId("translationProvider").addEventListener("change", async (event) => {
    customConfigDirty = false;
    applyResponse(
      await window.pywebview.api.set_translation_provider(event.target.value)
    );
  });

  CUSTOM_CONFIG_INPUT_IDS.forEach((id) => {
    byId(id).addEventListener("input", () => {
      customConfigDirty = true;
      byId("translationConfigStatus").textContent = "Có thay đổi chưa lưu vào file. Nếu bấm Start, job mới sẽ dùng chính các giá trị đang hiển thị này.";
    });
  });

  byId("translationSaveApiKey").addEventListener("change", (event) => {
    byId("translationConfigStatus").textContent = event.target.checked
      ? "Khi bấm Lưu config, API key hiện tại sẽ được ghi vào file JSON."
      : "Khi bấm Lưu config, API key sẽ không được ghi vào file JSON.";
  });

  byId("translationConfigSave").addEventListener("click", async () => {
    const persistApiKey = byId("translationSaveApiKey").checked;
    const response = await window.pywebview.api.set_translation_config(
      readCustomConfigDraft(),
      persistApiKey
    );
    if (response.ok) {
      customConfigDirty = false;
      byId("translationApiKey").value = "";
    }
    applyResponse(response);
    if (response.ok) {
      const configPath = response.state && response.state.translation_config
        ? response.state.translation_config.config_path
        : "";
      setCustomConfigFeedback(
        persistApiKey
          ? `Đã lưu config kèm API key${configPath ? `: ${configPath}.` : "."} Start vẫn dùng cấu hình đang hiển thị.`
          : `Đã lưu config không kèm API key${configPath ? `: ${configPath}.` : "."} Start vẫn dùng cấu hình đang hiển thị.`
      );
    }
  });

  byId("translationConfigOpen").addEventListener("click", async () => {
    const response = await window.pywebview.api.open_translation_config_file();
    applyResponse(response);
    if (response.ok) setCustomConfigFeedback("Đã mở thư mục chứa custom_openai_config.json.");
  });

  byId("translationConfigReload").addEventListener("click", async () => {
    const response = await window.pywebview.api.reload_translation_config_file();
    if (response.ok && !response.cancelled) {
      customConfigDirty = false;
      byId("translationApiKey").value = "";
    }
    applyResponse(response);
    if (response.ok && !response.cancelled) {
      setCustomConfigFeedback(`Đã nạp config: ${response.config_path || "file đã chọn"}.`);
    }
  });

  byId("autoTranslateAction").addEventListener("click", async () => {
    const jobState = currentState && currentState.translation_job
      ? currentState.translation_job.state
      : "idle";
    let response;
    if (jobState === "running") {
      response = await window.pywebview.api.pause_translation();
    } else if (["paused", "completed_with_pending"].includes(jobState)) {
      response = await window.pywebview.api.resume_translation();
    } else {
      const providerKey = byId("translationProvider").value;
      response = providerKey === "openai-compatible"
        ? await window.pywebview.api.start_translation(null, null, readCustomConfigDraft())
        : await window.pywebview.api.start_translation();
    }
    applyResponse(response);
    if (response.ok && jobState !== "running" && !["paused", "completed_with_pending"].includes(jobState) && byId("translationProvider").value === "openai-compatible") {
      byId("translationConfigStatus").textContent = "Job này đang dùng cấu hình vừa hiển thị trên màn hình. Lưu config chỉ dùng cho lần mở/nạp sau.";
    }
  });

  byId("knowledgeSearchAction").addEventListener("click", async () => {
    knowledgePage = 1;
    await loadTranslationKnowledge();
  });
  byId("knowledgeSearch").addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      knowledgePage = 1;
      await loadTranslationKnowledge();
    }
  });
  byId("knowledgePrev").addEventListener("click", async () => {
    knowledgePage = Math.max(1, knowledgePage - 1);
    await loadTranslationKnowledge();
  });
  byId("knowledgeNext").addEventListener("click", async () => {
    knowledgePage += 1;
    await loadTranslationKnowledge();
  });
  byId("knowledgeImport").addEventListener("click", (event) => {
    openKnowledgePathDialog("import", event.currentTarget);
  });
  byId("knowledgeExport").addEventListener("click", (event) => {
    openKnowledgePathDialog("export", event.currentTarget);
  });
  byId("knowledgePathCancel").addEventListener("click", () => {
    closeKnowledgePathDialog();
  });
  byId("knowledgePathConfirm").addEventListener("click", async () => {
    await confirmKnowledgePathDialog();
  });
  byId("knowledgePathInput").addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      await confirmKnowledgePathDialog();
    }
  });
  const knowledgePathDialog = byId("knowledgePathDialog");
  knowledgePathDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeKnowledgePathDialog();
  });

  all("[data-browse]").forEach((button) => {
    button.addEventListener("click", async () => {
      applyResponse(await window.pywebview.api.browse(button.dataset.browse));
    });
  });

  all("[data-open]").forEach((button) => {
    button.addEventListener("click", async () => {
      applyResponse(await window.pywebview.api.open_path(button.dataset.open));
    });
  });

  all("[data-operation]").forEach((button) => {
    button.addEventListener("click", async () => {
      applyResponse(await window.pywebview.api.start_operation(button.dataset.operation));
    });
  });
}

window.addEventListener("pywebviewready", async () => {
  if (window.pywebview.platform !== "edgechromium") {
    byId("compatibilityWarning").hidden = false;
  }
  bindInteractions();
  window.clearTimeout(pollTimer);
  await pollState();
  await loadTranslationKnowledge();
});
