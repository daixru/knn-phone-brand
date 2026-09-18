(function () {
  const form = document.getElementById("survey-form");
  if (!form) return;

  const formErrors = document.getElementById("form-errors");
  const btnSubmit = document.getElementById("btn-submit");
  const btnReset = document.getElementById("btn-reset");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultCard = document.getElementById("result-card");

  function collectAnswers() {
    const answers = {};
    form.querySelectorAll(".question-item").forEach((item) => {
      const key = item.dataset.key;
      const radios = item.querySelectorAll(`input[type="radio"][name="${key}"]`);
      const checkboxes = item.querySelectorAll(`input[type="checkbox"][name="${key}"]`);
      const single = item.querySelector(
        `input[name="${key}"]:not([type="radio"]):not([type="checkbox"])`
      );

      if (checkboxes.length) {
        answers[key] = Array.from(checkboxes)
          .filter((el) => el.checked)
          .map((el) => el.value);
      } else if (radios.length) {
        const checked = Array.from(radios).find((el) => el.checked);
        answers[key] = checked ? checked.value : "";
      } else if (single) {
        answers[key] = single.value;
      }
    });
    return answers;
  }

  function normalizeMessages(data) {
    if (Array.isArray(data?.messages)) return data.messages.map(String);
    if (data?.message) return [String(data.message)];
    return ["Ошибка сервера"];
  }

  function showErrors(messages) {
    if (!formErrors) return;
    const list = Array.isArray(messages) ? messages : [String(messages)];
    formErrors.classList.remove("hidden");
    formErrors.innerHTML =
      "<strong>Проверьте ответы.</strong><ul>" +
      list.map((m) => `<li>${escapeHtml(m)}</li>`).join("") +
      "</ul>";
    formErrors.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function clearErrors() {
    if (!formErrors) return;
    formErrors.classList.add("hidden");
    formErrors.innerHTML = "";
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function setLoading(loading) {
    if (!btnSubmit) return;
    btnSubmit.disabled = loading;
    const spinner = btnSubmit.querySelector(".btn-spinner");
    const label = btnSubmit.querySelector(".btn-label");
    if (spinner) spinner.classList.toggle("hidden", !loading);
    if (label) label.textContent = loading ? "Считаем…" : "Узнать результат";
  }

  function renderResult(data) {
    if (!resultPlaceholder || !resultCard) return;

    resultPlaceholder.classList.add("hidden");
    resultCard.classList.remove("hidden");

    const titleEl = document.getElementById("result-title");
    const confEl = document.getElementById("result-confidence");
    if (titleEl) titleEl.textContent = data.prediction_label || data.prediction || "—";
    const conf = Number(data.confidence);
    if (confEl) confEl.textContent = Number.isFinite(conf) ? `${Math.round(conf * 100)}%` : "—";

    const probBars = document.getElementById("prob-bars");
    if (probBars) {
      probBars.innerHTML = "";
      Object.entries(data.class_probabilities || {}).forEach(([label, p]) => {
        const pct = Math.round(Number(p) * 100);
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
        <span>${escapeHtml(label)}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
        <span>${pct}%</span>`;
        probBars.appendChild(row);
      });
    }

    const list = document.getElementById("neighbors-list");
    if (list) {
      list.innerHTML = "";
      (data.neighbors || []).forEach((n, i) => {
        const li = document.createElement("li");
        const labelRu =
          n.label === "iPhone" ? "iPhone" : n.label === "Android" ? "Android" : n.label;
        const dist = Number(n.distance);
        const distText = Number.isFinite(dist) ? dist.toFixed(2) : "—";
        li.innerHTML = `<span>#${i + 1} · ${escapeHtml(labelRu)}</span><span>${distText}</span>`;
        list.appendChild(li);
      });
    }

    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function submitSurvey() {
    clearErrors();
    const answers = collectAnswers();
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
      });
      let data = {};
      try {
        data = await res.json();
      } catch {
        showErrors(["Сервер вернул неожиданный ответ. Попробуйте ещё раз."]);
        return;
      }
      if (!res.ok) {
        showErrors(normalizeMessages(data));
        return;
      }
      renderResult(data);
    } catch {
      showErrors(["Нет связи с сервером. Проверьте подключение и попробуйте снова."]);
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitSurvey();
  });

  btnReset?.addEventListener("click", () => {
    form.reset();
    clearErrors();
    resultCard?.classList.add("hidden");
    resultPlaceholder?.classList.remove("hidden");
  });

  btnSubmit?.addEventListener("click", () => submitSurvey());
})();
