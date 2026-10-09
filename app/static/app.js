// Браузер собирает ответы и показывает результат. Сам KNN работает на сервере.
// Обёртка оставляет переменные внутри этого файла, не добавляя их в window.
(function () {
  const form = document.getElementById("survey-form");
  if (!form) return;

  const formErrors = document.getElementById("form-errors");
  const btnSubmit = document.getElementById("btn-submit");
  const btnReset = document.getElementById("btn-reset");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultCard = document.getElementById("result-card");

  function collectAnswers() {
    // Все поля имеют name=q1…q21, поэтому можно читать стандартный FormData.
    // У флажков бывает несколько значений, у остальных полей — только одно.
    const formData = new FormData(form);
    const answers = {};
    form.querySelectorAll(".question-item").forEach((item) => {
      const key = item.dataset.key;
      answers[key] = item.dataset.fieldType === "multi"
        ? formData.getAll(key)
        : formData.get(key) ?? "";
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

  // Текст ответа вставляем в HTML как текст, а не как исполняемую разметку.
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

  // Класс, доли голосов и расстояния приходят готовыми из /api/predict.
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
        const dist = Number(n.distance);
        const distText = Number.isFinite(dist) ? dist.toFixed(2) : "—";
        li.innerHTML = `<span>#${i + 1} · ${escapeHtml(n.label)}</span><span>${distText}</span>`;
        list.appendChild(li);
      });
    }

    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // На время запроса блокируем кнопку. finally вернёт её в обычное состояние
  // при любом исходе: успешном ответе, ошибке сервера или отсутствии связи.
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

})();
