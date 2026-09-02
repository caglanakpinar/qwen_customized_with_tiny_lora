(() => {
  const chatColumn = document.getElementById("chat-column");
  const chatScroll = document.getElementById("chat-scroll");
  const form = document.getElementById("composer-form");
  const input = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const newChatBtn = document.getElementById("new-chat");

  const SESSION_KEY = "tinylora-chat-session-id";
  let sessionId = localStorage.getItem(SESSION_KEY);
  let isSending = false;

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Minimal markdown: fenced code blocks, inline code, and **bold** on top of escaped text.
  function renderMarkdown(rawText) {
    const escaped = escapeHtml(rawText);
    const withCodeBlocks = escaped.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const cleanCode = code.replace(/\n$/, "");
      const label = lang || "text";
      return (
        `<div class="code-block">` +
        `<div class="code-block-header">` +
        `<span class="code-lang">${label}</span>` +
        `<button type="button" class="copy-btn">Copy</button>` +
        `</div>` +
        `<pre><code class="language-${label}">${cleanCode}</code></pre>` +
        `</div>`
      );
    });
    const withInlineCode = withCodeBlocks.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    return withInlineCode.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }

  function wireCodeBlockCopyButtons(container) {
    container.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const code = btn.closest(".code-block").querySelector("code").textContent;
        navigator.clipboard.writeText(code).then(() => {
          const original = btn.textContent;
          btn.textContent = "Copied";
          btn.disabled = true;
          setTimeout(() => {
            btn.textContent = original;
            btn.disabled = false;
          }, 1500);
        });
      });
    });
  }

  function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function addMessage(role, text) {
    const emptyState = document.getElementById("empty-state");
    if (emptyState) {
      emptyState.remove();
    }
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "Assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = renderMarkdown(text);
    wireCodeBlockCopyButtons(bubble);

    wrapper.appendChild(label);
    wrapper.appendChild(bubble);
    chatColumn.appendChild(wrapper);
    scrollToBottom();
    return bubble;
  }

  function addTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "message assistant";
    wrapper.id = "typing-indicator";

    const bubble = document.createElement("div");
    bubble.className = "bubble typing";
    bubble.innerHTML = "<span></span><span></span><span></span>";

    wrapper.appendChild(bubble);
    chatColumn.appendChild(wrapper);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const node = document.getElementById("typing-indicator");
    if (node) {
      node.remove();
    }
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 192)}px`;
  }

  function updateSendState() {
    sendBtn.disabled = isSending || input.value.trim().length === 0;
  }

  async function sendMessage(text) {
    isSending = true;
    updateSendState();
    addTypingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);

      removeTypingIndicator();
      addMessage("assistant", data.reply);
    } catch (err) {
      removeTypingIndicator();
      addMessage("assistant", `Something went wrong: ${err.message}`);
    } finally {
      isSending = false;
      updateSendState();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || isSending) {
      return;
    }
    addMessage("user", text);
    input.value = "";
    autoResize();
    updateSendState();
    sendMessage(text);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    autoResize();
    updateSendState();
  });

  newChatBtn.addEventListener("click", async () => {
    if (sessionId) {
      fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    }
    localStorage.removeItem(SESSION_KEY);
    sessionId = null;
    chatColumn.innerHTML = `
      <div class="empty-state" id="empty-state">
        <h1>What are you working on?</h1>
        <p>This chat runs entirely on the TinyLoRA adapter loaded by the server. Ask it anything.</p>
      </div>
    `;
  });

  updateSendState();
})();
