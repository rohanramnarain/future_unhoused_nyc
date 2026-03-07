(function () {
  const STORAGE_KEY = "fhfHideDashDebugToolbar";
  let restoreBtn = null;

  function isHiddenPreferred() {
    return sessionStorage.getItem(STORAGE_KEY) === "1";
  }

  function setHiddenPreferred(value) {
    sessionStorage.setItem(STORAGE_KEY, value ? "1" : "0");
  }

  function findToolbarRoot() {
    const candidates = document.querySelectorAll(
      ".dash-debug-menu__outer, .dash-debug-menu, [class*='dash-debug-menu__outer'], [class*=' dash-debug-menu']"
    );

    for (const el of candidates) {
      if (!(el instanceof HTMLElement)) {
        continue;
      }
      const rect = el.getBoundingClientRect();
      if (rect.width > 240 && rect.bottom >= window.innerHeight - 2) {
        return el;
      }
    }

    return null;
  }

  function ensureRestoreButton() {
    if (restoreBtn) {
      return restoreBtn;
    }

    restoreBtn = document.createElement("button");
    restoreBtn.type = "button";
    restoreBtn.className = "fhf-debug-restore-btn";
    restoreBtn.textContent = "Show Debug Bar";
    restoreBtn.style.display = "none";

    restoreBtn.addEventListener("click", function () {
      const toolbar = findToolbarRoot();
      if (!toolbar) {
        return;
      }
      toolbar.style.display = "";
      setHiddenPreferred(false);
      restoreBtn.style.display = "none";
    });

    document.body.appendChild(restoreBtn);
    return restoreBtn;
  }

  function attachHideButton(toolbar) {
    if (!toolbar || toolbar.querySelector(".fhf-debug-hide-btn")) {
      return;
    }

    const hideBtn = document.createElement("button");
    hideBtn.type = "button";
    hideBtn.className = "fhf-debug-hide-btn";
    hideBtn.textContent = "Hide";

    hideBtn.addEventListener("click", function () {
      toolbar.style.display = "none";
      setHiddenPreferred(true);
      ensureRestoreButton().style.display = "inline-flex";
    });

    toolbar.appendChild(hideBtn);
  }

  function applyState() {
    const toolbar = findToolbarRoot();
    if (!toolbar) {
      return;
    }

    attachHideButton(toolbar);

    const restore = ensureRestoreButton();
    if (isHiddenPreferred()) {
      toolbar.style.display = "none";
      restore.style.display = "inline-flex";
    } else {
      toolbar.style.display = "";
      restore.style.display = "none";
    }
  }

  const observer = new MutationObserver(applyState);

  function start() {
    applyState();
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
