(function () {
  const HASHES = new Set(["#sources", "#method", "#limits", "#read-map", "#technical-details"]);

  function scrollAccordionIntoView() {
    const hash = window.location.hash;
    if (!HASHES.has(hash)) {
      return;
    }

    const accordion = document.getElementById("info-accordion");
    if (!accordion) {
      return;
    }

    // Wait a beat so Dash can open the corresponding accordion item first.
    window.setTimeout(function () {
      accordion.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 140);
  }

  window.addEventListener("hashchange", scrollAccordionIntoView);
  window.addEventListener("load", scrollAccordionIntoView);

  document.addEventListener("click", function (event) {
    const anchor = event.target.closest("a[href^='#']");
    if (!anchor) {
      return;
    }
    const href = anchor.getAttribute("href");
    if (!HASHES.has(href)) {
      return;
    }
    // Handles repeat clicks on the same hash where hashchange does not fire.
    window.setTimeout(scrollAccordionIntoView, 140);
  });
})();
