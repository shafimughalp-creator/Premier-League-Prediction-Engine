/* Behaviours the prototype does in vanilla JS.
   The server renders the markup; this file adds the interactive parts:
   count-ups, scroll reveal, badge fallback, week filters, the team
   drawer, the mobile menu, Escape/Enter keyboard handling and titles. */
(function () {
  "use strict";

  var DATA = window.EPL_DATA || {};
  var $ = function (s) { return document.querySelector(s); };
  var SH = "M4 4H44V30C44 42 34 49 24 53C14 49 4 42 4 30Z";
  var PAT = {
    v: '<rect x="12" width="8" height="56"/><rect x="28" width="8" height="56"/>',
    h: '<rect y="16" width="48" height="12"/><rect y="40" width="48" height="12"/>',
    half: '<rect x="24" width="24" height="56"/>',
    diag: '<polygon points="0,14 0,28 48,50 48,36"/>',
    chev: '<path d="M0 22L24 38L48 22V32L24 48L0 32Z"/>',
    band: '<rect y="22" width="48" height="10"/>',
    top: '<rect width="48" height="15"/>',
    cross: '<rect x="20" width="8" height="56"/><rect y="24" width="48" height="8"/>'
  };
  var LBL = { home: "Home win", draw: "Draw", away: "Away win" };
  var MISS = "\u2014<i data-miss></i>";

  /* The prototype puts `<!-- TODO: real value -->` after each "—". Streamlit's
     sanitizer strips HTML comments, so render a placeholder and swap it for a
     real comment node — same text-run segmentation, same shaping. */
  function hydrateMiss() {
    var slots = document.querySelectorAll("[data-miss]");
    for (var i = 0; i < slots.length; i++) {
      slots[i].parentNode.replaceChild(
        document.createComment(" TODO: real value "), slots[i]
      );
    }
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function T(n) {
    return (DATA.table || []).find(function (t) { return t.team === n; });
  }
  function gd(n) {
    return n > 0 ? "+" + n : n < 0 ? "\u2212" + Math.abs(n) : "0";
  }
  function v(x) { return x == null ? MISS : esc(x); }
  function nm(n) { return esc((DATA.full || {})[n] || n); }

  function badge(sh, s) {
    var c = DATA.club[sh];
    return '<svg class="crest" width="' + s + '" height="' + Math.round(s * 1.17) +
      '" viewBox="0 0 48 56" aria-hidden="true"><g clip-path="url(#sh)">' +
      '<rect width="48" height="56" fill="' + c[0] + '"/><g fill="' + c[1] + '">' +
      PAT[c[2]] + '</g><rect width="48" height="28" fill="url(#shine)"/></g>' +
      '<path d="' + SH + '" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="2"/>' +
      '<text x="24" y="34" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" ' +
      'font-weight="800" fill="#fff" stroke="rgba(0,0,0,.65)" stroke-width="3" ' +
      'paint-order="stroke">' + sh + '</text></svg>';
  }
  function fb(el) {
    if (el && el.isConnected) el.outerHTML = badge(el.dataset.c, +el.dataset.s);
  }

  var MARK = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38003c" stroke-width="2">' +
    '<circle cx="12" cy="12" r="9"/><path d="M12 7l4 3-1.5 5h-5L8 10z"/></svg>';
  var DEFS = '<svg id="epl-defs" width="0" height="0" style="position:absolute" aria-hidden="true" ' +
    'xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="sh"><path d="' + SH + '"/></clipPath>' +
    '<linearGradient id="shine" x1="0" y1="0" x2="0" y2="1">' +
    '<stop offset="0" stop-color="#fff" stop-opacity=".3"/>' +
    '<stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs></svg>';

  /* Streamlit strips inline <svg> from st.html. Re-create the shared defs
     and the built-in badge/mark shapes in the live DOM. */
  function ensureDefs() {
    if (document.getElementById("epl-defs")) return;
    var wrap = document.createElement("div");
    wrap.innerHTML = DEFS;
    document.body.insertBefore(wrap.firstChild, document.body.firstChild);
  }
  function hydrateMark() {
    document.querySelectorAll("[data-mark]").forEach(function (el) {
      el.innerHTML = MARK;
    });
  }
  function hydrateSlots() {
    document.querySelectorAll(".crest-slot").forEach(function (el) {
      el.outerHTML = badge(el.dataset.c, +el.dataset.s);
    });
  }

  /* Streamlit writes its page title after our script; keep the route title. */
  function lockTitle() {
    if (!window.EPL_TITLE) return;
    document.title = window.EPL_TITLE;
    var t = document.querySelector("title");
    if (!t || t.dataset.eplLocked) return;
    t.dataset.eplLocked = "1";
    new MutationObserver(function () {
      if (document.title !== window.EPL_TITLE) document.title = window.EPL_TITLE;
    }).observe(t, { childList: true });
  }
  function crest(n, s) {
    s = s || 34;
    var t = T(n);
    var url = DATA.logos && DATA.logos[t.short];
    return url
      ? '<img class="crest" src="' + esc(url) + '" data-c="' + t.short + '" data-s="' + s +
        '" width="' + s + '" height="' + s + '" alt="" referrerpolicy="no-referrer">'
      : badge(t.short, s);
  }
  function chips(f) {
    return '<span class="fc" aria-label="Last five: ' + f.join(", ") + '">' +
      f.map(function (x) { return '<b class="' + x + '">' + x + '</b>'; }).join("") + "</span>";
  }
  function bar(c, big) {
    return '<div class="bar' + (big ? " big" : "") + '" role="img" aria-label="Home win ' +
      c.home + '%, draw ' + c.draw + '%, away win ' + c.away + '%">' +
      '<i class="h" style="flex:' + c.home + '"></i>' +
      '<i class="d" style="flex:' + c.draw + '"></i>' +
      '<i class="a" style="flex:' + c.away + '"></i></div>' +
      '<div class="pcs"><span data-count="' + c.home + '" data-suf="%">' + c.home + '%</span>' +
      '<span data-count="' + c.draw + '" data-suf="%">' + c.draw + '%</span>' +
      '<span data-count="' + c.away + '" data-suf="%">' + c.away + '%</span></div>';
  }
  function row(f) {
    return '<a class="fx" href="?page=match&amp;id=' + encodeURIComponent(f.id) +
      '" aria-label="' + esc(f.home) + " v " + esc(f.away) + '">' +
      '<span class="time">' + esc(f.time) + '</span>' +
      '<div class="tm r th"><span>' + esc(f.home) + "</span>" + crest(f.home) + "</div>" +
      '<div class="tb">' + bar(f.chances) + "</div>" +
      '<div class="tm ta">' + crest(f.away) + "<span>" + esc(f.away) + "</span></div>" +
      '<span class="tag ' + f.pick + '">' + LBL[f.pick] + "</span></a>";
  }
  function head(e, t, s) {
    return '<header class="head"><div class="eye">' + e + "</div><h1>" + t +
      '</h1><p class="sub-t">' + s + "</p></header>";
  }

  var filter = "all";

  function weekList() {
    var l = (DATA.fixtures || []).filter(function (f) {
      return filter === "all" || f.pick === filter;
    });
    return l.length
      ? l.map(row).join("")
      : '<p class="mut" style="padding:20px 0">No fixtures lean this way this week.</p>';
  }
  function week() {
    var c = function (k) {
      return k === "all" ? DATA.fixtures.length : DATA.fixtures.filter(function (f) {
        return f.pick === k;
      }).length;
    };
    var kickoff = DATA.fixtures && DATA.fixtures[0] ? DATA.fixtures[0].time : "";
    return head(
      "Matchweek " + DATA.meta.matchweek + " \u00b7 " + DATA.meta.season,
      "Matchweek " + DATA.meta.matchweek,
      DATA.meta.dateLong + " \u00b7 all kick-offs " + kickoff
    ) +
      '<div class="chips" role="group" aria-label="Filter fixtures">' +
      [["all", "All"], ["home", "Home wins"], ["draw", "Draws"], ["away", "Away wins"]]
        .map(function (pair) {
          var k = pair[0], l = pair[1];
          return '<button class="chip' + (filter === k ? " on" : "") + '" data-f="' + k +
            '" aria-pressed="' + (filter === k) + '">' + l + "<span>" + c(k) + "</span></button>";
        }).join("") +
      '</div><div id="wl">' + weekList() + "</div>" +
      '<p class="note">Chances are rounded to whole percentages and always add up to 100.</p>';
  }

  /* drawer */
  var lastFocus = null;
  function openTeam(n) {
    var t = T(n);
    lastFocus = document.activeElement;
    var d = $("#drawer");
    d.innerHTML = '<button class="x" data-close aria-label="Close">\u2715</button>' +
      '<div class="tm">' + crest(n, 56) + '<div><h2 style="font-size:24px;letter-spacing:-.02em">' +
      esc(n) + '</h2><span class="mut">' + t.position +
      (["st", "nd", "rd"][t.position - 1] || "th") + " \u00b7 " + t.points + " pts</span></div></div>" +
      '<div class="dg">' + [["P", t.played], ["W", t.won], ["D", t.drawn], ["L", t.lost],
        ["GF", t.gf], ["GA", t.ga], ["GD", gd(t.gd)], ["Pts", t.points]]
        .map(function (pair) {
          return "<div><b>" + pair[1] + "</b><span>" + pair[0] + "</span></div>";
        }).join("") + "</div>" +
      '<h3 class="t">Last five (oldest \u2192 newest)</h3>' + chips(t.form) +
      '<h3 class="t" style="margin-top:22px">Home and away</h3><div class="fm">' +
      '<span>Home points</span><span class="tr">' + MISS + "</span>" +
      '<span>Away points</span><span class="tr">' + MISS + "</span></div>";
    hydrateMiss();
    d.classList.add("open");
    $("#bd").classList.add("on");
    d.querySelector(".x").focus();
    bindCrests();
  }
  function closeTeam() {
    var d = $("#drawer");
    if (!d) return;
    d.classList.remove("open");
    var bd = $("#bd");
    if (bd) bd.classList.remove("on");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function countUp() {
    if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;
    document.querySelectorAll("[data-count]").forEach(function (el) {
      var to = +el.dataset.count;
      var dec = (el.dataset.count.split(".")[1] || "").length;
      var suf = el.dataset.suf || "";
      var t0 = performance.now();
      (function f(t) {
        var p = Math.min(1, (t - t0) / 900);
        var e = 1 - Math.pow(1 - p, 3);
        el.textContent = (to * e).toFixed(dec) + suf;
        if (p < 1) requestAnimationFrame(f);
      })(t0);
    });
  }

  var io;
  function reveal() {
    if (io) io.disconnect();
    if (!("IntersectionObserver" in window)) return;
    io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    document.querySelectorAll("#view .fx,#view .card,#view .cc,#view tbody tr").forEach(function (el) {
      if (el.getBoundingClientRect().top > innerHeight * 0.92) {
        el.classList.add("rv");
        io.observe(el);
      }
    });
  }

  function bindCrests() {
    document.querySelectorAll("img.crest").forEach(function (img) {
      if (img.dataset.bound) return;
      img.dataset.bound = "1";
      var fail = function () { fb(img); };
      img.addEventListener("error", fail);
      if (img.complete && img.naturalWidth === 0) fail();
    });
  }

  function setMenu(o) {
    var side = $("#side"), sbd = $("#sbd"), btn = $("#menuBtn");
    if (side) side.classList.toggle("open", o);
    if (sbd) sbd.classList.toggle("on", o);
    if (btn) btn.setAttribute("aria-expanded", o);
  }

  function init() {
    ensureDefs();
    hydrateMark();
    hydrateSlots();
    hydrateMiss();
    /* Streamlit's <main> has tabindex="0", which would become the first
       Tab stop ahead of the prototype's skip link and paint a full-page
       focus outline. Remove it from the tab order. */
    var stMain = document.querySelector('[data-testid="stMain"]');
    if (stMain && stMain.getAttribute("tabindex") === "0") {
      stMain.setAttribute("tabindex", "-1");
    }
    lockTitle();
    if (window.EPL_TITLE) {
      setTimeout(lockTitle, 0);
      setTimeout(lockTitle, 120);
      setTimeout(lockTitle, 500);
    }
    setMenu(false);
    window.scrollTo(0, 0);
    countUp();
    reveal();
    bindCrests();
  }

  document.addEventListener("click", function (e) {
    var t = e.target.closest("[data-team]");
    if (t) { openTeam(t.dataset.team); return; }
    if (e.target.closest("[data-close]") || e.target.id === "bd") { closeTeam(); return; }
    var f = e.target.closest("[data-f]");
    if (f) {
      filter = f.dataset.f;
      $("#view").innerHTML = week();
      hydrateMiss();
      countUp();
      reveal();
      bindCrests();
      return;
    }
    if (e.target.closest("[data-back]")) { location.href = "?page=week"; return; }
    if (e.target.id === "sbd") setMenu(false);
    if (e.target.closest("#menuBtn")) setMenu(!$("#side").classList.contains("open"));
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { closeTeam(); setMenu(false); }
    if (e.key === "Enter") {
      var t = e.target.closest && e.target.closest("tr[data-team],div[data-team]");
      if (t) openTeam(t.dataset.team);
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
