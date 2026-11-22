"use strict";
/* Store Sales model comparison: charts, cards, glossary and the EN / TH switch.
   Page text lives in index.html (English) and assets/i18n.js (Thai, keyed by data-i18n).
   Text built here (charts, cards, table, glossary) uses L(english, thai). */

const D = window.RESULTS, TH = window.TH || {};
const EN = {};                  // English text captured from the page on first load
let LANG = "en";                // default language is English
try {
  const q = new URLSearchParams(location.search).get("lang");
  const saved = localStorage.getItem("lang");
  LANG = (q === "th" || q === "en") ? q : (saved === "th" ? "th" : "en");
} catch (e) {}
const L = (en, th) => LANG === "th" ? th : en;

const f3 = v => v.toFixed(3), f5 = v => v.toFixed(5);

/* ---------- apply translations to the static page ---------- */
function applyI18n(){
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.dataset.i18n;
    if (!(k in EN)) EN[k] = el.innerHTML;
    el.innerHTML = (LANG === "th" && TH[k] != null) ? TH[k] : EN[k];
  });
  [["data-i18n-aria", "aria-label"], ["data-i18n-ph", "placeholder"]].forEach(([attr, target]) => {
    document.querySelectorAll("[" + attr + "]").forEach(el => {
      const k = el.getAttribute(attr), en = "__en_" + target;
      if (!(en in el)) el[en] = el.getAttribute(target);
      el.setAttribute(target, (LANG === "th" && TH[k] != null) ? TH[k] : el[en]);
    });
  });
  document.title = L("Store Sales Forecasting Explained", "อธิบายการพยากรณ์ยอดขาย Store Sales");
  const md = document.querySelector('meta[name="description"]');
  if (md) md.content = L("A plain-language walk-through of forecasting 16 days of grocery sales: the steps, why each choice was made, how LightGBM, XGBoost, CatBoost, Random Forest and LSTM compare, an FAQ, a glossary and the technical details.",
    "อธิบายการพยากรณ์ยอดขายของชำ 16 วันข้างหน้าด้วยภาษาง่าย ๆ: ขั้นตอน เหตุผลของแต่ละการตัดสินใจ การเปรียบเทียบ LightGBM, XGBoost, CatBoost, Random Forest และ LSTM พร้อมคำถามที่พบบ่อย อภิธานศัพท์ และรายละเอียดทางเทคนิค");
}

/* ---------- models ---------- */
const M = {
  lightgbm_tuned:{en:"LightGBM", th:"LightGBM", color:"--s1", dev:"CPU"},
  d3:{en:"LightGBM per-horizon", th:"LightGBM แยกตามวันที่พยากรณ์", color:"--s6", dev:"CPU"},
  xgboost_tuned:{en:"XGBoost", th:"XGBoost", color:"--s2", dev:"GPU"},
  catboost_tuned:{en:"CatBoost", th:"CatBoost", color:"--s3", dev:"GPU"},
  rf_tuned:{en:"Random Forest", th:"Random Forest", color:"--s4", dev:"CPU"},
  lstm:{en:"LSTM", th:"LSTM", color:"--s5", dev:"GPU"},
};
const mname = k => L(M[k].en, M[k].th);
const S = D.single, B = D.blend, H = D.horizon;
const bestBlend = B["lightgbm_tuned+d3"];

/* ---------- tooltip ---------- */
const tip = document.getElementById("tip");
function showTip(html, x, y){
  tip.innerHTML = html; tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(Math.max(8, x + 14), innerWidth - w - 8) + "px";
  tip.style.top = Math.min(Math.max(8, y + 14), innerHeight - h - 8) + "px";
}
const hideTip = () => tip.style.opacity = 0;
function attachTip(el, html){
  el.tabIndex = 0;
  el.addEventListener("mousemove", e => showTip(html, e.clientX, e.clientY));
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip(html, r.left + r.width/2, r.top); });
  el.addEventListener("blur", hideTip);
}

/* ---------- dot-plot helper ---------- */
function dotPlot(host, tickHost, rows, lo, hi, step, refVal){
  const pos = v => ((v - lo) / (hi - lo) * 100);
  tickHost.innerHTML = ""; host.innerHTML = "";
  const ticks = []; for (let t = lo; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(4));
  ticks.forEach(t => { const s = document.createElement("span"); s.style.left = pos(t) + "%"; s.textContent = t.toFixed(step < 0.01 ? 3 : 2); tickHost.appendChild(s); });
  rows.forEach(r => {
    const row = document.createElement("div"); row.className = "row"; row.setAttribute("role","listitem");
    const name = document.createElement("div"); name.className = "name"; name.innerHTML = r.label + (r.sub ? `<small>${r.sub}</small>` : "");
    const tr = document.createElement("div"); tr.className = "track";
    ticks.forEach(t => { const g = document.createElement("i"); g.className = "gl"; g.style.left = pos(t) + "%"; tr.appendChild(g); });
    const ln = document.createElement("i"); ln.className = "ln"; tr.appendChild(ln);
    if (refVal != null){ const rf = document.createElement("i"); rf.className = "ref"; rf.style.left = pos(refVal) + "%"; tr.appendChild(rf); }
    if (r.range){ const g = document.createElement("i"); g.className = "rng"; g.style.left = pos(r.range[0]) + "%"; g.style.width = (pos(r.range[1]) - pos(r.range[0])) + "%"; g.style.background = `var(${r.color})`; tr.appendChild(g); }
    if (r.mark != null){ const k = document.createElement("i"); k.className = "tick"; k.style.left = pos(r.mark) + "%"; tr.appendChild(k); }
    const d = document.createElement("i"); d.className = "dot"; d.style.left = pos(r.v) + "%"; d.style.background = `var(${r.color})`; tr.appendChild(d);
    const val = document.createElement("div"); val.className = "val"; val.textContent = r.text || f3(r.v);
    row.append(name, tr, val); host.appendChild(row); attachTip(row, r.tip);
  });
}

/* ---------- headline numbers ---------- */
function renderHeadline(){
  document.getElementById("heroNum").textContent = f3(bestBlend);
  const f1 = document.getElementById("f1"), f2 = document.getElementById("f2");
  if (f1) f1.textContent = f3(S.lightgbm);
  if (f2) f2.textContent = (S.lightgbm - bestBlend).toFixed(4);
}

/* ---------- results: ranking ---------- */
function renderRanking(){
  const vs = v => (v - S.lightgbm >= 0 ? "+" : "−") + Math.abs(v - S.lightgbm).toFixed(5);
  const ranking = [
    {label:L("Blend: LightGBM + per-horizon LightGBM","เฉลี่ย LightGBM + LightGBM แยกตามวัน"), sub:L("2-model average","เฉลี่ย 2 โมเดล"), v:bestBlend, color:"--accent",
     tip:`<b>${L("2-model average","เฉลี่ย 2 โมเดล")}</b><br>RMSLE ${f5(bestBlend)}<br>${L("Better than baseline LightGBM by","ดีกว่า LightGBM ตั้งต้นอยู่")} ${(S.lightgbm-bestBlend).toFixed(5)}`},
    {label:L("LightGBM (tuned)","LightGBM (ปรับแล้ว)"), sub:"CPU", v:S.lightgbm_tuned, color:"--s1"},
    {label:mname("d3"), sub:L("16 models · CPU","16 โมเดล · CPU"), v:S.d3, color:"--s6"},
    {label:L("XGBoost (tuned)","XGBoost (ปรับแล้ว)"), sub:"GPU", v:S.xgboost_tuned, color:"--s2"},
    {label:L("CatBoost (tuned)","CatBoost (ปรับแล้ว)"), sub:"GPU", v:S.catboost_tuned, color:"--s3"},
    {label:L("Random Forest (tuned)","Random Forest (ปรับแล้ว)"), sub:"CPU", v:S.rf_tuned, color:"--s4"},
    {label:"LSTM", sub:L("GPU · not tuned","GPU · ยังไม่ได้ปรับ"), v:S.lstm, color:"--s5"},
  ];
  ranking.forEach((r, i) => {
    r.text = f3(r.v);
    r.tip = r.tip || `<b>${r.label}</b><br>RMSLE ${f5(r.v)}<br>${L("Rank","อันดับ")} ${i+1} ${L("of","จาก")} ${ranking.length}<br>${L("vs baseline LightGBM:","เทียบ LightGBM ตั้งต้น:")} ${vs(r.v)}`;
  });
  dotPlot(document.getElementById("rank"), document.getElementById("rankTicks"), ranking, 0.37, 0.43, 0.01, S.lightgbm);
}

/* ---------- results: what helped ---------- */
function renderHelped(){
  const steps = [
    {t:L("Trim the leading zeros before a series really starts selling","ตัดศูนย์ก่อนอนุกรมเริ่มขายจริง"), d:S.s0 - S.s1,
     why:L("About half of all zero rows are from before the store or product opened, which distorts the lags (0.39001 → 0.38571)","ราวครึ่งหนึ่งของศูนย์ทั้งหมดมาจากช่วงก่อนร้าน/สินค้าเปิด ทำให้ lag เพี้ยน (0.39001 → 0.38571)"), ok:true},
    {t:L("Richer holiday features","เพิ่มฟีเจอร์วันหยุดแบบละเอียด"), d:S.s1 - S.s2,
     why:L("Tried, no help: slightly worse (0.38571 → 0.38595), so it was not kept","ลองแล้วไม่ช่วย แย่ลงเล็กน้อย (0.38571 → 0.38595) จึงไม่เก็บไว้"), ok:false},
    {t:L("Tune LightGBM hyper-parameters","ปรับแต่งไฮเปอร์พารามิเตอร์ของ LightGBM"), d:S.s1 - S.lightgbm_tuned,
     why:L("Very small gain (0.38571 → 0.38500): the defaults were already close to the best settings","ได้น้อยมาก (0.38571 → 0.38500) ค่าตั้งต้นใกล้ค่าที่ดีที่สุดอยู่แล้ว"), ok:true},
    {t:L("Average with the per-horizon models","เฉลี่ยกับโมเดลแยกตามวันที่พยากรณ์"), d:S.lightgbm_tuned - bestBlend,
     why:L("The two kinds of model make different mistakes, so their average beats either alone (0.38500 → 0.38206)","โมเดลสองแบบผิดพลาดคนละที่ เมื่อเฉลี่ยจึงดีกว่าแต่ละตัว (0.38500 → 0.38206)"), ok:true},
  ];
  const maxD = Math.max(...steps.map(s => Math.abs(s.d)));
  const el0 = document.getElementById("steps-chart"); el0.innerHTML = "";
  steps.forEach(s => {
    const el = document.createElement("div"); el.className = "step" + (s.ok ? "" : " no");
    const amt = (s.d >= 0 ? "−" : "+") + Math.abs(s.d).toFixed(5);
    el.innerHTML = `<div class="lbl"><b>${s.t}</b><div class="why">${s.why}</div></div>
      <div class="barrow"><div class="bar" style="width:${Math.max(1, Math.abs(s.d)/maxD*70)}%"></div><div class="amt">${amt}</div></div>`;
    el0.appendChild(el);
    attachTip(el, `<b>${s.t}</b><br>RMSLE ${amt.replace("−", L("reduced by ","ลดลง ")).replace("+", L("increased by ","เพิ่มขึ้น "))}<br>${s.ok ? L("Helped","ช่วย") : L("Did not help","ไม่ช่วย")}`);
  });
}

/* ---------- results: error by forecast day ---------- */
const hz = Array.from({length:16}, (_, i) => i + 1);
const order = ["lightgbm_tuned","d3","xgboost_tuned","catboost_tuned","rf_tuned","lstm"];
const on = new Set(order);
let cur = 8;
const lineEl = document.getElementById("line");

function renderChips(){
  const chips = document.getElementById("chips"); chips.innerHTML = "";
  order.forEach(k => {
    const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.setAttribute("aria-pressed", on.has(k));
    b.innerHTML = `<i style="background:var(${M[k].color})"></i>${mname(k)}`;
    b.onclick = () => { on.has(k) ? on.delete(k) : on.add(k); b.setAttribute("aria-pressed", on.has(k)); drawLine(); };
    chips.appendChild(b);
  });
}
function drawLine(){
  const W = Math.max(300, lineEl.clientWidth), Ht = W < 520 ? 260 : 320;
  const m = {l:48, r:16, t:12, b:34}, iw = W - m.l - m.r, ih = Ht - m.t - m.b;
  const keys = order.filter(k => on.has(k));
  const all = (keys.length ? keys : order).flatMap(k => H[k]);
  let lo = Math.floor(Math.min(...all) * 100) / 100, hi = Math.ceil(Math.max(...all) * 100) / 100;
  if (hi - lo < 0.04) hi = lo + 0.04;
  const x = h => m.l + (h - 1) / 15 * iw, y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih;
  let s = `<svg viewBox="0 0 ${W} ${Ht}" width="${W}" height="${Ht}" role="img">`;
  const yStep = (hi - lo) > 0.1 ? 0.02 : 0.01;
  for (let t = lo; t <= hi + 1e-9; t += yStep){
    s += `<line x1="${m.l}" x2="${W-m.r}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--grid)" stroke-width="1"/>`;
    s += `<text x="${m.l-8}" y="${y(t)+4}" text-anchor="end" font-size="11" style="fill:var(--ink2)">${t.toFixed(2)}</text>`;
  }
  s += `<line x1="${m.l}" x2="${W-m.r}" y1="${Ht-m.b}" y2="${Ht-m.b}" style="stroke:var(--axis)"/>`;
  const stepX = W < 520 ? 3 : 1;
  hz.forEach(h => { if (h === 1 || h % stepX === 0 || h === 16) s += `<text x="${x(h)}" y="${Ht-m.b+18}" text-anchor="middle" font-size="11" style="fill:var(--ink2)">${h}</text>`; });
  s += `<text x="${m.l+iw/2}" y="${Ht-2}" text-anchor="middle" font-size="11" style="fill:var(--ink2)">${L("Forecast day (after the last training date)","วันที่พยากรณ์ (หลังวันสุดท้ายของข้อมูลเทรน)")}</text>`;
  keys.forEach(k => {
    const p = H[k].map((v,i) => `${i?"L":"M"}${x(hz[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
    s += `<path d="${p}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(${M[k].color})"/>`;
    s += `<circle cx="${x(16)}" cy="${y(H[k][15])}" r="4.5" style="fill:var(${M[k].color});stroke:var(--surface)" stroke-width="2"/>`;
  });
  s += `<g id="cross" style="display:none"><line id="cl" y1="${m.t}" y2="${Ht-m.b}" style="stroke:var(--axis)" stroke-width="1"/></g>`;
  s += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" id="ov"/></svg>`;
  lineEl.innerHTML = s;
  const svg = lineEl.querySelector("svg"), cross = svg.querySelector("#cross"), cl = svg.querySelector("#cl");
  function at(h, cx, cy){
    cur = h; cross.style.display = ""; cl.setAttribute("x1", x(h)); cl.setAttribute("x2", x(h));
    cross.querySelectorAll("circle").forEach(c => c.remove());
    const list = keys.map(k => ({k, v:H[k][h-1]})).sort((a,b) => a.v - b.v);
    list.forEach(o => { const c = document.createElementNS("http://www.w3.org/2000/svg","circle"); c.setAttribute("cx", x(h)); c.setAttribute("cy", y(o.v)); c.setAttribute("r", 4.5); c.setAttribute("stroke-width", 2); c.style.fill = `var(${M[o.k].color})`; c.style.stroke = "var(--surface)"; cross.appendChild(c); });
    showTip(`<b>${L("Day","วันที่")} ${h}</b>` + list.map(o => `<br>${mname(o.k)}: <b>${o.v.toFixed(4)}</b>`).join(""), cx, cy);
  }
  const ov = svg.querySelector("#ov");
  const pick = e => { const r = svg.getBoundingClientRect(); const px = (e.clientX - r.left) * (W / r.width); return Math.min(16, Math.max(1, Math.round((px - m.l) / iw * 15) + 1)); };
  ov.addEventListener("mousemove", e => at(pick(e), e.clientX, e.clientY));
  ov.addEventListener("touchstart", e => { const t = e.touches[0]; at(pick(t), t.clientX, t.clientY); }, {passive:true});
  ov.addEventListener("touchmove", e => { const t = e.touches[0]; at(pick(t), t.clientX, t.clientY); }, {passive:true});
  ov.addEventListener("mouseleave", () => { cross.style.display = "none"; hideTip(); });
  lineEl.onkeydown = e => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft"){ e.preventDefault(); const h = Math.min(16, Math.max(1, cur + (e.key === "ArrowRight" ? 1 : -1))); const r = svg.getBoundingClientRect(); at(h, r.left + x(h) * r.width / W, r.top + 20); }
  };
  lineEl.onblur = () => { cross.style.display = "none"; hideTip(); };
}
function renderHzNote(){
  const avg = (k,a,b) => H[k].slice(a,b).reduce((s,v) => s+v, 0) / (b-a);
  const a = avg("lightgbm_tuned",0,4).toFixed(3), b = avg("lightgbm_tuned",12,16).toFixed(3);
  const d1 = H.d3[0].toFixed(3), l1 = H.lightgbm_tuned[0].toFixed(3), d16 = H.d3[15].toFixed(3), l16 = H.lightgbm_tuned[15].toFixed(3);
  document.getElementById("hzNote").textContent = L(
    `LightGBM: days 1–4 average ${a}, but days 13–16 average ${b} · The lines zigzag because each point is measured on a single day (1,782 rows); it is not a smoothed trend · The per-horizon model is better on day 1 (${d1} vs ${l1}) but worse on day 16 (${d16} vs ${l16})`,
    `LightGBM: วันที่ 1–4 เฉลี่ย ${a} แต่วันที่ 13–16 เฉลี่ย ${b} · เส้นขึ้นลงเพราะแต่ละจุดวัดจากวันเดียว (1,782 แถว) ไม่ใช่เส้นแนวโน้มที่เรียบ · โมเดลแยกตามวันดีกว่าในวันที่ 1 (${d1} เทียบ ${l1}) แต่แย่กว่าในวันที่ 16 (${d16} เทียบ ${l16})`);
}

/* ---------- results: tuning ---------- */
function renderTune(){
  const TN = {lightgbm:["LightGBM","--s1"], xgboost:["XGBoost","--s2"], catboost:["CatBoost","--s3"], rf:["Random Forest","--s4"]};
  const rows = Object.keys(TN).map(k => {
    const t = D.trials[k].map(o => o.rmsle), def = t[0], best = Math.min(...t), worst = Math.max(...t), n = t.length;
    return {label:TN[k][0], sub:L(`${n} settings tried · improved by ${(def-best).toFixed(4)}`, `ลอง ${n} ชุด · ดีขึ้น ${(def-best).toFixed(4)}`),
      v:best, mark:def, range:[best,worst], color:TN[k][1], text:f3(best),
      tip:L(`<b>${TN[k][0]}</b> (${n} settings tried)<br>Default ${f5(def)}<br>Best ${f5(best)} (improved by ${(def-best).toFixed(5)})<br>Worst ${f5(worst)}`,
            `<b>${TN[k][0]}</b> (ลอง ${n} ชุด)<br>ค่าตั้งต้น ${f5(def)}<br>ดีที่สุด ${f5(best)} (ดีขึ้น ${(def-best).toFixed(5)})<br>แย่ที่สุด ${f5(worst)}`)};
  });
  dotPlot(document.getElementById("tune"), document.getElementById("tuneTicks"), rows, 0.38, 0.42, 0.01, null);
}

/* ---------- results: blends ---------- */
function renderBlend(){
  const nm = {lightgbm_tuned:"LightGBM", d3:L("per-horizon","แยกตามวัน"), s1:"LGB s1", s2:"LGB s2", xgboost_tuned:"XGBoost", catboost_tuned:"CatBoost", rf_tuned:"RF", lstm:"LSTM"};
  const rows = Object.entries(B).sort((a,b) => a[1]-b[1]).map(([k,v]) => {
    const parts = k.split("+"), better = v < S.lightgbm_tuned, label = parts.map(p => nm[p] || p).join(" + ");
    return {label, sub: better ? L("Better than LightGBM alone","ดีกว่า LightGBM ตัวเดียว") : L("Worse than LightGBM alone","แย่กว่า LightGBM ตัวเดียว"),
      v, color: better ? "--accent" : "--muted", text:f3(v),
      tip:`<b>${label}</b><br>RMSLE ${f5(v)}<br>${better ? L("Better","ดีกว่า") : L("Worse","แย่กว่า")} ${L("than LightGBM alone by","LightGBM ตัวเดียวอยู่")} ${Math.abs(v-S.lightgbm_tuned).toFixed(5)}`};
  });
  dotPlot(document.getElementById("blend"), document.getElementById("blendTicks"), rows, 0.38, 0.39, 0.002, S.lightgbm_tuned);
}

/* ---------- the models (cards) ---------- */
const CARDS = [
 {k:"lightgbm_tuned", score:S.lightgbm_tuned,
  how:["Builds decision trees one at a time, each one trying to correct the errors left by the previous trees (gradient boosting), then adds them up into the prediction.",
       "สร้างต้นไม้ตัดสินใจทีละต้น แต่ละต้นพยายามแก้ข้อผิดพลาดที่ต้นก่อนหน้าทิ้งไว้ (gradient boosting) แล้วรวมกันเป็นคำทำนาย"],
  good:["The most accurate here, fast, uses lag features directly and handles missing values on its own","แม่นที่สุดในงานนี้ เร็ว ใช้ฟีเจอร์ lag ได้ตรง ๆ และจัดการค่าว่างได้เอง"],
  bad:["Cannot extrapolate beyond values it has seen (for example sales that keep growing), and the installed build does not use the GPU","ประมาณค่านอกช่วงที่เคยเห็นไม่ได้ (เช่นยอดขายที่โตขึ้นเรื่อย ๆ) และตัวที่ติดตั้งไม่ได้ใช้ GPU"],
  cfg:["num_leaves=127, min_data_in_leaf=50, feature_fraction=0.95, learning_rate=0.05, ~1,849 rounds","num_leaves=127, min_data_in_leaf=50, feature_fraction=0.95, learning_rate=0.05, ~1,849 รอบ"],
  time:["About 1–2 minutes per training run (CPU)","ราว 1–2 นาทีต่อรอบเทรน (CPU)"]},
 {k:"d3", score:S.d3,
  how:["Trains 16 LightGBM models, one per forecast day (1–16). Models for nearer days use fresher lags instead of lags of at least 16 days for every day.",
       "เทรน LightGBM 16 ตัว หนึ่งตัวต่อวันที่พยากรณ์ (วันที่ 1–16) ตัวที่พยากรณ์วันใกล้ใช้ lag ที่สดกว่า แทนที่จะใช้ lag ตั้งแต่ 16 วันขึ้นไปกับทุกวัน"],
  good:["Better on nearby days (day 1: 0.385 vs 0.392) and makes different mistakes from the single model, so averaging them helps a lot","ดีกว่าในวันใกล้ ๆ (วันที่ 1: 0.385 เทียบกับ 0.392) และผิดพลาดคนละแบบกับโมเดลเดี่ยว การเฉลี่ยจึงช่วยได้มาก"],
  bad:["Worse on far days (day 16), slow to train (16 models × 2 fits), and the 600 rounds it was given may not be enough","แย่กว่าในวันไกล (วันที่ 16) เทรนช้า (16 โมเดล × 2 รอบ) และ 600 รอบที่ตั้งไว้อาจยังไม่พอ"],
  cfg:["learning_rate=0.08, 600 rounds per model, s2 feature set (leading-zero trim + richer holidays)","learning_rate=0.08, 600 รอบต่อโมเดล, ฟีเจอร์ชุด s2 (ตัดศูนย์ + วันหยุดละเอียด)"],
  time:["About 11 minutes for the whole set (CPU)","ราว 11 นาทีทั้งชุด (CPU)"]},
 {k:"xgboost_tuned", score:S.xgboost_tuned,
  how:["Gradient boosting like LightGBM, but it grows trees level by level (level-wise) and controls complexity with regularization.",
       "gradient boosting เหมือน LightGBM แต่ขยายต้นไม้ทีละระดับ (level-wise) และคุมความซับซ้อนด้วย regularization"],
  good:["Very fast on the GPU (about 17 seconds) with results close to LightGBM","เร็วมากบน GPU (ราว 17 วินาที) และผลใกล้เคียง LightGBM"],
  bad:["Slightly worse than LightGBM and its predictions are very similar to it (correlation 0.999), so averaging the two does not help","แพ้ LightGBM เล็กน้อย และค่าทำนายคล้ายกันมาก (สหสัมพันธ์ 0.999) การเฉลี่ยทั้งสองจึงไม่ช่วย"],
  cfg:["max_depth=10, min_child_weight=100, reg_lambda=10, subsample=0.8, learning_rate=0.05","max_depth=10, min_child_weight=100, reg_lambda=10, subsample=0.8, learning_rate=0.05"],
  time:["17 seconds in the first run (GPU)","17 วินาทีในรอบแรก (GPU)"]},
 {k:"catboost_tuned", score:S.catboost_tuned,
  how:["Gradient boosting with symmetric (oblivious) trees that also handles categorical variables (product family, city, store type) natively.",
       "gradient boosting ที่ใช้ต้นไม้สมมาตร (oblivious trees) และจัดการตัวแปรหมวดหมู่ (กลุ่มสินค้า เมือง ประเภทร้าน) ได้ในตัว"],
  good:["Takes categorical variables directly with no manual encoding, and trains on the GPU","รับตัวแปรหมวดหมู่ได้ตรง ๆ ไม่ต้องเข้ารหัสเอง และเทรนบน GPU ได้"],
  bad:["Worse than LightGBM and XGBoost on this problem, and the GPU gives slightly different results from run to run","แย่กว่า LightGBM และ XGBoost ในงานนี้ และผลบน GPU ต่างกันเล็กน้อยทุกรอบ"],
  cfg:["depth=8, l2_leaf_reg=3, learning_rate=0.05, ~2,077 rounds","depth=8, l2_leaf_reg=3, learning_rate=0.05, ~2,077 รอบ"],
  time:["79 seconds in the first run at learning rate 0.1 (GPU); the final run takes longer","79 วินาทีในรอบแรกที่ learning rate 0.1 (GPU) รอบสุดท้ายนานกว่านี้"]},
 {k:"rf_tuned", score:S.rf_tuned,
  how:["Grows many independent trees on random samples of the data and averages their answers. Unlike boosting, the trees do not correct each other's errors.",
       "สร้างต้นไม้จำนวนมากที่เป็นอิสระต่อกันจากข้อมูลที่สุ่มมา แล้วเฉลี่ยคำตอบ ต่างจาก boosting ตรงที่ต้นไม้ไม่ช่วยแก้ข้อผิดพลาดของกันและกัน"],
  good:["Easy to configure and robust to outliers","ตั้งค่าง่ายและทนต่อค่าผิดปกติ (outlier)"],
  bad:["No round-by-round error correction, so clearly less accurate than boosting, and slow to train on the CPU","ไม่ได้แก้ข้อผิดพลาดทีละรอบ จึงแม่นน้อยกว่า boosting ชัดเจน และเทรนบน CPU ช้า"],
  cfg:["250 trees, min_samples_leaf=20, max_features=0.5, max_samples=0.7","250 ต้น, min_samples_leaf=20, max_features=0.5, max_samples=0.7"],
  time:["67 seconds per 100 trees in the first run (CPU); the final run uses 250 trees so it takes longer","67 วินาทีต่อ 100 ต้นในรอบแรก (CPU) รอบสุดท้ายใช้ 250 ต้นจึงนานกว่า"]},
 {k:"lstm", score:S.lstm,
  how:["A recurrent neural network with memory. It reads the last 56 days of sales of each series (ending 16 days before the target day), combines that with information known in advance (promotions, holidays, calendar) and predicts.",
       "โครงข่ายประสาทแบบมีความจำ อ่านยอดขายย้อนหลัง 56 วันของแต่ละอนุกรม (จบที่ 16 วันก่อนวันเป้าหมาย) แล้วรวมกับข้อมูลที่รู้ล่วงหน้า (โปรโมชัน วันหยุด ปฏิทิน) เพื่อทำนาย"],
  good:["Learns sequence patterns by itself, and in theory makes different mistakes from the trees","เรียนรู้รูปแบบตามลำดับเวลาได้เอง และในทางทฤษฎีผิดพลาดคนละแบบกับต้นไม้"],
  bad:["The weakest here: its predictions are still very close to the trees (correlation 0.997), so averaging does not help. It has not been tuned.","อ่อนที่สุดในที่นี้ ค่าทำนายยังใกล้เคียงต้นไม้มาก (สหสัมพันธ์ 0.997) การเฉลี่ยจึงไม่ช่วย และยังไม่ได้ปรับแต่ง"],
  cfg:["2-layer LSTM, size 128, 56-day window, 12 epochs, 2-seed average","LSTM 2 ชั้น ขนาด 128, หน้าต่าง 56 วัน, 12 epoch, เฉลี่ย 2 seed"],
  time:["About 21 seconds per epoch (GPU)","ราว 21 วินาทีต่อ epoch (GPU)"]},
];
function renderCards(){
  const host = document.getElementById("cards"); host.innerHTML = "";
  CARDS.forEach((c, i) => {
    const a = document.createElement("article"); a.className = "card";
    a.innerHTML = `<header><div><h3><span class="sw" style="background:var(${M[c.k].color})"></span>${mname(c.k)}</h3>
        <div class="tags" style="margin-top:6px"><span class="tag">${M[c.k].dev}</span><span class="tag">${L("Rank","อันดับ")} ${i+2} ${L("of","จาก")} 7</span></div></div>
        <div class="score"><b>${f3(c.score)}</b><span>RMSLE</span></div></header>
      <p class="small">${L(...c.how)}</p>
      <dl><dt>${L("Strengths","จุดแข็ง")}</dt><dd>${L(...c.good)}</dd><dt>${L("Weaknesses","จุดอ่อน")}</dt><dd>${L(...c.bad)}</dd>
      <dt>${L("Best settings","ค่าตั้งที่ดีที่สุด")}</dt><dd><code>${L(...c.cfg)}</code></dd><dt>${L("Training time","เวลาเทรน")}</dt><dd>${L(...c.time)}</dd></dl>`;
    host.appendChild(a);
  });
}

/* ---------- all numbers table ---------- */
function renderTable(){
  const rows = [
    [L("Blend: tuned LightGBM + per-horizon","เฉลี่ย LightGBM (ปรับแล้ว) + แยกตามวัน"), bestBlend, "CPU"],
    [L("LightGBM (tuned)","LightGBM (ปรับแล้ว)"), S.lightgbm_tuned, "CPU"],
    [L("LightGBM s1 (leading-zero trim)","LightGBM s1 (ตัดศูนย์ก่อนเริ่มขาย)"), S.s1, "CPU"],
    [L("LightGBM s2 (+ richer holidays)","LightGBM s2 (+ วันหยุดละเอียด)"), S.s2, "CPU"],
    [L("LightGBM per-horizon (d3)","LightGBM แยกตามวันที่พยากรณ์ (d3)"), S.d3, "CPU"],
    [L("LightGBM baseline","LightGBM ตั้งต้น"), S.lightgbm, "CPU"],
    [L("LightGBM s0 (lags on a complete day grid)","LightGBM s0 (lag บนตารางวันที่ครบ)"), S.s0, "CPU"],
    [L("XGBoost (tuned)","XGBoost (ปรับแล้ว)"), S.xgboost_tuned, "GPU"],
    [L("XGBoost (first run, old features)","XGBoost (รอบแรก ฟีเจอร์เก่า)"), S.xgboost, "GPU"],
    [L("CatBoost (tuned)","CatBoost (ปรับแล้ว)"), S.catboost_tuned, "GPU"],
    [L("CatBoost (first run, old features)","CatBoost (รอบแรก ฟีเจอร์เก่า)"), S.catboost, "GPU"],
    [L("Random Forest (tuned)","Random Forest (ปรับแล้ว)"), S.rf_tuned, "CPU"],
    [L("Random Forest (first run, old features)","Random Forest (รอบแรก ฟีเจอร์เก่า)"), S.rf, "CPU"],
    [L("LSTM (2 seeds)","LSTM (2 seed)"), S.lstm, "GPU"],
  ];
  document.getElementById("tbl").innerHTML = `<thead><tr><th>${L("Model","โมเดล")}</th><th style="text-align:right">RMSLE</th><th>${L("Hardware","ฮาร์ดแวร์")}</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r[0]}</td><td class="n">${f5(r[1])}</td><td>${r[2]}</td></tr>`).join("") + `</tbody>`;
}

/* ---------- the data: simple bar lists (bars always start at zero) ---------- */
const FACTS = {   // from EDA.ipynb: figures computed on the training data
  weekday: [[["Mon","จ."],618],[["Tue","อ."],570],[["Wed","พ."],593],[["Thu","พฤ."],505],[["Fri","ศ."],580],[["Sat","ส."],772],[["Sun","อา."],825]],   // thousand units per day, all stores and products
  year:    [[["2013","2013"],386],[["2014","2014"],575],[["2015","2015"],662],[["2016","2016"],791],[["2017","2017"],856]],                              // thousand units per day (2017 = Jan-Aug)
  promo:   [[["0 items","0 รายการ"],158],[["1–2","1–2"],529],[["3–5","3–5"],935],[["6–10","6–10"],1133],[["11–20","11–20"],1403],[["21–50","21–50"],2167],[["50+","50+"],3540]], // mean units per store-product-day
  zeros:   {lead: 49.7, other: 50.3},
};
function bars(host, items, fmt, what){
  const max = Math.max(...items.map(i => i[1]));
  host.innerHTML = "";
  items.forEach(([label, v]) => {
    const lb = L(label[0], label[1]);
    const row = document.createElement("div"); row.className = "brow";
    row.innerHTML = `<span class="bl">${lb}</span><span class="bt"><i style="width:${(v / max * 100).toFixed(1)}%"></i></span><span class="bv">${fmt(v)}</span>`;
    host.appendChild(row); attachTip(row, `<b>${lb}</b><br>${what}: ${fmt(v)}`);
  });
}
function renderData(){
  bars(document.getElementById("dWeekday"), FACTS.weekday, v => v + "k", L("average daily total","ยอดรวมเฉลี่ยต่อวัน"));
  bars(document.getElementById("dYear"), FACTS.year, v => v + "k", L("average daily total","ยอดรวมเฉลี่ยต่อวัน"));
  bars(document.getElementById("dPromo"), FACTS.promo, v => v.toLocaleString("en-US"), L("average units sold","ยอดขายเฉลี่ย (ชิ้น)"));
  const seg = document.getElementById("dZeros"), lg = document.getElementById("dZerosLegend");
  seg.setAttribute("aria-label", L(`${FACTS.zeros.lead} percent of zero records are before the series started selling; ${FACTS.zeros.other} percent are other zeros`,
    `ศูนย์ ${FACTS.zeros.lead}% อยู่ก่อนอนุกรมเริ่มขาย และ ${FACTS.zeros.other}% เป็นศูนย์อื่น ๆ`));
  seg.innerHTML = `<div style="width:${FACTS.zeros.lead}%;background:var(--s2)">${FACTS.zeros.lead}%</div><div style="width:${FACTS.zeros.other}%;background:var(--s1)">${FACTS.zeros.other}%</div>`;
  lg.innerHTML = `<li><i style="background:var(--s2)"></i><span>${L(
      `<b>${FACTS.zeros.lead}%: before the series first sold.</b> These mean "not open yet", not "no demand".`,
      `<b>${FACTS.zeros.lead}%: ก่อนอนุกรมเริ่มขายครั้งแรก</b> ศูนย์เหล่านี้หมายถึง "ยังไม่เปิด" ไม่ใช่ "ไม่มีความต้องการ"`)}</span></li>
    <li><i style="background:var(--s1)"></i><span>${L(
      `<b>${FACTS.zeros.other}%: all other zeros.</b> Real zero-sales days after the first sale, plus 53 series that never sold at all.`,
      `<b>${FACTS.zeros.other}%: ศูนย์อื่น ๆ ทั้งหมด</b> วันที่ขายได้ศูนย์จริงหลังเริ่มขายแล้ว รวมถึง 53 อนุกรมที่ไม่เคยขายได้เลย`)}</span></li>`;
}

/* ---------- glossary with a filter box ---------- */
const GLOSSARY = [
  ["Baseline","A simple first model used as the yardstick. Every later idea has to beat it. Here: the first LightGBM, scoring 0.388.","จุดเริ่มต้น (Baseline)","โมเดลง่าย ๆ ตัวแรกที่ใช้เป็นไม้บรรทัด ทุกไอเดียต่อจากนี้ต้องชนะมัน ในที่นี้คือ LightGBM ตัวแรก ได้ 0.388"],
  ["Blend (ensemble)","Averaging the predictions of several models. It helps when the models make different mistakes.","การเฉลี่ยโมเดล (Ensemble)","การเฉลี่ยค่าทำนายของหลายโมเดล ช่วยได้เมื่อโมเดลผิดพลาดคนละแบบ"],
  ["CPU / GPU","The processor types that run the training. A GPU is much faster for some algorithms (XGBoost, CatBoost, neural networks).","CPU / GPU","ชนิดของหน่วยประมวลผลที่ใช้เทรน GPU เร็วกว่ามากสำหรับบางอัลกอริทึม (XGBoost, CatBoost, โครงข่ายประสาท)"],
  ["Data leakage","Accidentally using information that would not be available at the moment of a real forecast, such as tomorrow's sales. It makes scores look better than they truly are.","การรั่วของข้อมูล (Data leakage)","การเผลอใช้ข้อมูลที่ไม่มีในตอนพยากรณ์จริง เช่นยอดขายของพรุ่งนี้ ทำให้คะแนนดูดีเกินจริง"],
  ["Decision tree","A flowchart of yes/no questions about the data (for example \"is it a weekend?\") that ends in a predicted number.","ต้นไม้ตัดสินใจ (Decision tree)","แผนผังคำถามใช่/ไม่ใช่เกี่ยวกับข้อมูล (เช่น \"เป็นวันสุดสัปดาห์ไหม\") ที่จบด้วยตัวเลขที่ทำนาย"],
  ["Early stopping","Stop adding trees once the score on held-out days stops improving, to avoid over-fitting.","การหยุดก่อนกำหนด (Early stopping)","หยุดเพิ่มต้นไม้เมื่อคะแนนบนวันที่กันไว้ไม่ดีขึ้นอีก เพื่อป้องกัน over-fitting"],
  ["Embedding","A short list of numbers a neural network learns to describe a category, such as a store or a product family.","เอ็มเบดดิง (Embedding)","ชุดตัวเลขสั้น ๆ ที่โครงข่ายประสาทเรียนรู้ขึ้นเพื่ออธิบายหมวดหมู่ เช่น ร้านหรือกลุ่มสินค้า"],
  ["Feature","One input column given to a model, for example the weekday, the oil price or the sales 28 days ago.","ฟีเจอร์ (Feature)","คอลัมน์ข้อมูลนำเข้าหนึ่งคอลัมน์ที่ให้โมเดล เช่น วันในสัปดาห์ ราคาน้ำมัน หรือยอดขาย 28 วันก่อน"],
  ["Forecast horizon","How far ahead we predict. Here 16 days; \"day 1\" is the day right after the data ends.","ระยะพยากรณ์ (Forecast horizon)","พยากรณ์ล่วงหน้าไกลแค่ไหน ที่นี่คือ 16 วัน \"วันที่ 1\" คือวันถัดจากวันสุดท้ายของข้อมูล"],
  ["Gradient boosting","Build many small decision trees one after another, each fixing the mistakes the previous ones left behind. LightGBM, XGBoost and CatBoost all work this way.","Gradient boosting","สร้างต้นไม้ตัดสินใจเล็ก ๆ จำนวนมากทีละต้น แต่ละต้นแก้ข้อผิดพลาดที่ต้นก่อนหน้าทิ้งไว้ LightGBM, XGBoost และ CatBoost ทำงานแบบนี้"],
  ["Hyper-parameter / tuning","A setting chosen before training (such as how deep a tree may be). Tuning means trying different settings and keeping the best.","ไฮเปอร์พารามิเตอร์ / การปรับแต่ง","ค่าตั้งที่เลือกก่อนเทรน (เช่น ต้นไม้ลึกได้แค่ไหน) การปรับแต่งคือการลองค่าตั้งหลายแบบแล้วเก็บแบบที่ดีที่สุด"],
  ["Lag","The value from some days earlier. \"Lag 28\" means the sales of the same series 28 days ago.","Lag (ค่าย้อนหลัง)","ค่าจากหลายวันก่อน \"Lag 28\" หมายถึงยอดขายของอนุกรมเดียวกันเมื่อ 28 วันก่อน"],
  ["Leading zeros","Zero-sales records from before a store or product started selling. They mean \"not open yet\", not \"no demand\".","ศูนย์นำหน้า (Leading zeros)","แถวยอดขายศูนย์จากก่อนร้านหรือสินค้าเริ่มขาย หมายถึง \"ยังไม่เปิด\" ไม่ใช่ \"ไม่มีความต้องการ\""],
  ["Learning rate","How big a step the model takes each training round. Smaller steps are slower but often more precise.","อัตราการเรียนรู้ (Learning rate)","โมเดลก้าวใหญ่แค่ไหนในแต่ละรอบเทรน ก้าวเล็กช้ากว่าแต่มักแม่นกว่า"],
  ["log1p / expm1","log(1 + x) and its reverse. It squeezes huge values together and handles zero. Sales are modelled on this scale, then converted back.","log1p / expm1","log(1 + x) และตัวผกผัน ช่วยบีบค่ามหาศาลให้ใกล้กันและรองรับศูนย์ เราสร้างโมเดลบนสเกลนี้แล้วแปลงกลับ"],
  ["LSTM","A neural network with a memory, designed to read sequences such as a series of daily sales.","LSTM","โครงข่ายประสาทที่มีความจำ ออกแบบมาอ่านข้อมูลเป็นลำดับ เช่นยอดขายรายวันต่อเนื่อง"],
  ["Over-fitting","Memorising the training data, including its noise, so the model does worse on new days.","Over-fitting (จำข้อมูลเทรนมากเกิน)","จำข้อมูลเทรนรวมถึงสัญญาณรบกวนจนทำงานกับวันใหม่ได้แย่ลง"],
  ["Random Forest","Many independent decision trees, each trained on a random sample of the data, whose answers are averaged.","Random Forest","ต้นไม้ตัดสินใจจำนวนมากที่เป็นอิสระต่อกัน แต่ละต้นเรียนจากข้อมูลที่สุ่มมา แล้วเฉลี่ยคำตอบ"],
  ["Rolling mean","The average over the most recent N days, for example the last 7 days.","ค่าเฉลี่ยเคลื่อนที่ (Rolling mean)","ค่าเฉลี่ยของ N วันล่าสุด เช่น 7 วันล่าสุด"],
  ["RMSLE","Root Mean Squared Logarithmic Error: the typical size of the miss measured on log(1 + sales). It cares about relative misses, so it treats small and large sellers fairly. Lower is better; 0 is perfect.","RMSLE","Root Mean Squared Logarithmic Error: ขนาดของความคลาดเคลื่อนโดยทั่วไปที่วัดบน log(1 + ยอดขาย) ใส่ใจความคลาดเคลื่อนเชิงสัดส่วน จึงปฏิบัติต่อสินค้าขายน้อยและขายมากอย่างเป็นธรรม ยิ่งต่ำยิ่งดี 0 คือสมบูรณ์แบบ"],
  ["Seed","The starting point of a random process. Averaging runs with different seeds makes results steadier.","Seed","จุดเริ่มต้นของกระบวนการสุ่ม การเฉลี่ยผลจากหลาย seed ทำให้ผลนิ่งขึ้น"],
  ["Series","One store × one product family. There are 1,782 of them.","อนุกรม (Series)","หนึ่งร้าน × หนึ่งกลุ่มสินค้า มีทั้งหมด 1,782 อนุกรม"],
  ["Validation window","Days held out of training and used as an exam to compare models. Here: the last 16 days of the training data.","ช่วงตรวจสอบ (Validation window)","วันที่กันออกจากการเทรนไว้ใช้เป็นข้อสอบเปรียบเทียบโมเดล ที่นี่คือ 16 วันสุดท้ายของข้อมูลเทรน"],
];
let glossQuery = "";
function renderGlossary(){
  const host = document.getElementById("gloss"), count = document.getElementById("glossCount");
  const slug = s => "g-" + s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const q = glossQuery.trim().toLowerCase();
  const list = GLOSSARY.filter(g => { const t = L(g[0], g[2]), d = L(g[1], g[3]); return !q || t.toLowerCase().includes(q) || d.toLowerCase().includes(q); });
  host.innerHTML = list.map(g => `<div class="g"><dt id="${slug(g[0])}">${L(g[0], g[2])}</dt><dd>${L(g[1], g[3])}</dd></div>`).join("") ||
    `<p class="muted small">${L("No term matches","ไม่พบคำที่ตรงกับ")} "${glossQuery.replace(/</g, "&lt;")}".</p>`;
  count.textContent = L(`${list.length} of ${GLOSSARY.length} terms`, `${list.length} จาก ${GLOSSARY.length} คำ`);
}

/* ---------- the 'why' table: label the cells so it can stack on phones ---------- */
function labelWhyTable(){
  document.querySelectorAll("table.why").forEach(t => {
    const heads = [...t.querySelectorAll("thead th")].map(th => th.textContent.trim());
    t.querySelectorAll("tbody tr").forEach(tr => [...tr.children].forEach((td, i) => td.dataset.label = heads[i] || ""));
  });
}

/* ---------- theme and language buttons ---------- */
const themeBtn = document.getElementById("themeBtn"), root = document.documentElement;
try { const sv = localStorage.getItem("theme"); if (sv === "light" || sv === "dark") root.dataset.theme = sv; } catch (e) {}
const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
function syncButtons(){
  themeBtn.textContent = isDark() ? L("Light mode","โหมดสว่าง") : L("Dark mode","โหมดมืด");
  themeBtn.setAttribute("aria-pressed", isDark());
  document.getElementById("langEn").setAttribute("aria-pressed", LANG === "en");
  document.getElementById("langTh").setAttribute("aria-pressed", LANG === "th");
}
themeBtn.onclick = () => { const n = isDark() ? "light" : "dark"; root.dataset.theme = n; try { localStorage.setItem("theme", n); } catch (e) {} syncButtons(); };
function setLang(l){ LANG = l; try { localStorage.setItem("lang", l); } catch (e) {} render(); }
document.getElementById("langEn").onclick = () => setLang("en");
document.getElementById("langTh").onclick = () => setLang("th");

/* ---------- render everything that depends on the language ---------- */
function render(){
  hideTip();
  applyI18n();
  renderHeadline(); renderRanking(); renderHelped(); renderChips(); drawLine(); renderHzNote();
  renderTune(); renderBlend(); renderCards(); renderTable(); renderData(); renderGlossary();
  labelWhyTable(); syncButtons();
}
render();

/* ---------- one-time behaviour ---------- */
new ResizeObserver(() => drawLine()).observe(lineEl);
document.getElementById("glossSearch").addEventListener("input", e => { glossQuery = e.target.value; renderGlossary(); });
(function tech(){
  const all = () => document.querySelectorAll(".tech details");
  document.getElementById("expandAll").onclick = () => all().forEach(d => d.open = true);
  document.getElementById("collapseAll").onclick = () => all().forEach(d => d.open = false);
  const open = () => { const t = location.hash && document.querySelector(".tech " + location.hash); if (t && t.tagName === "DETAILS") t.open = true; };
  addEventListener("hashchange", open); open();
})();
(function spy(){
  const links = [...document.querySelectorAll("#navLinks a")];
  const map = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
  const set = id => { links.forEach(a => a.removeAttribute("aria-current")); const a = map.get(id); if (a){ a.setAttribute("aria-current", "true"); a.scrollIntoView({inline: "center", block: "nearest"}); } };
  if (!("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) set(e.target.id); }), {rootMargin: "-20% 0px -70% 0px"});
  map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
})();
