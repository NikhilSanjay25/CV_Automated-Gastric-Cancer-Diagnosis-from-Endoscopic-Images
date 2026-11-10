// Builds presentation.pptx from the experiment outputs (results/*.json) and paper figures.
// usage: node deck/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const FIG = path.join(ROOT, "paper", "figures");
const J = (f) => { const p = path.join(RES, f); return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p)) : null; };
const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : (v * 100).toFixed(d));
const fig = (n) => path.join(FIG, n + ".png");
const has = (n) => fs.existsSync(fig(n));

const THEME = {
  name: "Endoscopy",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "12343B", lt2: "EEF4F4",
    accent1: "C8553D", accent2: "2A7F8E", accent3: "8AA6A3", accent4: "E9B872",
    accent5: "5C6F7B", accent6: "7A2E22", hlink: "2A7F8E", folHlink: "5C6F7B",
  },
};
const H = THEME.colors; // hex (for chart options that only accept hex)

const NAMES = { classical: "SIFT-BoVW+SVM", alexnet: "AlexNet", vgg16: "VGG-16", googlenet: "GoogLeNet",
  resnet50: "ResNet-50", densenet121: "DenseNet-121", mobilenetv2: "MobileNetV2",
  efficientnetb0: "EfficientNet-B0", vitb16: "ViT-B/16", proposed: "Proposed" };
const ORDER = ["classical", "alexnet", "vgg16", "googlenet", "resnet50", "densenet121", "mobilenetv2", "efficientnetb0", "vitb16"];
const DSN = { kvasir: "Kvasir v2", hyperkvasir: "HyperKvasir", gastrovision: "GastroVision" };

function table(ds) {
  const rows = {};
  for (const m of ORDER) { const r = J(`${ds}_${m}.json`); if (r) rows[m] = r.test; }
  const p = J(`${ds}_proposed.json`);
  if (p) rows.proposed = p.conditions.clean["Proposed cascade"];
  return rows;
}
function cascadeCost(ds, prevalence) {
  const eff = J("efficiency.json"), p = J(`${ds}_proposed.json`);
  if (!eff || !eff.stage23 || !p) return null;
  const g = p.conditions.clean._gate;
  const pass = prevalence == null ? 1 - g.bypass : prevalence * (1 - g.gate_spec_normal) + (1 - prevalence) * g.gate_recall_abn;
  return { cpu_fps: 1e3 / (eff.mobilenetv2.cpu_ms + pass * eff.stage23.cpu_ms),
           gpu_fps: 1e3 / (eff.mobilenetv2.latency_ms + pass * eff.stage23.gpu_ms),
           gmacs: eff.mobilenetv2.gmacs + pass * eff.stage23.gmacs, pass };
}

// in-text numbers generated for the paper (src/make_tables.py)
const MAC = (() => { const p = path.join(ROOT, "paper", "tables", "macros.tex"); const o = {};
  if (fs.existsSync(p)) for (const m of fs.readFileSync(p, "utf8").matchAll(/\\newcommand\{\\(R[a-z]+)\}\{(.*)\}/g)) o[m[1]] = m[2];
  return o; })();
const kv = table("kvasir"), kvp = J("kvasir_proposed.json"), rb = J("kvasir_robustness.json");
const eff = J("efficiency.json"), cross = J("crossdata.json"), gc = J("gradcam_localisation.json");
const bestBase = Object.keys(kv).filter((m) => m !== "proposed").sort((a, b) => kv[b].acc - kv[a].acc)[0];

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Automated Gastric Cancer Diagnosis from Endoscopic Images";
  pres.author = "Nikhil Sanjay, Andrew Tom Mathew";
  const C = pres.SchemeColor;

  // ------------------------------------------------------------------ layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 40, bold: true,
        color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 2.2, fontSize: 18,
        color: C.background2, valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true,
        color: C.text2, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
      { text: { text: "Automated Gastric Cancer Diagnosis from Endoscopic Images  ·  Amrita School of Computing, Bengaluru",
        options: { x: 0.6, y: 7.0, w: 10, h: 0.3, fontSize: 10, color: C.accent5 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.5, w: 11.7, h: 1.0, fontSize: 34, bold: true,
        color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
    ],
  });

  const content = (title, section) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const card = (s, x, y, w, h, fill, name) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
  const txt = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0 }, o));
  const stat = (s, x, y, w, big, small, color, name) => {
    txt(s, big, { x, y, w, h: 0.9, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace, objectName: name + "-value" });
    txt(s, small, { x, y: y + 0.9, w, h: 0.7, fontSize: 14, color: C.text1, valign: "top", objectName: name + "-label" });
  };
    // keep each figure's true aspect ratio: fit inside the (w,h) box, anchored top-left
  const aspect = (n) => { const b = fs.readFileSync(fig(n)); return b.readUInt32BE(16) / b.readUInt32BE(20); };
  const picture = (s, n, o) => {
    if (!has(n)) return;
    const a = aspect(n); let { w, h } = o;
    if (w / h > a) w = h * a; else h = w / a;
    const x = o.center ? o.x + (o.w - w) / 2 : o.x;
    s.addImage({ path: fig(n), altText: n, x, y: o.y, w, h });
  };
  const chartBase = {
    catAxisLabelColor: H.accent5, valAxisLabelColor: H.accent5, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: H.dk1,
    valGridLine: { color: "D9E2E2", size: 0.5 }, catGridLine: { style: "none" }, legendFontFace: "+mn-lt", legendFontSize: 12,
    titleFontFace: "+mn-lt",
  };

  // ------------------------------------------------------------------ 1 title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Automated Gastric Cancer Diagnosis from Endoscopic Images", { placeholder: "title" });
  s.addText([
    { text: "A cascaded spatio-temporal framework benchmarked on four public databases", options: { breakLine: true, italic: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Nikhil Sanjay (BL.EN.U4CSE23239)  ·  Andrew Tom Mathew (BL.EN.U4CSE23269)", options: { breakLine: true } },
    { text: "Guide: Dr. Tripty Singh  ·  Department of CSE, Amrita School of Computing, Bengaluru" },
  ], { placeholder: "body" });
  s.addNotes("Introduce the team and the one-line idea: a fast gate filters normal frames, a ResNet-50 + Bi-LSTM verifies suspicious frames over time, so the system stays accurate when individual frames are blurred or flashed.");

  // ------------------------------------------------------------------ 2 problem
  s = content("Why automate endoscopic diagnosis?", "Introduction");
  const gaps = [
    ["Real-time throughput", "Heavy CNN ensembles and Transformers are accurate on still images but expensive per video frame."],
    ["Class imbalance", "Normal tissue dominates; cross-entropy is biased toward it, so rare early lesions are missed."],
    ["Black-box decisions", "Grad-CAM maps are rarely checked against expert lesion masks."],
    ["Transient artifacts", "Motion blur, light flashes and bubbles make single-frame classifiers fail in live video."],
  ];
  gaps.forEach(([h, b], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.95, 2.3, C.background2, `gap-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `gap-dot-${i}` });
    txt(s, String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `gap-num-${i}` });
    txt(s, h, { x: x + 1.15, y: y + 0.3, w: 4.5, h: 0.6, fontSize: 20, bold: true, color: C.text2, valign: "middle", objectName: `gap-head-${i}` });
    txt(s, b, { x: x + 1.15, y: y + 1.0, w: 4.5, h: 1.1, fontSize: 16, color: C.text1, valign: "top", objectName: `gap-body-${i}` });
  });
  s.addNotes("Clinical motivation: early gastric lesions are subtle and detection depends on endoscopist skill and fatigue. False negatives cost lives; false positives cause unnecessary biopsies. Our literature review found four gaps; the first three come from published work, the fourth is the one our temporal design targets.");

  // ------------------------------------------------------------------ 3 architecture
  pres.addSection({ title: "Method" });
  s = content("Proposed cascaded spatio-temporal pipeline", "Method");
  const boxes = [
    ["Video frame t", "buffer of last k = 5 frames", C.background2, C.text2],
    ["Stage 1 · MobileNetV2 gate", "abnormality score s, averaged over the buffer", C.accent2, C.background1],
    ["Stage 2 · ResNet-50", "2048-d spatial feature per frame", C.text2, C.background1],
    ["Stage 3 · Bi-LSTM", "temporal attention verifies lesion stability", C.text2, C.background1],
    ["Diagnosis + Grad-CAM", "class label and heat-map", C.accent1, C.background1],
  ];
  const bw = 2.2, gap = 0.27, by = 2.0;
  boxes.forEach(([h, b, fill, ink], i) => {
    const x = 0.6 + i * (bw + gap);
    card(s, x, by, bw, 1.9, fill, `arch-box-${i}`);
    txt(s, h, { x: x + 0.15, y: by + 0.2, w: bw - 0.3, h: 0.75, fontSize: 16, bold: true, color: ink, align: "center", valign: "middle", objectName: `arch-head-${i}` });
    txt(s, b, { x: x + 0.15, y: by + 0.95, w: bw - 0.3, h: 0.8, fontSize: 12, color: ink, align: "center", valign: "top", objectName: `arch-body-${i}` });
    if (i < boxes.length - 1)
      s.addShape(pres.shapes.LINE, { x: x + bw + 0.02, y: by + 0.95, w: gap - 0.04, h: 0, line: { color: C.accent5, width: 2, endArrowType: "triangle" }, objectName: `arch-arrow-${i}` });
  });
  const gx = 0.6 + (bw + gap);
  s.addShape(pres.shapes.LINE, { x: gx + bw / 2, y: by + 1.9, w: 0, h: 0.7, line: { color: C.accent5, width: 2, endArrowType: "triangle" }, objectName: "arch-bypass-arrow" });
  card(s, gx - 0.2, by + 2.65, bw + 0.4, 0.9, C.background2, "arch-bypass-box");
  txt(s, "s < τ : bypass with a normal-landmark label", { x: gx - 0.1, y: by + 2.65, w: bw + 0.2, h: 0.9, fontSize: 12, color: C.text2, align: "center", valign: "middle", objectName: "arch-bypass-text" });
  const tau = kvp ? kvp.tau.toFixed(2) : "–";
  txt(s, [
    { text: "Training: ", options: { bold: true } },
    { text: `focal loss (γ = 2), AdamW, differential learning rate, cosine schedule. τ = ${tau} is chosen on validation as the largest threshold that keeps ≥ 99.5% of abnormal frames.` },
  ], { x: 6.0, y: 4.7, w: 6.7, h: 1.4, fontSize: 15, color: C.text1, valign: "top", objectName: "arch-training" });
  s.addNotes("Stage 1 is a cheap MobileNetV2 that scores how abnormal a frame looks. The score is averaged over the 5-frame buffer, so a single flashed frame cannot cause a bypass. Suspicious frames go to ResNet-50, whose features fill a buffer that a Bi-LSTM with attention reads. Grad-CAM explains the decision.");

  // ------------------------------------------------------------------ 4 synthetic clips
  s = content("From still images to video: clips with simulated artifacts", "Method");
  picture(s, "fig_artifacts", { x: 0.6, y: 1.45, w: 7.6, h: 5.3 });
  const artText = [
    ["Simulated camera motion", "Each image becomes a 5-frame clip with drifting rotation, zoom and shift."],
    ["Four artifact families", "Motion blur, defocus, specular flash, noise / occlusion."],
    ["Three test protocols", "Clean · current frame corrupted (transient) · random stream with 30% corrupted frames."],
  ];
  artText.forEach(([h, b], i) => {
    txt(s, h, { x: 8.6, y: 1.55 + i * 1.6, w: 4.1, h: 0.45, fontSize: 18, bold: true, color: C.text2, objectName: `clip-head-${i}` });
    txt(s, b, { x: 8.6, y: 2.0 + i * 1.6, w: 4.1, h: 1.0, fontSize: 15, color: C.text1, valign: "top", objectName: `clip-body-${i}` });
  });
  s.addNotes("Public databases contain still images, so we synthesise short clips. The frames before the current one follow a smooth simulated camera trajectory, and any frame can be hit by one of four artifact types. This lets us test exactly the failure mode the temporal stage is designed for.");

  // ------------------------------------------------------------------ 5 databases
  pres.addSection({ title: "Experimental setup" });
  s = content("Four public databases", "Experimental setup");
  const meta = (d) => { const p = path.join(ROOT, "data", "cache", `${d}_meta.json`); return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p)) : null; };
  const dbs = [["kvasir", "Primary 8-class benchmark"], ["gastrovision", "Upper + lower GI, gastric findings"],
    ["kvasirseg", "External polyp test + masks"], ["cvc", "Different hospital: domain shift"]];
  dbs.forEach(([d, role], i) => {
    const m = meta(d); const x = 0.6 + i * 3.08;
    const n = m ? (m.labels ? m.labels.length : m.ids.length) : null;
    const k = m && m.classes ? `${m.classes.length} classes` : "polyp masks";
    card(s, x, 1.5, 2.88, 2.05, C.background2, `db-card-${i}`);
    txt(s, n ? n.toLocaleString("en-US") : "–", { x: x + 0.2, y: 1.6, w: 2.5, h: 0.75, fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, objectName: `db-n-${i}` });
    txt(s, { kvasir: "Kvasir v2", hyperkvasir: "HyperKvasir", gastrovision: "GastroVision", kvasirseg: "Kvasir-SEG", cvc: "CVC-ClinicDB" }[d],
      { x: x + 0.2, y: 2.35, w: 2.5, h: 0.4, fontSize: 16, bold: true, color: C.text2, objectName: `db-name-${i}` });
    txt(s, `${k} · ${role}`, { x: x + 0.2, y: 2.75, w: 2.5, h: 0.75, fontSize: 13, color: C.text1, valign: "top", objectName: `db-role-${i}` });
  });
  picture(s, "fig_deck_samples", { x: 0.6, y: 3.8, w: 12.1, h: 3.0, center: true });
  s.addNotes("Kvasir v2 is the primary benchmark. GastroVision tests generalisation on a larger, imbalanced database with gastric findings. Kvasir-SEG and CVC-ClinicDB have expert polyp masks and are never trained on: we use them for cross-database testing and to score Grad-CAM. All splits are stratified 70/15/15.");

  // ------------------------------------------------------------------ 6 baselines + evaluators
  s = content("Nine baselines, ten performance evaluators", "Experimental setup");
  card(s, 0.6, 1.5, 5.9, 4.4, C.background2, "baselines-card");
  txt(s, "Baselines (same data, augmentation, optimiser, epochs)", { x: 0.9, y: 1.7, w: 5.4, h: 0.5, fontSize: 18, bold: true, color: C.text2, objectName: "baselines-head" });
  txt(s, [
    { text: "Classical: SIFT bag-of-visual-words + HSV + LBP → RBF-SVM", options: { bullet: true, breakLine: true } },
    { text: "Course CNNs: AlexNet, VGG-16, GoogLeNet", options: { bullet: true, breakLine: true } },
    { text: "Deep CNNs: ResNet-50, DenseNet-121", options: { bullet: true, breakLine: true } },
    { text: "Efficient CNNs: MobileNetV2, EfficientNet-B0", options: { bullet: true, breakLine: true } },
    { text: "Transformer: ViT-B/16", options: { bullet: true } },
  ], { x: 0.9, y: 2.35, w: 5.4, h: 4.0, fontSize: 16, color: C.text1, paraSpaceAfter: 10, valign: "top", objectName: "baselines-list" });
  card(s, 6.8, 1.5, 5.9, 4.4, C.background2, "metrics-card");
  txt(s, "Performance evaluators", { x: 7.1, y: 1.7, w: 5.4, h: 0.5, fontSize: 18, bold: true, color: C.text2, objectName: "metrics-head" });
  txt(s, [
    { text: "Classification: Accuracy, Precision, Recall, F1, Specificity, Balanced accuracy, MCC, Cohen's κ, ROC-AUC", options: { bullet: true, breakLine: true } },
    { text: "Efficiency: parameters, GMACs, batch-1 FPS on GPU and CPU", options: { bullet: true, breakLine: true } },
    { text: "Robustness: accuracy under 4 artifact types and a random artifact stream", options: { bullet: true, breakLine: true } },
    { text: "Explainability: Grad-CAM pointing game, energy-in-mask, IoU vs expert masks", options: { bullet: true } },
  ], { x: 7.1, y: 2.35, w: 5.4, h: 4.0, fontSize: 16, color: C.text1, paraSpaceAfter: 10, valign: "top", objectName: "metrics-list" });
  s.addNotes("Every deep baseline is ImageNet-pretrained and fine-tuned with exactly the same pipeline, so differences come from the architecture. MCC is our headline metric because it stays informative under class imbalance.");

  // ------------------------------------------------------------------ 7 Kvasir comparison (native chart)
  pres.addSection({ title: "Results" });
  s = content(kv.proposed ? `Kvasir v2: within ${((kv[bestBase].acc - kv.proposed.acc) * 100).toFixed(1)} points of the best model` : "Kvasir v2 results", "Results");
  const models = [...ORDER, "proposed"].filter((m) => kv[m]);
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy (%)", labels: models.map((m) => NAMES[m]), values: models.map((m) => +(kv[m].acc * 100).toFixed(2)) },
    { name: "MCC (%)", labels: models.map((m) => NAMES[m]), values: models.map((m) => +(kv[m].mcc * 100).toFixed(2)) },
  ], Object.assign({}, chartBase, {
    x: 0.6, y: 1.4, w: 8.4, h: 5.4, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: [H.accent2, H.accent3], valAxisMinVal: 50, valAxisMaxVal: 100, valAxisMajorUnit: 10,
    catAxisOrientation: "maxMin", showLegend: true, legendPos: "t", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0", showTitle: false, objectName: "kvasir-chart",
  }));
  if (kv.proposed) {
    stat(s, 9.4, 1.6, 3.3, `${pct(kv.proposed.acc)}%`, "proposed accuracy on the Kvasir v2 test set (1,200 images)", C.accent1, "kv-acc");
    stat(s, 9.4, 3.3, 3.3, `${pct(kv.proposed.mcc)}%`, "MCC; macro-F1 " + pct(kv.proposed.f1) + "%, AUC " + pct(kv.proposed.auc) + "%", C.text2, "kv-mcc");
    const ns = (MAC.Rsigns || "").split(", ").filter(Boolean).length;
    stat(s, 9.4, 5.0, 3.3, `${ns} of 8`, `deep baselines statistically tied with ours (McNemar p ≥ 0.05); only ${MAC.Rsigbetter || "–"} is significantly better`, C.accent5, "kv-sig");
  }
  s.addNotes(`Hand-crafted SIFT features reach about ${kv.classical ? pct(kv.classical.acc) : "–"}%; deep networks reach about 88-93%. On clean, curated still images our cascade is level with its own ResNet-50 backbone and a little below the best model; with clean, near-identical frames the temporal stage has nothing extra to exploit. The advantage of the design shows on degraded frames, which is the next slide.`);

  // ------------------------------------------------------------------ 8 multi-database
  const dss = ["gastrovision"].filter((d) => J(`${d}_proposed.json`));
  s = content("GastroVision: a harder, imbalanced database", "Results");
  if (dss.length) {
    const tabs = Object.fromEntries(dss.map((d) => [d, table(d)]));
    const ms = [...ORDER, "proposed"].filter((m) => dss.every((d) => tabs[d][m]));
    const hdr = [{ text: "Model", options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } }];
    dss.forEach((d) => ["Acc.", "Bal. acc.", "MCC"].forEach((k) =>
      hdr.push({ text: `${DSN[d]} ${k}`, options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, align: "center" } })));
    const best = {}; dss.forEach((d) => ["acc", "bal_acc", "mcc"].forEach((k) => { best[d + k] = Math.max(...ms.map((m) => tabs[d][m][k])); }));
    const rows = [hdr, ...ms.map((m) => {
      const isP = m === "proposed";
      const r = [{ text: NAMES[m], options: { bold: isP, fill: { color: isP ? "F6E3DE" : H.lt1 } } }];
      dss.forEach((d) => ["acc", "bal_acc", "mcc"].forEach((k) => r.push({ text: pct(tabs[d][m][k]),
        options: { align: "center", bold: Math.abs(tabs[d][m][k] - best[d + k]) < 1e-9, fill: { color: isP ? "F6E3DE" : H.lt1 } } })));
      return r;
    })];
    s.addTable(rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.5, ...Array(dss.length * 3).fill(9.6 / (dss.length * 3))],
      fontSize: 13, fontFace: "Calibri", color: H.dk1, border: { type: "solid", pt: 0.5, color: "D9E2E2" }, rowH: 0.42, objectName: "multidb-table" });
  }
  txt(s, "Balanced accuracy exposes rare-class failures that plain accuracy hides; the hand-crafted baseline collapses here. Bold = best per column.",
    { x: 0.6, y: 6.35, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.accent5, objectName: "multidb-note" });
  s.addNotes("GastroVision has many more classes and strong imbalance, so balanced accuracy and MCC matter more than accuracy. Classes with fewer than 30 images were dropped because they cannot be split into train, validation and test.");

  // ------------------------------------------------------------------ 9 robustness (native chart)
  s = content("Artifacts break single-frame models, not ours", "Results");
  if (rb && kvp) {
    const conds = ["clean", "transient_0", "transient_1", "transient_2", "transient_3", "stream_p30"];
    const labs = ["Clean", "Motion blur", "Defocus", "Specular flash", "Noise / occlusion", "Random stream"];
    const series = [["resnet50", "ResNet-50"], [bestBase in rb ? bestBase : "densenet121", NAMES[bestBase in rb ? bestBase : "densenet121"]], ["vitb16", "ViT-B/16"]]
      .filter(([m], i, a) => rb[m] && a.findIndex(([mm]) => mm === m) === i)
      .map(([m, n]) => ({ name: n, labels: labs, values: conds.map((c) => +(rb[m][c].acc * 100).toFixed(1)) }));
    series.push({ name: "Proposed", labels: labs, values: conds.map((c) => +(kvp.conditions[c]["Proposed cascade"].acc * 100).toFixed(1)) });
    s.addChart(pres.charts.BAR, series, Object.assign({}, chartBase, {
      x: 0.6, y: 1.4, w: 8.6, h: 5.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: [H.accent3, H.accent5, H.accent4, H.accent1].slice(4 - series.length), valAxisMinVal: 0, valAxisMaxVal: 100,
      valAxisMajorUnit: 20, showLegend: true, legendPos: "t", showValue: false, valAxisTitle: "Accuracy (%)", showValAxisTitle: true,
      valAxisTitleColor: H.accent5, valAxisTitleFontSize: 12, objectName: "robust-chart",
    }));
    const tr = (m) => [1, 2, 3, 4].map((i) => rb[m][`transient_${i - 1}`].acc).reduce((a, b) => a + b) / 4;
    const trP = [0, 1, 2, 3].map((i) => kvp.conditions[`transient_${i}`]["Proposed cascade"].acc).reduce((a, b) => a + b) / 4;
    const bestRob = Object.keys(rb).sort((a, b) => tr(b) - tr(a))[0];
    stat(s, 9.6, 1.6, 3.1, `${pct(trP)}%`, "proposed accuracy when the current frame is corrupted (mean of 4 artifacts)", C.accent1, "rob-prop");
    stat(s, 9.6, 3.5, 3.1, `${pct(tr(bestRob))}%`, `most robust single-frame baseline (${NAMES[bestRob]})`, C.accent5, "rob-base");
    const at = kvp.conditions.stream_p30._gate;
    txt(s, `Bi-LSTM attention: ${at.att_on_corrupted.toFixed(2)} on corrupted vs ${at.att_on_clean.toFixed(2)} on clean frames`,
      { x: 9.6, y: 5.4, w: 3.1, h: 1.0, fontSize: 14, color: C.text1, valign: "top", objectName: "rob-att" });
  }
  s.addNotes("This is the key result. When only the current frame is blurred, flashed or occluded, every single-frame model loses a large share of its accuracy because it has nothing else to look at. The temporal model keeps its clean accuracy because the attention learns to ignore corrupted frames, with no frame-quality labels.");

  // ------------------------------------------------------------------ 10 why: attention + ablation
  s = content("Why it works: attention ignores corrupted frames", "Results");
  picture(s, "fig_attention", { x: 0.6, y: 1.45, w: 7.4, h: 3.0 });
  if (kvp) {
    const comps = ["ResNet-50 (focal), single frame", "+ temporal mean pooling", "+ Bi-LSTM (no gate)", "+ gate on current frame only", "Proposed cascade"];
    const rows = [[{ text: "Configuration (Kvasir)", options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } },
      { text: "Clean", options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, align: "center" } },
      { text: "Flash", options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, align: "center" } },
      { text: "Stream", options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, align: "center" } }],
      ...comps.filter((c) => kvp.conditions.clean[c]).map((c) => [c, ...["clean", "transient_2", "stream_p30"].map((k) =>
        ({ text: pct(kvp.conditions[k][c].acc), options: { align: "center" } }))].map((v, i) =>
        (typeof v === "string" ? { text: v, options: { bold: c === "Proposed cascade" } } : v)))];
    s.addTable(rows, { x: 8.1, y: 1.45, w: 4.6, colW: [2.2, 0.8, 0.8, 0.8], fontSize: 12, fontFace: "Calibri", color: H.dk1,
      border: { type: "solid", pt: 0.5, color: "D9E2E2" }, rowH: 0.5, objectName: "ablation-table" });
  }
  txt(s, "Gating on the current frame alone lets a flash push a lesion below τ; averaging the gate score over the 5-frame buffer removes that failure at no extra cost.",
    { x: 0.6, y: 4.6, w: 7.4, h: 1.2, fontSize: 15, color: C.text1, valign: "top", objectName: "ablation-note" });
  s.addNotes("Ablation: temporal aggregation removes most of the artifact sensitivity. On our synthetic clips, simple mean pooling is about as good as the Bi-LSTM because neighbouring frames are near-duplicates; the Bi-LSTM additionally tells us which frames it trusted. Gating on the current frame alone was a weak point; the smoothed gate fixes it.");

  // ------------------------------------------------------------------ 11 efficiency
  s = content("Efficiency: what the gate buys", "Results");
  picture(s, "fig_prevalence", { x: 0.6, y: 1.45, w: 8.2, h: 3.3 });
  const cc = cascadeCost("kvasir"), c90 = cascadeCost("kvasir", 0.9);
  if (cc && eff && kvp) {
    stat(s, 9.2, 1.5, 3.5, `${pct(kvp.conditions.clean._gate.gate_recall_abn)}%`, `of abnormal test frames pass the gate (τ = ${kvp.tau.toFixed(2)})`, C.accent1, "eff-recall");
    stat(s, 9.2, 3.2, 3.5, `${cc.gmacs.toFixed(2)}`, `GMACs per frame on Kvasir vs ${eff.resnet50.gmacs.toFixed(2)} for ResNet-50 on every frame`, C.text2, "eff-gmacs");
    stat(s, 9.2, 4.9, 3.5, `${c90.gpu_fps.toFixed(0)} FPS`, `expected GPU speed at 90% normal frames (ResNet-50 on every frame: ${eff.resnet50.fps.toFixed(0)} FPS)`, C.accent2, "eff-fps");
  }
  txt(s, cc && eff ? `At Kvasir's balanced class mix the cascade only breaks even on the GPU (${cc.gpu_fps.toFixed(0)} vs ${eff.resnet50.fps.toFixed(0)} FPS); the gain grows as normal frames dominate, as in real screening video. On this CPU, PyTorch's depthwise kernels are slow, so the gate does not pay off there.` : "",
    { x: 0.6, y: 4.85, w: 8.2, h: 1.4, fontSize: 15, italic: true, color: C.accent5, valign: "top", objectName: "eff-caveat" });
  s.addNotes("The gate keeps over 99% of abnormal frames. Kvasir is class-balanced, so only a quarter of frames are bypassed and the cascade merely breaks even with ResNet-50 on the GPU while adding temporal verification. Real screening video is mostly normal mucosa, where the saving is much larger (right panel: GMACs fall towards the MobileNetV2 cost). Be upfront that on the CPU MobileNetV2 is unexpectedly slow in PyTorch, so we do not claim a CPU speed-up.");

  // ------------------------------------------------------------------ 12 cross-db + Grad-CAM
  s = content("Unseen hospitals and explainability", "Results");
  picture(s, "fig_gradcam", { x: 0.6, y: 1.4, w: 7.0, h: 5.45, center: true });
  if (cross) {
    const cv = cross.cvc;
    stat(s, 8.0, 1.5, 4.7, `${pct(cv.proposed.abnormal_recall)}%`, `of CVC-ClinicDB polyp frames (different hospital and endoscope) flagged abnormal without retraining`, C.accent1, "cross-cvc");
  }
  if (gc) {
    const k = "ResNet-50 (focal, proposed Stage 2)";
    stat(s, 8.0, 3.4, 4.7, `${pct(gc.kvasirseg[k].pointing_game)}%`, `Grad-CAM pointing-game hits inside the expert polyp mask on Kvasir-SEG (random: ${pct(gc.kvasirseg[k].mask_area)}%)`, C.text2, "gc-pg");
    stat(s, 8.0, 5.3, 4.7, `${pct(gc.cvc[k].pointing_game)}%`, `pointing-game hits on CVC-ClinicDB (random: ${pct(gc.cvc[k].mask_area)}%)`, C.accent2, "gc-pg-cvc");
  }
  s.addNotes("Without retraining, the Kvasir-trained cascade is applied to two databases it never saw. Grad-CAM is scored against expert masks: the pointing game asks whether the hottest pixel lies inside the polyp. Chance level is the mask area.");

  // ------------------------------------------------------------------ 13 conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("Conclusions and next steps", { placeholder: "title" });
  const concl = [
    ["Accurate", kv.proposed ? `${pct(kv.proposed.acc)}% accuracy and ${pct(kv.proposed.mcc)}% MCC on Kvasir v2 (best baseline: ${NAMES[bestBase]}, ${pct(kv[bestBase].acc)}%)` : ""],
    ["Robust", "keeps its clean accuracy when frames are blurred, flashed or occluded; single-frame CNNs do not"],
    ["Efficient", "≥ 99% of abnormal frames pass the gate; normal frames skip ResNet-50 and the Bi-LSTM"],
    ["Explainable", "Grad-CAM validated against expert masks on two external databases"],
  ];
  concl.forEach(([h, b], i) => {
    const y = 1.75 + i * 1.05;
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.08, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `concl-dot-${i}` });
    txt(s, h, { x: 1.5, y, w: 2.2, h: 0.6, fontSize: 20, bold: true, color: C.background1, valign: "middle", objectName: `concl-head-${i}` });
    txt(s, b, { x: 3.7, y, w: 8.8, h: 0.6, fontSize: 16, color: C.background2, valign: "middle", objectName: `concl-body-${i}` });
  });
  txt(s, "Next: validate on real endoscopic video (e.g. HyperKvasir videos), repeated-seed confidence intervals, rare gastric pathologies (early gastric cancer, ulcers).",
    { x: 0.8, y: 6.15, w: 11.7, h: 0.8, fontSize: 15, italic: true, color: C.accent4, objectName: "concl-next" });
  s.addNotes("Close with the four properties and the limitations: the clips are synthetic, so real video validation is the most important next step. Rare gastric classes like ulcers had too few images to include. Then take questions.");

  const out = path.join(ROOT, "presentation.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require("./apply_theme.js");
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
