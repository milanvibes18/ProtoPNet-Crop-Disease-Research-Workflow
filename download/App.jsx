// ─── POWER-ProtoPNet Research Platform ───────────────────────────────────────
// PlantWild AI · Explainable Plant Disease Diagnostic System
// Integrates real POWER-ProtoPNet model inference, XAI dashboard, research metrics
// Set DEMO_MODE=false and point API_BASE to your FastAPI server to use live model
import { useState, useRef, useCallback, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend, ReferenceLine, ComposedChart, Area,
} from "recharts";

// ─── Configuration ────────────────────────────────────────────────────────────
const API_BASE = typeof window !== "undefined" && window.__API_URL__ ? window.__API_URL__ : "http://localhost:8000";
const DEMO_MODE = true; // Set false when FastAPI backend is running

// ─── Design Token System ──────────────────────────────────────────────────────
const T = {
  cream:     "#F9F8F5",
  paper:     "#FFFFFF",
  forest:    "#1A3609",
  forestMd:  "#264F11",
  sage:      "#5A7A38",
  sageLt:    "#8BAA60",
  sagePale:  "#DCE9CC",
  border:    "#E6E2D8",
  muted:     "#8A8375",
  dim:       "#A8A194",
  ink:       "#16210A",
  scan:      "rgba(139,170,96,0.85)",
  dark:      "#0F172A",
  darkCard:  "#1E293B",
  darkBorder:"rgba(148,163,184,0.12)",
  green:     "#4ADE80",
  blue:      "#60A5FA",
  amber:     "#FBBF24",
  red:       "#F87171",
  purple:    "#A78BFA",
};

const cardBase = { background:T.paper, border:`1px solid rgba(0,0,0,0.05)`, borderRadius:"20px", boxShadow:"0 12px 36px -12px rgba(26,54,9,0.07)" };
const darkCardBase = { background:T.darkCard, border:`1px solid ${T.darkBorder}`, borderRadius:"14px" };

// ─── Real Training Data (from parsed_training_data.json) ─────────────────────
const PHASE2 = [
  {ep:1,  train:95.94, val:98.26, loss:0.785,  gap:-2.32},
  {ep:2,  train:98.57, val:99.02, loss:0.758,  gap:-0.45},
  {ep:3,  train:99.22, val:99.39, loss:0.744,  gap:-0.17},
  {ep:4,  train:99.49, val:99.51, loss:0.745,  gap:-0.02},
  {ep:5,  train:99.67, val:99.60, loss:0.746,  gap: 0.07},
  {ep:6,  train:99.77, val:99.61, loss:0.746,  gap: 0.16},
  {ep:7,  train:99.77, val:99.39, loss:0.753,  gap: 0.38},
  {ep:8,  train:99.82, val:99.70, loss:0.741,  gap: 0.12},
  {ep:9,  train:99.86, val:99.70, loss:0.744,  gap: 0.16},
  {ep:10, train:99.91, val:99.72, loss:0.741,  gap: 0.19},
  {ep:11, train:99.89, val:99.72, loss:0.744,  gap: 0.17},
  {ep:12, train:99.91, val:99.74, loss:0.743,  gap: 0.17},
  {ep:13, train:99.95, val:99.74, loss:0.744,  gap: 0.21},
  {ep:14, train:99.94, val:99.72, loss:0.743,  gap: 0.22},
  {ep:15, train:99.93, val:99.72, loss:0.744,  gap: 0.21},
  {ep:16, train:99.95, val:99.77, loss:0.741,  gap: 0.18},  // BEST EP
  {ep:17, train:99.95, val:99.70, loss:0.745,  gap: 0.25},
  {ep:18, train:99.95, val:99.68, loss:0.743,  gap: 0.27},
  {ep:19, train:99.98, val:99.68, loss:0.744,  gap: 0.30},
  {ep:20, train:99.95, val:99.72, loss:0.742,  gap: 0.23},
  {ep:21, train:99.96, val:99.68, loss:0.745,  gap: 0.28},
  {ep:22, train:99.98, val:99.74, loss:0.743,  gap: 0.24},
  {ep:23, train:99.95, val:99.68, loss:0.743,  gap: 0.27},
  {ep:24, train:99.98, val:99.68, loss:0.743,  gap: 0.30},
  {ep:25, train:99.97, val:99.70, loss:0.744,  gap: 0.27},
  {ep:26, train:99.98, val:99.70, loss:0.744,  gap: 0.28},
  {ep:27, train:99.98, val:99.72, loss:0.743,  gap: 0.26},
  {ep:28, train:99.98, val:99.74, loss:0.742,  gap: 0.24},
];
const PROTO_PUSH_EPS = [2,4,6,8,10,12,14,16,18,20,22,24,26,28];
const PROTO_PUSH_IMPACT = [
  {ep:2,  delta:0.76, pos:true},  {ep:4,  delta:0.12,  pos:true},
  {ep:6,  delta:0.01, pos:true},  {ep:8,  delta:0.31,  pos:true},
  {ep:10, delta:0.02, pos:true},  {ep:12, delta:0.02,  pos:true},
  {ep:14, delta:-0.02,pos:false}, {ep:16, delta:0.05,  pos:true},
  {ep:18, delta:-0.02,pos:false}, {ep:20, delta:0.04,  pos:true},
  {ep:22, delta:0.06, pos:true},  {ep:24, delta:0.00,  pos:true},
  {ep:26, delta:0.00, pos:true},  {ep:28, delta:0.02,  pos:true},
];

// ─── 38 PlantWild v3 Classes (real alphabetical order from training) ──────────
// Source: classification_report.txt · test accuracy 99.65% on 5700 samples
const CLASSES = [
  "Apple_scab(Apple)",                     //  0
  "Bacterial_spot(Peach)",                 //  1
  "Bacterial_spot(Pepper)",                //  2
  "Bacterial_spot(Tomato)",                //  3
  "Black_rot(Apple)",                      //  4
  "Black_rot(Grape)",                      //  5
  "Cedar_apple_rust(Apple)",               //  6
  "Cercospora_leaf_spot(Corn)",            //  7
  "Common_rust(Corn)",                     //  8
  "Early_blight(Potato)",                  //  9
  "Early_blight(Tomato)",                  // 10
  "Esca(Grape)",                           // 11
  "Haunglongbing(Orange)",                 // 12
  "Late_blight(Potato)",                   // 13
  "Late_blight(Tomato)",                   // 14
  "Leaf_Mold(Tomato)",                     // 15
  "Leaf_blight(Grape)",                    // 16
  "Leaf_scorch(Strawberry)",               // 17
  "Northern_Leaf_Blight(Corn)",            // 18
  "Powdery_mildew(Cherry)",               // 19
  "Septoria_leaf_spot(Tomato)",            // 20
  "Spider_mites(Tomato)",                  // 21
  "Target_Spot(Tomato)",                   // 22
  "Tomato_Yellow_Leaf_Curl_Virus(Tomato)", // 23
  "Tomato_mosaic_virus(Tomato)",           // 24
  "healthy(Apple)",                        // 25
  "healthy(Blueberry)",                    // 26
  "healthy(Cherry)",                       // 27
  "healthy(Corn)",                         // 28
  "healthy(Grape)",                        // 29
  "healthy(Orange)",                       // 30
  "healthy(Peach)",                        // 31
  "healthy(Pepper)",                       // 32
  "healthy(Potato)",                       // 33
  "healthy(Raspberry)",                    // 34
  "healthy(Soybean)",                      // 35
  "healthy(Strawberry)",                   // 36
  "healthy(Tomato)",                       // 37
];

// ─── Disease Information Database ─────────────────────────────────────────────
// ─── Disease Information Database (keyed by REAL alphabetical class index) ────
const DISEASE_DB = {
  19: { // Powdery_mildew(Cherry) — real index 19
    scientific: "Podosphaera clandestina",
    causal: "Fungal",
    severity: "Moderate",
    description: "An obligate biotrophic fungus causing white powdery mycelial colonies on cherry leaf surfaces. Thrives in warm, dry conditions with high ambient humidity, reducing photosynthetic efficiency.",
    symptoms: ["White powdery patches on adaxial leaf surface","Circular lesion patterns near margins","Leaf curling and distortion","Premature defoliation","Reduced fruit set"],
    actions: [
      { level: "urgent", text: "Apply sulfur-based or DMI fungicide within 48 hours" },
      { level: "high",   text: "Remove and destroy heavily infected leaves" },
      { level: "medium", text: "Improve canopy air circulation by pruning" },
      { level: "low",    text: "Switch to drip irrigation to reduce foliar wetness" },
    ],
  },
  14: { // Late_blight(Tomato) — real index 14
    scientific: "Phytophthora infestans",
    causal: "Oomycete",
    severity: "Critical",
    description: "The causative agent of the Irish Great Famine (1845). Modern strains overcome resistance genes rapidly. Can destroy entire tomato fields within 7–14 days under cool, wet conditions.",
    symptoms: ["Dark water-soaked lesions on leaves","White mold on abaxial surface","Rapid tissue necrosis","Brown-black stem cankers","Fruit rot with hard brown lesions"],
    actions: [
      { level: "critical", text: "Immediately remove and destroy all infected material" },
      { level: "urgent",   text: "Apply copper-based or mancozeb fungicide to entire crop" },
      { level: "high",     text: "Alert neighboring farms for coordinated response" },
      { level: "medium",   text: "Improve drainage; reduce leaf wetness duration" },
    ],
  },
  13: { // Late_blight(Potato) — real index 13
    scientific: "Phytophthora infestans",
    causal: "Oomycete",
    severity: "Critical",
    description: "Same pathogen as tomato late blight. Airborne sporangia can travel miles. Causes catastrophic tuber rot in storage if not managed immediately.",
    symptoms: ["Dark water-soaked lesions","White mycelium on leaf underside","Rapid brown-black necrosis","Foul-smelling tuber rot"],
    actions: [
      { level: "critical", text: "Destroy all infected foliage immediately" },
      { level: "urgent",   text: "Apply metalaxyl-M or fluopicolide at first symptoms" },
      { level: "high",     text: "Harvest tubers early to prevent storage losses" },
    ],
  },
  12: { // Haunglongbing(Orange) — real index 12
    scientific: "Candidatus Liberibacter asiaticus",
    causal: "Bacterial",
    severity: "Critical",
    description: "Citrus greening (HLB): the most destructive citrus disease globally, transmitted by the Asian citrus psyllid. No cure exists — infected trees must be destroyed.",
    symptoms: ["Asymmetric blotchy-mottle yellowing","Small, lopsided bitter fruit","Stunted shoot growth","Zinc-deficiency-like symptoms"],
    actions: [
      { level: "critical", text: "Contact plant health authority — notifiable disease" },
      { level: "critical", text: "Remove and destroy infected trees" },
      { level: "urgent",   text: "Control psyllid vector with systemic insecticide" },
    ],
  },
  23: { // Tomato_Yellow_Leaf_Curl_Virus — real index 23
    scientific: "Tomato yellow leaf curl virus (TYLCV)",
    causal: "Viral",
    severity: "Critical",
    description: "Begomovirus transmitted by whitefly Bemisia tabaci. Causes severe stunting and up to 100% yield loss. No chemical cure — vector control is the only option.",
    symptoms: ["Upward and inward leaf curling","Yellow leaf margins (chlorosis)","Stunted plant growth","Small distorted fruit"],
    actions: [
      { level: "critical", text: "Remove and bag infected plants immediately" },
      { level: "urgent",   text: "Apply systemic insecticide to control whitefly vector" },
      { level: "high",     text: "Install UV-reflective mulch to deter whiteflies" },
    ],
  },
};

const FALLBACK_DISEASE = {
  scientific: "See full diagnostic report",
  causal: "Mixed",
  severity: "Moderate",
  description: "Analysis complete. Refer to the confidence score and prototype activations for detailed explanation of the model's diagnostic reasoning.",
  symptoms: ["Pattern detected in uploaded specimen"],
  actions: [{ level: "medium", text: "Consult agronomist for treatment protocol" }],
};

// ─── Demo Inference Data (real indices: Powdery_mildew(Cherry) = 19) ──────────
const DEMO_RESULT = {
  prediction: {
    class_name: "Powdery_mildew(Cherry)",
    class_index: 19,
    confidence: 0.9854,
    top_k: [
      { name: "Powdery_mildew(Cherry)",  prob: 0.9854, idx: 19 },
      { name: "Apple_scab(Apple)",       prob: 0.0082, idx: 0  },
      { name: "Leaf_blight(Grape)",      prob: 0.0034, idx: 16 },
      { name: "Leaf_scorch(Strawberry)", prob: 0.0019, idx: 17 },
      { name: "Early_blight(Tomato)",    prob: 0.0011, idx: 10 },
    ],
  },
  prototypes: [
    // Prototype IDs: class_idx * 5 + rank_within_class = 19*5 = 95..99
    { id:97, class_name:"Powdery_mildew(Cherry)", similarity:0.924, rank:1, desc:"Dense white sporulation on adaxial leaf surface" },
    { id:98, class_name:"Powdery_mildew(Cherry)", similarity:0.891, rank:2, desc:"Circular lesion expansion near leaf margin" },
    { id:96, class_name:"Powdery_mildew(Cherry)", similarity:0.843, rank:3, desc:"Fungal colony in early expansion phase" },
    { id:95, class_name:"Powdery_mildew(Cherry)", similarity:0.773, rank:4, desc:"Sub-marginal chlorotic halo pattern" },
    { id:99, class_name:"Powdery_mildew(Cherry)", similarity:0.634, rank:5, desc:"Low-density secondary lesion cluster" },
  ],
  disease_info: DISEASE_DB[19],
  gradcam_url: null,
  shap_url: null,
};

// ─── Architecture Pipeline Stages ─────────────────────────────────────────────
const ARCH_STAGES = [
  { id:"input", label:"Input", title:"Input Image", shape:"B × 3 × 224 × 224", params:"—", color:T.blue,
    desc:"RGB image normalized to ImageNet statistics. Preprocessing: resize → centre-crop 224×224 → normalize (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]).",
    formula:"x ∈ ℝ^{B×3×224×224}" },
  { id:"backbone", label:"ConvNeXt", title:"ConvNeXt-Tiny Backbone", shape:"B × 768 × 7 × 7", params:"28.6M", color:T.purple,
    desc:"ImageNet-1K pretrained backbone. 7×7 depthwise convolutions, Layer Norm, GELU activations, residual connections. Fine-tuned at lr=2×10⁻⁵ to prevent catastrophic forgetting.",
    formula:"f = ConvNeXt(x), f ∈ ℝ^{B×768×7×7}" },
  { id:"se", label:"SE Attn", title:"Squeeze-and-Excitation Block", shape:"B × 768 × 7 × 7", params:"≈75K", color:T.amber,
    desc:"Channel-wise recalibration (r=16). Amplifies disease-relevant channels (colour, texture); suppresses background. Bottleneck: 768 → 48 → 768 hidden units.",
    formula:"e = σ(W₂·ReLU(W₁·GAP(f))), out = f ⊗ e" },
  { id:"srm", label:"SRM", title:"Spatial Refinement Module", shape:"B × 256 × 7 × 7", params:"≈200K", color:T.green,
    desc:"Depthwise-separable convolution reduces 768→256 channels at 8× parameter efficiency. BN-ReLU-DWConv3×3-BN-ReLU-PWConv pipeline. Spatial dropout p=0.3 for regularisation.",
    formula:"y = BN(PW(BN(ReLU(BN(DW(x))))))" },
  { id:"proto", label:"Prototype", title:"Prototype Layer (190 prototypes)", shape:"B × 190", params:"190×256", color:T.sageLt,
    desc:"5 prototype vectors per class × 38 classes = 190 total. Each prototype is a 256-dim vector. Cosine similarity computed patch-by-patch; max over spatial positions retained. Pushed to real training patches every 2 epochs.",
    formula:"sim[b,j] = max_s cosine(f̂[b,:,s], p̂_j)" },
  { id:"fc", label:"FC Head", title:"FC Classification Head", shape:"B × 38", params:"38×190", color:T.red,
    desc:"Bias-free linear layer. Initialized: own-class prototypes +1.0, others −0.5. L1 sparsity (λ=10⁻⁵) in Phase 4 encourages sparse, interpretable connections.",
    formula:"logits = W_fc · sim, W_fc ∈ ℝ^{38×190}" },
  { id:"output", label:"Output", title:"Softmax Prediction", shape:"B × 38", params:"—", color:T.forest,
    desc:"Softmax converts logits to class probabilities. Prediction = argmax(P). Confidence = max(P). Every decision is traceable back to prototype similarities, making the model fully interpretable.",
    formula:"P(c|x) = exp(l_c) / Σ_k exp(l_k)" },
];

// ─── Utility Functions ─────────────────────────────────────────────────────────
const toTitle = (s) => s.replace(/\b\w/g, c => c.toUpperCase());

const SEV_STYLES = {
  "Low":      { bg:"#ECFDF5", text:"#065F46", dot:"#10B981" },
  "Moderate": { bg:"#FFFBEB", text:"#92400E", dot:"#F59E0B" },
  "High":     { bg:"#FEF2F2", text:"#991B1B", dot:"#EF4444" },
  "Critical": { bg:"#FFF1F2", text:"#881337", dot:"#E11D48" },
  "Healthy":  { bg:"#F0FDF4", text:"#14532D", dot:"#22C55E" },
};

const CAUSE_COLOR = { Fungal:T.amber, Bacterial:T.blue, Viral:T.purple, Oomycete:T.red, Healthy:T.green };

const URGENCY_STYLES = {
  critical: { bg:"#FFF1F2", text:"#881337", dot:"#E11D48", label:"CRITICAL" },
  urgent:   { bg:"#FEF2F2", text:"#991B1B", dot:"#EF4444", label:"URGENT" },
  high:     { bg:"#FFFBEB", text:"#92400E", dot:"#F59E0B", label:"HIGH" },
  medium:   { bg:"#EFF6FF", text:"#1E40AF", dot:"#3B82F6", label:"MEDIUM" },
  low:      { bg:"#F0FDF4", text:"#14532D", dot:"#22C55E", label:"LOW" },
};

// ─── API Layer ─────────────────────────────────────────────────────────────────
const runPredict = (imageFile) =>
  DEMO_MODE
    ? new Promise(r => setTimeout(() => r(DEMO_RESULT), 2400))
    : fetch(`${API_BASE}/predict`, { method:"POST", body: Object.assign(new FormData(), { file: imageFile }) }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

// ─── SVG Components ────────────────────────────────────────────────────────────
const LeafMark = ({ size=32, fill=T.sageLt, stroke=T.forestMd }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="M16 4C16 4 7 9 7 17C7 23.627 11.373 28 16 28C20.627 28 25 23.627 25 17C25 9 16 4 16 4Z" fill={fill} stroke={stroke} strokeWidth="1.4"/>
    <path d="M16 8L16 25M12 13C13.8 15.5 15 17.5 16 20C17 17.5 18.2 15.5 20 13" stroke={stroke} strokeWidth="1.3" strokeLinecap="round"/>
  </svg>
);

const ChevronIcon = ({ open }) => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ transform: open?"rotate(180deg)":"rotate(0deg)", transition:"transform 0.25s" }}>
    <path d="M4.5 6.75L9 11.25L13.5 6.75" stroke={T.muted} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// ─── Confidence Gauge (original, preserved) ────────────────────────────────────
const ConfidenceGauge = ({ value }) => {
  const pct = Math.round(value * 100);
  const R = 68, circ = 2 * Math.PI * R, dash = (pct/100)*circ;
  const meta = pct >= 90 ? { color:"#1A5E0A", track:"#22863A", bg:"#E6F4E0", label:"High Confidence" }
             : pct >= 70 ? { color:"#92400E", track:"#D97706", bg:"#FEF3C7", label:"Moderate Confidence" }
             :             { color:"#991B1B", track:"#DC2626", bg:"#FEE2E2", label:"Low Confidence" };
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:"12px" }}>
      <svg width="176" height="176" viewBox="0 0 176 176">
        <circle cx="88" cy="88" r={R} fill="none" stroke="#F0EFEA" strokeWidth="13"/>
        <circle cx="88" cy="88" r={R} fill="none" stroke={meta.track} strokeWidth="13"
          strokeLinecap="round" strokeDasharray={`${dash} ${circ}`} transform="rotate(-90 88 88)"
          style={{ transition:"stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1)" }}/>
        <circle cx="88" cy="88" r="52" fill={meta.bg}/>
        <text x="88" y="82" textAnchor="middle" fontFamily="'DM Serif Display',serif" fontSize="28" fontWeight="700" fill={meta.color}>{pct}%</text>
        <text x="88" y="102" textAnchor="middle" fontFamily="'DM Sans',sans-serif" fontSize="11" fill={meta.color} opacity="0.75">CONFIDENCE</text>
      </svg>
      <span style={{ display:"inline-flex", alignItems:"center", gap:"6px", background:meta.bg, color:meta.color, padding:"5px 14px", borderRadius:"99px", fontSize:"12px", fontWeight:"600" }}>
        <span style={{ width:"7px", height:"7px", borderRadius:"50%", background:meta.track, display:"inline-block" }}/>
        {meta.label}
      </span>
    </div>
  );
};

// ─── Section Header ────────────────────────────────────────────────────────────
const SectionTag = ({ tag, title, subtitle, dark=false }) => (
  <div style={{ marginBottom:"28px" }}>
    <p style={{ fontSize:"11px", fontWeight:700, letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:"8px",
      color: dark ? T.green : T.sage, fontFamily:"'DM Sans',sans-serif" }}>{tag}</p>
    <h2 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"clamp(24px,3vw,32px)", color: dark ? "#F8FAFC" : T.forest, marginBottom:"8px" }}>{title}</h2>
    {subtitle && <p style={{ fontSize:"14px", color: dark ? "rgba(248,250,252,0.6)" : T.muted, lineHeight:1.6 }}>{subtitle}</p>}
  </div>
);

// ─── Metric Summary Card ───────────────────────────────────────────────────────
const MetricCard = ({ label, value, sub, color=T.green }) => (
  <div style={{ ...darkCardBase, padding:"20px 24px" }}>
    <div style={{ fontSize:"11px", fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:"rgba(148,163,184,0.8)", marginBottom:"8px" }}>{label}</div>
    <div style={{ fontFamily:"'DM Serif Display',serif", fontSize:"28px", color, fontWeight:700 }}>{value}</div>
    {sub && <div style={{ fontSize:"12px", color:"rgba(148,163,184,0.65)", marginTop:"4px" }}>{sub}</div>}
  </div>
);

// ─── Dark Chart Tooltip ────────────────────────────────────────────────────────
const DarkTooltip = ({ active, payload, label, unit="" }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:"#0F172A", border:`1px solid ${T.darkBorder}`, borderRadius:"8px", padding:"10px 14px",
      boxShadow:"0 8px 24px rgba(0,0,0,0.4)" }}>
      <div style={{ fontSize:"12px", color:"rgba(148,163,184,0.7)", marginBottom:"6px" }}>Epoch {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:i<payload.length-1?"4px":0 }}>
          <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:p.color }}/>
          <span style={{ fontSize:"12px", color:"#F1F5F9" }}>{p.name}: </span>
          <span style={{ fontSize:"12px", fontWeight:700, color:p.color }}>{typeof p.value==="number"? p.value.toFixed(4):p.value}{unit}</span>
        </div>
      ))}
    </div>
  );
};

const LightTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:T.paper, border:`1px solid ${T.border}`, borderRadius:"8px", padding:"8px 12px",
      fontSize:"12px", boxShadow:"0 4px 16px rgba(0,0,0,0.08)" }}>
      <div style={{ fontWeight:600, color:T.forest, marginBottom:"2px" }}>{payload[0].payload.name}</div>
      <div style={{ color:T.muted }}>{payload[0].value.toFixed(2)}%</div>
    </div>
  );
};

// ─── Prototype Card (real, with actual ID, similarity, class) ─────────────────
const PrototypeCard = ({ proto, maxSim }) => {
  const pct = Math.round((proto.similarity / maxSim) * 100);
  return (
    <div className="premium-card" style={{ ...cardBase, padding:"16px", display:"flex", flexDirection:"column", gap:"10px" }}>
      {/* Prototype visual area */}
      <div style={{ position:"relative", borderRadius:"10px", overflow:"hidden", aspectRatio:"1/1", background:`linear-gradient(135deg, ${T.sagePale} 0%, ${T.cream} 100%)`, display:"flex", alignItems:"center", justifyContent:"center" }}>
        <div style={{ textAlign:"center" }}>
          <LeafMark size={28} fill="#C0D8A0" stroke={T.sageLt}/>
          <div style={{ marginTop:"4px", fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:T.forest }}>#{proto.id}</div>
        </div>
        {proto.rank === 1 && (
          <div style={{ position:"absolute", top:"7px", right:"7px", background:T.forestMd, color:"#fff",
            fontSize:"9px", fontWeight:700, padding:"3px 7px", borderRadius:"99px", letterSpacing:"0.06em" }}>TOP</div>
        )}
        <div style={{ position:"absolute", bottom:"7px", left:"7px", background:"rgba(26,54,9,0.7)", backdropFilter:"blur(6px)",
          color:"#E6F4E0", fontSize:"9px", fontWeight:600, padding:"3px 8px", borderRadius:"6px" }}>
          Rank #{proto.rank}
        </div>
      </div>
      {/* Prototype info */}
      <div>
        <div style={{ fontSize:"10px", fontWeight:700, color:T.sage, textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:"3px" }}>
          Prototype #{proto.id} · {proto.class_name}
        </div>
        <div style={{ fontSize:"11px", color:T.muted, lineHeight:1.4 }}>{proto.desc}</div>
      </div>
      {/* Similarity bar */}
      <div>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"4px" }}>
          <span style={{ fontSize:"10px", color:T.dim, fontWeight:600 }}>COSINE SIM</span>
          <span style={{ fontSize:"12px", color:T.forest, fontWeight:700 }}>{proto.similarity.toFixed(3)}</span>
        </div>
        <div style={{ background:T.sagePale, borderRadius:"99px", height:"5px", overflow:"hidden" }}>
          <div style={{ height:"100%", width:`${pct}%`, background:`linear-gradient(90deg,${T.sageLt},${T.forestMd})`, borderRadius:"99px",
            transition:"width 1s cubic-bezier(.4,0,.2,1) 0.3s" }}/>
        </div>
      </div>
    </div>
  );
};

// ─── Disease Info Panel ────────────────────────────────────────────────────────
const DiseaseInfoPanel = ({ info, className }) => {
  const sev = SEV_STYLES[info.severity] || SEV_STYLES["Moderate"];
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ ...cardBase, padding:"28px", marginBottom:"0" }}>
      <div style={{ display:"flex", flexWrap:"wrap", alignItems:"flex-start", justifyContent:"space-between", gap:"12px", marginBottom:"20px" }}>
        <div>
          <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Identified Pathogen</div>
          <h3 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"22px", color:T.forest, fontStyle:"italic" }}>{info.scientific}</h3>
        </div>
        <div style={{ display:"flex", gap:"8px", flexWrap:"wrap" }}>
          <span style={{ display:"inline-flex", alignItems:"center", gap:"5px", background:sev.bg, color:sev.text, padding:"5px 12px", borderRadius:"99px", fontSize:"11px", fontWeight:700 }}>
            <span style={{ width:"6px", height:"6px", borderRadius:"50%", background:sev.dot }}/>
            {info.severity} Severity
          </span>
          <span style={{ display:"inline-flex", alignItems:"center", gap:"5px", background:"rgba(139,170,96,0.15)", color:T.forestMd, padding:"5px 12px", borderRadius:"99px", fontSize:"11px", fontWeight:700 }}>
            <span style={{ width:"6px", height:"6px", borderRadius:"50%", background: CAUSE_COLOR[info.causal] || T.sageLt }}/>
            {info.causal}
          </span>
        </div>
      </div>
      <p style={{ fontSize:"14px", color:T.muted, lineHeight:1.7, marginBottom:"20px" }}>{info.description}</p>
      {/* Symptoms */}
      <div style={{ marginBottom:"20px" }}>
        <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"10px" }}>Key Symptoms</div>
        <div style={{ display:"flex", flexDirection:"column", gap:"6px" }}>
          {info.symptoms.map((s, i) => (
            <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:"8px" }}>
              <div style={{ width:"5px", height:"5px", borderRadius:"50%", background:T.sageLt, marginTop:"6px", flexShrink:0 }}/>
              <span style={{ fontSize:"13px", color:T.ink, lineHeight:1.5 }}>{s}</span>
            </div>
          ))}
        </div>
      </div>
      {/* Recommended actions */}
      <div style={{ borderTop:`1px solid ${T.border}`, paddingTop:"20px" }}>
        <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"12px" }}>Recommended Actions</div>
        <div style={{ display:"flex", flexDirection:"column", gap:"8px" }}>
          {info.actions.map((a, i) => {
            const us = URGENCY_STYLES[a.level] || URGENCY_STYLES.medium;
            return (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:"10px", background:us.bg, borderRadius:"10px", padding:"10px 14px" }}>
                <span style={{ fontSize:"9px", fontWeight:800, color:us.text, background:"rgba(0,0,0,0.06)", padding:"2px 6px", borderRadius:"4px", letterSpacing:"0.08em", flexShrink:0 }}>{us.label}</span>
                <span style={{ fontSize:"13px", color:us.text, lineHeight:1.4 }}>{a.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// ─── GradCAM Panel ────────────────────────────────────────────────────────────
const GradCAMPanel = ({ imageUrl, gradcamUrl }) => {
  const [mode, setMode] = useState("overlay");
  const [intensity, setIntensity] = useState(0.7);
  const isReal = !!gradcamUrl;

  const TAB = ({ id, label }) => (
    <button onClick={()=>setMode(id)} style={{ background: mode===id ? T.forestMd : "transparent",
      color: mode===id ? "#fff" : T.muted, border:"none", padding:"6px 14px", borderRadius:"8px",
      fontSize:"12px", fontWeight:600, cursor:"pointer", transition:"all 0.2s" }}>{label}</button>
  );

  return (
    <div style={{ ...cardBase, padding:"28px" }}>
      <div style={{ display:"flex", flexWrap:"wrap", alignItems:"center", justifyContent:"space-between", gap:"12px", marginBottom:"20px" }}>
        <div>
          <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>GradCAM++ Activation</div>
          <h3 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"20px", color:T.forest }}>Spatial Feature Attribution</h3>
        </div>
        <div style={{ display:"flex", gap:"4px", background:T.cream, borderRadius:"10px", padding:"4px" }}>
          <TAB id="original" label="Original"/><TAB id="overlay" label="Overlay"/><TAB id="side" label="Side by Side"/>
        </div>
      </div>
      {!isReal && (
        <div style={{ background:T.sagePale, border:`1px solid ${T.border}`, borderRadius:"10px", padding:"10px 16px", marginBottom:"16px", fontSize:"12px", color:T.forestMd, display:"flex", alignItems:"center", gap:"8px" }}>
          <span>ℹ️</span>
          <span>Demo mode: simulated heatmap shown. Connect backend for real GradCAM++ output.</span>
        </div>
      )}
      <div style={{ display:"flex", flexWrap:"wrap", gap:"16px" }}>
        {(mode==="original" || mode==="side") && (
          <div style={{ flex:"1 1 240px", borderRadius:"12px", overflow:"hidden", border:`1px solid ${T.border}` }}>
            <div style={{ padding:"8px 12px", background:T.cream, borderBottom:`1px solid ${T.border}`, fontSize:"11px", fontWeight:600, color:T.muted, textTransform:"uppercase", letterSpacing:"0.07em" }}>Original</div>
            <img src={imageUrl} alt="Original" style={{ width:"100%", height:"260px", objectFit:"cover", display:"block" }}/>
          </div>
        )}
        {(mode==="overlay" || mode==="side") && (
          <div style={{ flex:"1 1 240px", borderRadius:"12px", overflow:"hidden", border:`1px solid ${T.border}` }}>
            <div style={{ padding:"8px 12px", background:T.dark, borderBottom:`1px solid ${T.darkBorder}`, fontSize:"11px", fontWeight:600, color:"rgba(148,163,184,0.8)", textTransform:"uppercase", letterSpacing:"0.07em" }}>
              GradCAM++ {isReal ? "(Live)" : "(Simulated)"}
            </div>
            <div style={{ position:"relative", height:"260px" }}>
              {imageUrl && <img src={isReal?gradcamUrl:imageUrl} alt="Heatmap" style={{ width:"100%", height:"100%", objectFit:"cover", display:"block" }}/>}
              {!isReal && <div style={{ position:"absolute", inset:0, background:`radial-gradient(ellipse 55% 45% at 42% 52%, rgba(234,179,8,${intensity}) 0%, rgba(239,68,68,${intensity*0.7}) 35%, rgba(59,130,246,${intensity*0.3}) 70%, transparent 100%)`, mixBlendMode:"screen" }}/>}
            </div>
          </div>
        )}
      </div>
      <div style={{ marginTop:"16px", display:"flex", alignItems:"center", gap:"12px" }}>
        <span style={{ fontSize:"12px", color:T.muted, whiteSpace:"nowrap" }}>Heatmap Intensity</span>
        <input type="range" min="0.2" max="1" step="0.05" value={intensity}
          onChange={e => setIntensity(parseFloat(e.target.value))}
          style={{ flex:1, accentColor:T.forestMd }}/>
        <span style={{ fontSize:"12px", color:T.forest, fontWeight:700 }}>{Math.round(intensity*100)}%</span>
      </div>
      <p style={{ marginTop:"12px", fontSize:"13px", color:T.muted, lineHeight:1.6 }}>
        GradCAM++ computes gradient-weighted class activation maps using the final convolutional feature maps to localise discriminative regions. Warm colours (yellow→red) indicate high-activation areas driving the classification decision.
      </p>
    </div>
  );
};

// ─── SHAP Panel ───────────────────────────────────────────────────────────────
const SHAPPanel = ({ shapUrl }) => {
  const isReal = !!shapUrl;
  // Pseudo-random 14×14 SHAP grid for demo
  const GRID = Array.from({length:196}, (_, i) => {
    const v = Math.sin(i * 0.37) * Math.cos(i * 0.21) * 0.8 + Math.sin(i * 1.1) * 0.2;
    return v;
  });
  const max = Math.max(...GRID.map(Math.abs));

  return (
    <div style={{ ...cardBase, padding:"28px" }}>
      <div style={{ marginBottom:"20px" }}>
        <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>SHAP Explainability</div>
        <h3 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"20px", color:T.forest }}>Pixel-Level Feature Attribution</h3>
      </div>
      {!isReal && (
        <div style={{ background:"#EFF6FF", border:"1px solid #DBEAFE", borderRadius:"10px", padding:"10px 16px", marginBottom:"16px", fontSize:"12px", color:"#1E40AF", display:"flex", alignItems:"center", gap:"8px" }}>
          <span>ℹ️</span>
          <span>Demo mode: synthetic SHAP grid shown. Connect backend for KernelSHAP on real model.</span>
        </div>
      )}
      <div style={{ display:"flex", flexWrap:"wrap", gap:"24px", alignItems:"flex-start" }}>
        {/* SHAP heatmap grid */}
        <div style={{ flex:"0 0 auto" }}>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(14,1fr)", gap:"2px", width:"280px" }}>
            {GRID.map((v, i) => {
              const n = v / max;
              const bg = n > 0 ? `rgba(16,185,129,${Math.abs(n)*0.9})` : `rgba(239,68,68,${Math.abs(n)*0.9})`;
              return <div key={i} style={{ width:"18px", height:"18px", borderRadius:"3px", background:bg }}/>;
            })}
          </div>
          {/* Legend */}
          <div style={{ display:"flex", alignItems:"center", gap:"8px", marginTop:"10px" }}>
            <div style={{ display:"flex", alignItems:"center", gap:"4px" }}>
              <div style={{ width:"12px", height:"12px", borderRadius:"2px", background:"rgba(16,185,129,0.9)" }}/>
              <span style={{ fontSize:"10px", color:T.muted }}>Positive (supports class)</span>
            </div>
            <div style={{ display:"flex", alignItems:"center", gap:"4px" }}>
              <div style={{ width:"12px", height:"12px", borderRadius:"2px", background:"rgba(239,68,68,0.9)" }}/>
              <span style={{ fontSize:"10px", color:T.muted }}>Negative (against class)</span>
            </div>
          </div>
        </div>
        {/* Interpretation */}
        <div style={{ flex:"1 1 200px" }}>
          <div style={{ marginBottom:"16px" }}>
            <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:"8px" }}>Positive Evidence Regions</div>
            {["Central leaf lesion zone","Powdery margin patterns","Adaxial surface texture"].map((s,i) => (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"6px" }}>
                <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:"#10B981" }}/>
                <span style={{ fontSize:"12px", color:T.ink }}>{s}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize:"11px", fontWeight:700, color:T.red, textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:"8px" }}>Suppressive Regions</div>
            {["Background foliage (uninformative)","Healthy tissue outside lesion"].map((s,i) => (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"6px" }}>
                <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:T.red }}/>
                <span style={{ fontSize:"12px", color:T.ink }}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <p style={{ marginTop:"16px", fontSize:"13px", color:T.muted, lineHeight:1.6 }}>
        SHAP (SHapley Additive exPlanations) computes marginal feature contributions using game-theoretic Shapley values. Green regions are positively correlated with the predicted class; red regions reduce class confidence. This analysis validates that the model relies on genuine pathological features rather than background artifacts.
      </p>
    </div>
  );
};

// ─── Training Curve Chart ─────────────────────────────────────────────────────
const TrainingCurveChart = () => (
  <div style={{ ...darkCardBase, padding:"24px" }}>
    <div style={{ marginBottom:"16px" }}>
      <div style={{ fontSize:"11px", fontWeight:700, color:"rgba(148,163,184,0.7)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Phase 2 · Joint Training</div>
      <div style={{ fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:"#F8FAFC" }}>Train vs Validation Accuracy</div>
    </div>
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={PHASE2} margin={{left:-10,right:20,top:4,bottom:4}}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)"/>
        <XAxis dataKey="ep" tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false}/>
        <YAxis domain={[95.5,100.2]} tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false} tickFormatter={v=>`${v}%`}/>
        <Tooltip content={<DarkTooltip unit="%"/>}/>
        <Legend wrapperStyle={{fontSize:"11px",color:"rgba(148,163,184,0.7)"}}/>
        {PROTO_PUSH_EPS.map(ep => <ReferenceLine key={ep} x={ep} stroke="rgba(167,139,250,0.3)" strokeDasharray="3 3"/>)}
        <ReferenceLine x={28} stroke="rgba(248,113,113,0.6)" strokeDasharray="5 3" label={{value:"Early Stop",fill:"rgba(248,113,113,0.7)",fontSize:10,position:"top"}}/>
        <ReferenceLine x={16} stroke={T.amber} strokeDasharray="0" dot={false} label={{value:"★ Best",fill:T.amber,fontSize:10,position:"insideTopRight"}}/>
        <Line name="Train Acc" type="monotone" dataKey="train" stroke={T.blue} strokeWidth={2} dot={false} activeDot={{r:4}}/>
        <Line name="Val Acc"   type="monotone" dataKey="val"   stroke={T.green} strokeWidth={2} dot={false} activeDot={{r:4}}/>
      </LineChart>
    </ResponsiveContainer>
    <p style={{ fontSize:"12px", color:"rgba(148,163,184,0.55)", marginTop:"10px", lineHeight:1.5 }}>
      Training converged rapidly — validation accuracy reached 99.39% by epoch 3. Best checkpoint at epoch 16 (99.77% val). Purple dashed lines mark prototype push operations. Early stopping triggered at epoch 28.
    </p>
  </div>
);

const ValLossChart = () => (
  <div style={{ ...darkCardBase, padding:"24px" }}>
    <div style={{ marginBottom:"16px" }}>
      <div style={{ fontSize:"11px", fontWeight:700, color:"rgba(148,163,184,0.7)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Phase 2 · Joint Training</div>
      <div style={{ fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:"#F8FAFC" }}>Validation Loss Curve</div>
    </div>
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={PHASE2} margin={{left:-10,right:20,top:4,bottom:4}}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)"/>
        <XAxis dataKey="ep" tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false}/>
        <YAxis domain={[0.73,0.79]} tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false}/>
        <Tooltip content={<DarkTooltip/>}/>
        {PROTO_PUSH_EPS.map(ep=><ReferenceLine key={ep} x={ep} stroke="rgba(167,139,250,0.3)" strokeDasharray="3 3"/>)}
        <ReferenceLine x={8}  stroke={T.amber} strokeDasharray="0" label={{value:"★ Min 0.741",fill:T.amber,fontSize:10,position:"top"}}/>
        <Line name="Val Loss" type="monotone" dataKey="loss" stroke={T.amber} strokeWidth={2} dot={false} activeDot={{r:4}}/>
      </LineChart>
    </ResponsiveContainer>
    <p style={{ fontSize:"12px", color:"rgba(148,163,184,0.55)", marginTop:"10px", lineHeight:1.5 }}>
      Loss decreased steeply in first 3 epochs then stabilised around 0.741–0.745. Minimum at epoch 8 (0.7408). The prototype loss components (cluster + separation) prevent the model from lowering cross-entropy loss at the expense of interpretability.
    </p>
  </div>
);

const GenGapChart = () => (
  <div style={{ ...darkCardBase, padding:"24px" }}>
    <div style={{ marginBottom:"16px" }}>
      <div style={{ fontSize:"11px", fontWeight:700, color:"rgba(148,163,184,0.7)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Overfitting Monitor</div>
      <div style={{ fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:"#F8FAFC" }}>Generalisation Gap (Train − Val Acc)</div>
    </div>
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={PHASE2} margin={{left:-10,right:20,top:4,bottom:4}}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)"/>
        <XAxis dataKey="ep" tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false}/>
        <YAxis tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false} tickFormatter={v=>`${v}%`}/>
        <Tooltip content={<DarkTooltip unit="%"/>}/>
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeWidth={1}/>
        <Area type="monotone" dataKey="gap" fill="rgba(248,113,113,0.15)" stroke={T.red} strokeWidth={2}
          name="Gen Gap" dot={false}/>
      </ComposedChart>
    </ResponsiveContainer>
    <p style={{ fontSize:"12px", color:"rgba(148,163,184,0.55)", marginTop:"10px", lineHeight:1.5 }}>
      Negative gap in early epochs (val outperforming train) indicates effective regularisation from MixUp/CutMix augmentation. The gap stabilises near +0.25–0.30% after epoch 10, confirming robust generalisation without significant overfitting.
    </p>
  </div>
);

const ProtoPushChart = () => (
  <div style={{ ...darkCardBase, padding:"24px" }}>
    <div style={{ marginBottom:"16px" }}>
      <div style={{ fontSize:"11px", fontWeight:700, color:"rgba(148,163,184,0.7)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Prototype Push Analysis</div>
      <div style={{ fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:"#F8FAFC" }}>ΔVal Accuracy Per Push Operation</div>
    </div>
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={PROTO_PUSH_IMPACT} margin={{left:-10,right:20,top:4,bottom:4}}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)"/>
        <XAxis dataKey="ep" tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false}/>
        <YAxis tick={{fontSize:11,fill:"rgba(148,163,184,0.6)"}} axisLine={false} tickLine={false} tickFormatter={v=>`${v>0?"+":""}${v.toFixed(2)}%`}/>
        <Tooltip formatter={(v,n,p)=>[`${v>0?"+":""}${v.toFixed(4)}%`,n]}/>
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)"/>
        <Bar dataKey="delta" name="ΔVal Acc" radius={[3,3,0,0]}>
          {PROTO_PUSH_IMPACT.map((d,i) => <Cell key={i} fill={d.pos ? T.green : T.red}/>)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    <p style={{ fontSize:"12px", color:"rgba(148,163,184,0.55)", marginTop:"10px", lineHeight:1.5 }}>
      Prototype push operations replace learned vectors with real training patches. Early pushes (ep 2, 8) yield largest gains (+0.76%, +0.31%), confirming that grounding prototypes in genuine image patches is critical. Later pushes show smaller but consistent positive impact.
    </p>
  </div>
);

const TrainingTimeline = () => {
  const phases = [
    { label:"Phase 1", sub:"Warmup", eps:"10 eps", desc:"SRM + FC only, lr=10⁻³. Freeze backbone.", pct:21, color:T.blue },
    { label:"Phase 2", sub:"Joint Training", eps:"28 eps", desc:"All layers, backbone lr=2×10⁻⁵. Prototype pushes every 2 epochs.", pct:58, color:T.green },
    { label:"Phase 4", sub:"FC Fine-tune", eps:"10 eps", desc:"FC only + L1 sparsity, lr=10⁻⁵.", pct:21, color:T.red },
  ];
  return (
    <div style={{ ...darkCardBase, padding:"24px" }}>
      <div style={{ marginBottom:"20px" }}>
        <div style={{ fontSize:"11px", fontWeight:700, color:"rgba(148,163,184,0.7)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"4px" }}>Training Protocol</div>
        <div style={{ fontSize:"18px", fontFamily:"'DM Serif Display',serif", color:"#F8FAFC" }}>3-Phase Training Timeline</div>
      </div>
      {/* Timeline bar */}
      <div style={{ display:"flex", borderRadius:"8px", overflow:"hidden", marginBottom:"20px", height:"48px" }}>
        {phases.map((p, i) => (
          <div key={i} style={{ flex:`0 0 ${p.pct}%`, background:`${p.color}22`, borderRight: i<2 ? `1px solid ${T.darkBorder}` : "none",
            display:"flex", alignItems:"center", justifyContent:"center", gap:"6px" }}>
            <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:p.color }}/>
            <span style={{ fontSize:"12px", fontWeight:700, color:p.color }}>{p.label}</span>
          </div>
        ))}
      </div>
      {/* Phase details */}
      <div style={{ display:"flex", flexWrap:"wrap", gap:"12px" }}>
        {phases.map((p, i) => (
          <div key={i} style={{ flex:"1 1 180px", background:`${p.color}0F`, border:`1px solid ${p.color}30`, borderRadius:"10px", padding:"14px" }}>
            <div style={{ display:"flex", alignItems:"center", gap:"6px", marginBottom:"6px" }}>
              <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:p.color }}/>
              <span style={{ fontSize:"12px", fontWeight:700, color:p.color }}>{p.label}: {p.sub}</span>
            </div>
            <div style={{ fontSize:"13px", color:"rgba(248,250,252,0.85)", marginBottom:"4px" }}>{p.eps}</div>
            <div style={{ fontSize:"11px", color:"rgba(148,163,184,0.7)", lineHeight:1.5 }}>{p.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop:"16px", background:"rgba(74,222,128,0.08)", border:"1px solid rgba(74,222,128,0.2)", borderRadius:"10px", padding:"12px 16px" }}>
        <span style={{ fontSize:"13px", color:"#4ADE80", fontWeight:700 }}>Final Test Accuracy: 99.65%</span>
        <span style={{ fontSize:"12px", color:"rgba(148,163,184,0.7)", marginLeft:"12px" }}>Best Val Acc: 99.77% @ Epoch 16</span>
      </div>
    </div>
  );
};

// ─── Architecture Stage Card ──────────────────────────────────────────────────
const ArchStage = ({ stage, expanded, onToggle, isLast }) => (
  <div>
    <div onClick={onToggle} style={{ cursor:"pointer", ...cardBase, padding:"18px 22px", marginBottom:expanded?"0":"0",
      borderBottomLeftRadius: expanded?"0":"20px", borderBottomRightRadius: expanded?"0":"20px",
      transition:"box-shadow 0.2s" }} className="premium-card">
      <div style={{ display:"flex", alignItems:"center", gap:"14px" }}>
        <div style={{ width:"44px", height:"44px", borderRadius:"12px", background:`${stage.color}18`,
          border:`1px solid ${stage.color}40`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <div style={{ fontSize:"18px" }}>{["🖼️","🧠","🎯","🔬","🧩","📊","✅"][ARCH_STAGES.indexOf(stage)]}</div>
        </div>
        <div style={{ flex:1 }}>
          <div style={{ display:"flex", flexWrap:"wrap", gap:"8px", alignItems:"center" }}>
            <span style={{ fontFamily:"'DM Serif Display',serif", fontSize:"17px", color:T.forest }}>{stage.title}</span>
            <span style={{ background:`${stage.color}18`, color:stage.color, border:`1px solid ${stage.color}35`, fontSize:"10px", fontWeight:700,
              padding:"2px 8px", borderRadius:"6px", letterSpacing:"0.06em" }}>
              {stage.label}
            </span>
          </div>
          <div style={{ fontSize:"12px", color:T.muted, marginTop:"2px" }}>Output: <code style={{ fontSize:"11px", background:T.cream, padding:"1px 5px", borderRadius:"4px", color:T.forestMd }}>{stage.shape}</code></div>
        </div>
        <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:"4px", marginRight:"8px" }}>
          <span style={{ fontSize:"11px", color:T.dim }}>Parameters</span>
          <span style={{ fontSize:"13px", fontWeight:700, color:T.forest }}>{stage.params}</span>
        </div>
        <ChevronIcon open={expanded}/>
      </div>
    </div>
    {expanded && (
      <div style={{ borderTop:`1px solid ${T.border}`, background:T.cream, borderRadius:"0 0 20px 20px",
        padding:"20px 24px", marginBottom:"0", border:`1px solid rgba(0,0,0,0.05)`, borderTop:`1px solid ${T.border}` }}>
        <p style={{ fontSize:"14px", color:T.ink, lineHeight:1.7, marginBottom:"12px" }}>{stage.desc}</p>
        <div style={{ background:T.paper, border:`1px solid ${T.border}`, borderRadius:"8px", padding:"10px 16px" }}>
          <span style={{ fontSize:"10px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase" }}>Formula</span>
          <div style={{ fontFamily:"monospace", fontSize:"13px", color:T.forestMd, marginTop:"4px" }}>{stage.formula}</div>
        </div>
      </div>
    )}
    {!isLast && (
      <div style={{ display:"flex", justifyContent:"center", alignItems:"center", height:"28px", position:"relative" }}>
        <div style={{ width:"2px", height:"100%", background:T.sagePale }}/>
        <div style={{ position:"absolute", width:"8px", height:"8px", borderRadius:"50%", background:T.sageLt, border:`2px solid ${T.paper}` }}/>
      </div>
    )}
  </div>
);

// ─── Research Metrics Tab ─────────────────────────────────────────────────────
const ResearchMetricsTab = () => (
  <div style={{ background:T.dark, minHeight:"calc(100vh - 64px)", padding:"60px 24px 80px" }}>
    <div style={{ maxWidth:"1200px", margin:"0 auto" }}>
      <SectionTag dark tag="POWER-ProtoPNet · Phase 2 Results" title="Training Research Dashboard" subtitle="Live metrics from parsed training logs. 28 joint-training epochs, 14 prototype push operations, 10 FC fine-tune epochs."/>
      {/* Key metrics */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(165px,1fr))", gap:"12px", marginBottom:"36px" }}>
        <MetricCard label="Best Val Acc" value="99.77%" sub="Epoch 16" color={T.green}/>
        <MetricCard label="Min Val Loss" value="0.7408" sub="Epoch 8" color={T.amber}/>
        <MetricCard label="Test Accuracy" value="99.65%" sub="Final evaluation" color={T.green}/>
        <MetricCard label="Best Epoch" value="16" sub="Early stop @ 28" color={T.blue}/>
        <MetricCard label="Proto Pushes" value="14" sub="Every 2 epochs" color={T.purple}/>
        <MetricCard label="Total Epochs" value="48" sub="P1+P2+P4" color={T.red}/>
      </div>
      {/* Charts grid */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(480px,1fr))", gap:"20px", marginBottom:"20px" }}>
        <TrainingCurveChart/>
        <ValLossChart/>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(480px,1fr))", gap:"20px", marginBottom:"20px" }}>
        <GenGapChart/>
        <ProtoPushChart/>
      </div>
      <TrainingTimeline/>
      {/* Reference images note */}
      <div style={{ marginTop:"20px", ...darkCardBase, padding:"16px 20px" }}>
        <p style={{ fontSize:"12px", color:"rgba(148,163,184,0.6)", lineHeight:1.6 }}>
          <strong style={{ color:"rgba(148,163,184,0.9)" }}>Static Training Graphs:</strong> High-resolution matplotlib figures (00_dashboard.png through 07_training_timeline.png) should be placed in <code style={{ background:"rgba(255,255,255,0.06)", padding:"1px 5px", borderRadius:"4px", color:T.amber }}>public/training/</code> for reference. Charts above are rendered from raw JSON training logs.
        </p>
      </div>
    </div>
  </div>
);

// ─── Architecture Explorer Tab ────────────────────────────────────────────────
const ArchitectureTab = () => {
  const [expanded, setExpanded] = useState(null);
  const toggle = id => setExpanded(p => p===id ? null : id);
  return (
    <div style={{ maxWidth:"800px", margin:"0 auto", padding:"60px 24px 80px" }}>
      <SectionTag tag="Model Architecture" title="POWER-ProtoPNet Pipeline" subtitle="Click any stage to expand its technical description, input/output shapes, and governing equations."/>
      <div style={{ marginBottom:"8px", padding:"12px 16px", background:T.sagePale, borderRadius:"12px", display:"flex", alignItems:"center", gap:"8px", marginBottom:"28px" }}>
        <LeafMark size={20} fill={T.sageLt} stroke={T.forestMd}/>
        <span style={{ fontSize:"13px", color:T.forestMd }}>
          Architecture: ConvNeXt-Tiny + SE Attention + SRM + Prototype Layer (190 protos) + FC Head → 38 classes
        </span>
      </div>
      {ARCH_STAGES.map((stage, i) => (
        <ArchStage key={stage.id} stage={stage} expanded={expanded===stage.id} onToggle={()=>toggle(stage.id)} isLast={i===ARCH_STAGES.length-1}/>
      ))}
      {/* Model stats */}
      <div style={{ marginTop:"32px", display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(140px,1fr))", gap:"12px" }}>
        {[
          ["Backbone","ConvNeXt-Tiny"],["Input Size","224×224"],["Classes","38"],
          ["Prototypes","190 (5/class)"],["Proto Dim","256"],["Test Acc","99.65%"],
        ].map(([k,v]) => (
          <div key={k} style={{ ...cardBase, padding:"14px 16px", textAlign:"center" }}>
            <div style={{ fontSize:"10px", fontWeight:700, color:T.muted, textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:"4px" }}>{k}</div>
            <div style={{ fontFamily:"'DM Serif Display',serif", fontSize:"16px", color:T.forest }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Research Paper Tab ───────────────────────────────────────────────────────
const ResearchPaperTab = () => {
  const Section = ({ tag, title, children }) => (
    <div style={{ ...cardBase, padding:"32px", marginBottom:"20px" }}>
      <div style={{ width:"36px", height:"2px", background:T.sageLt, marginBottom:"16px", borderRadius:"2px" }}/>
      <div style={{ fontSize:"10px", fontWeight:700, color:T.sage, letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:"8px" }}>{tag}</div>
      <h3 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"22px", color:T.forest, marginBottom:"16px" }}>{title}</h3>
      {children}
    </div>
  );
  const P = ({ children }) => <p style={{ fontSize:"14px", color:T.ink, lineHeight:1.75, marginBottom:"12px" }}>{children}</p>;
  const Li = ({ children }) => <div style={{ display:"flex", alignItems:"flex-start", gap:"8px", marginBottom:"8px" }}><div style={{ width:"5px", height:"5px", borderRadius:"50%", background:T.sageLt, marginTop:"7px", flexShrink:0 }}/><span style={{ fontSize:"14px", color:T.ink, lineHeight:1.6 }}>{children}</span></div>;
  return (
    <div style={{ maxWidth:"900px", margin:"0 auto", padding:"60px 24px 80px" }}>
      <SectionTag tag="Research Paper" title="POWER-ProtoPNet" subtitle="Prototype-based Explainable Deep Learning for Fine-grained Plant Disease Recognition · PlantWild v3"/>
      <Section tag="01 · Abstract" title="A Prototype-based Interpretable CNN for Plant Disease Recognition">
        <P>We present POWER-ProtoPNet (Prototype-based Weighted Explainable Recognition), an interpretable plant disease diagnostic system achieving 99.65% top-1 accuracy on the PlantWild v3 benchmark across 38 disease categories. Unlike black-box classifiers, POWER-ProtoPNet provides a fully transparent reasoning chain: each classification is explicitly grounded in prototype patches extracted from training images.</P>
        <P>Our architecture integrates a ConvNeXt-Tiny backbone with Squeeze-and-Excitation attention, a Spatial Refinement Module (SRM) with depthwise-separable convolutions, and a cosine-similarity prototype layer. A 3-phase training protocol (warmup, joint training with prototype pushing every 2 epochs, and FC sparsity fine-tuning) achieves superior accuracy while maintaining full explainability.</P>
        <P>Key contribution: the first demonstration that a prototype-based interpretable architecture can match or exceed state-of-the-art black-box performance on a large-scale plant disease dataset, with formal GradCAM++ and SHAP explainability validation.</P>
      </Section>
      <Section tag="02 · Methodology" title="Architecture & Training Protocol">
        <P><strong>Backbone.</strong> ConvNeXt-Tiny pretrained on ImageNet-1K provides hierarchical 768-channel feature maps at 7×7 spatial resolution. The large 7×7 depthwise kernels and Layer Norm layers make it superior to ResNet families for disease pattern localisation.</P>
        <P><strong>SE Attention.</strong> Channel-wise recalibration (reduction ratio r=16) amplifies pathologically relevant channels. For lesion images where disease occupies ≤15% of the frame, this suppression of irrelevant background channels significantly improves discriminability.</P>
        <P><strong>SRM.</strong> Depthwise-separable convolution pipeline reduces 768→256 channels at 8× parameter efficiency. Spatial dropout (p=0.3) prevents prototype co-adaptation.</P>
        <P><strong>Prototype Layer.</strong> 190 prototype vectors (5 per class × 38 classes), each 256-dimensional. Cosine similarity over all spatial patches with max-pooling ensures rotation/translation invariance. Prototype push operations every 2 epochs ground each vector in a real training patch.</P>
        <P><strong>Training.</strong> 3-phase protocol: Phase 1 (10 ep warmup, SRM+FC only), Phase 2 (28 ep joint training, all modules, full composite loss), Phase 4 (10 ep FC sparsity fine-tuning with L1 λ=10⁻⁵). MixUp (α=0.2) and CutMix (α=1.0) augmentation applied 50/50 throughout Phase 2.</P>
      </Section>
      <Section tag="03 · Results" title="Performance on PlantWild v3">
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(160px,1fr))", gap:"12px", marginBottom:"20px" }}>
          {[
            ["Test Accuracy","99.65%","PlantWild v3"],["Best Val Acc","99.77%","Epoch 16"],
            ["Min Val Loss","0.7408","Epoch 8"],["Gen Gap","≤0.30%","Epoch 28"],
          ].map(([k,v,sub]) => (
            <div key={k} style={{ background:T.cream, borderRadius:"12px", padding:"16px", textAlign:"center" }}>
              <div style={{ fontSize:"10px", fontWeight:700, color:T.muted, textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:"4px" }}>{k}</div>
              <div style={{ fontFamily:"'DM Serif Display',serif", fontSize:"24px", color:T.forest }}>{v}</div>
              <div style={{ fontSize:"11px", color:T.dim }}>{sub}</div>
            </div>
          ))}
        </div>
        <P>The model achieved 99.65% test accuracy with a generalisation gap of ≤0.30% at early stopping (epoch 28), indicating no meaningful overfitting. The best validation accuracy of 99.77% at epoch 16 demonstrates that the prototype push operations actively improve validation performance, with early pushes (epochs 2, 8) providing the largest gains (+0.76%, +0.31% respectively).</P>
      </Section>
      <Section tag="04 · Explainability Analysis" title="Prototype Activations & XAI Validation">
        <P><strong>Prototype Transparency.</strong> Each prediction cites the top-N prototype patches from training data. This provides instance-level explanations with semantic grounding: "this leaf region resembles Prototype #27 (Cherry — Powdery Mildew, similarity 0.924)" rather than abstract feature attribution.</P>
        <P><strong>GradCAM++ Validation.</strong> Gradient-weighted class activation maps confirm that the model attends to genuine pathological regions (lesion centres, sporulation zones) rather than image artefacts, validating the prototype-based explanation qualitatively.</P>
        <P><strong>SHAP Analysis.</strong> KernelSHAP on the prototype similarity vector identifies disease-specific features as strongly positive contributors, while background vegetation and healthy tissue show negative SHAP values. This formally establishes that the classification relies on pathologically meaningful features.</P>
        <P><strong>Bias Mitigation.</strong> SHAP spatial analysis revealed mild bias toward background soil texture in early training. Addressed by CutMix augmentation which randomly replaces background regions with patches from other classes, forcing the model to focus on lesion-intrinsic features.</P>
      </Section>
      <Section tag="05 · Key Contributions" title="Novel Contributions to Explainable Plant Pathology AI">
        {["First prototype-based XAI architecture achieving >99.5% accuracy on a 38-class plant disease dataset.",
          "SE Attention + SRM combination providing 8× parameter efficiency with minimal accuracy trade-off.",
          "Empirical validation that prototype pushing every 2 epochs provides sustained accuracy improvements throughout training.",
          "Formal SHAP analysis revealing background texture bias and its mitigation through targeted augmentation.",
          "A complete 3-phase training protocol with early stopping, composite prototype loss, and FC sparsity fine-tuning.",
          "Open-source platform integrating inference, GradCAM++, SHAP, and prototype visualisation in a unified research dashboard."
        ].map((t, i) => <Li key={i}>{t}</Li>)}
      </Section>
    </div>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function PlantWildApp() {
  const [activeTab, setActiveTab]   = useState("diagnose");
  const [phase, setPhase]           = useState("upload");
  const [imageUrl, setImageUrl]     = useState(null);
  const [imageFile, setImageFile]   = useState(null);
  const [result, setResult]         = useState(null);
  const [dragging, setDragging]     = useState(false);
  const [showInfo, setShowInfo]     = useState(false);
  const [error, setError]           = useState(null);
  const fileRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file); setImageUrl(URL.createObjectURL(file)); setPhase("preview"); setError(null);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const analyze = async () => {
    setPhase("loading"); setError(null);
    try {
      const data = await runPredict(imageFile);
      setResult(data); setPhase("results");
    } catch (err) {
      setError(`Analysis failed: ${err.message}. Ensure the FastAPI backend is running at ${API_BASE}.`);
      setPhase("preview");
    }
  };

  const reset = () => { setPhase("upload"); setImageUrl(null); setImageFile(null); setResult(null); setError(null); };

  // ── Diagnose tab bar chart data
  const chartData = result?.prediction?.top_k?.map(k => ({
    name: k.name.replace("— ", "").substring(0, 22),
    value: parseFloat((k.prob * 100).toFixed(3)),
  })) ?? [];
  const BAR_COLORS = [T.forest, T.forestMd, T.sage, T.sageLt, "#B4CCAA"];
  const maxSim = result ? Math.max(...result.prototypes.map(p => p.similarity)) : 1;
  const diseaseInfo = result?.disease_info ?? FALLBACK_DISEASE;

  const TABS = [
    { id:"diagnose",    label:"Diagnose" },
    { id:"research",    label:"Research Metrics" },
    { id:"architecture",label:"Architecture" },
    { id:"paper",       label:"Research Paper" },
  ];

  const isResearch = activeTab === "research";
  const bgStyle = isResearch
    ? { background:T.dark }
    : { background:`radial-gradient(ellipse at 50% -10%, #FFFFFF 0%, ${T.cream} 60%)` };

  return (
    <div style={{ minHeight:"100vh", ...bgStyle, fontFamily:"'DM Sans',sans-serif", color:T.ink, transition:"background 0.4s" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        @keyframes scanLine { 0%{top:-4px} 100%{top:100%} }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.55;transform:scale(.9)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        .fade-up{animation:fadeUp .6s cubic-bezier(.16,1,.3,1) both}
        .fade-up-d1{animation-delay:.1s} .fade-up-d2{animation-delay:.2s} .fade-up-d3{animation-delay:.3s}
        .drop-zone{border:1px dashed rgba(90,122,56,.35);background:linear-gradient(180deg,#FFFFFF 0%,#FDFCF8 100%);box-shadow:0 16px 32px -10px rgba(26,54,9,.04);transition:all .4s cubic-bezier(.16,1,.3,1);cursor:pointer}
        .drop-zone:hover,.drop-zone.active{border:1px solid rgba(90,122,56,.7);box-shadow:0 24px 48px -12px rgba(90,122,56,.15);transform:translateY(-3px)}
        .btn-primary{background:#1A3609;color:#fff;border:none;padding:14px 36px;border-radius:12px;font-size:15px;font-weight:600;font-family:'DM Sans',sans-serif;cursor:pointer;letter-spacing:.02em;box-shadow:0 8px 20px -6px rgba(26,54,9,.4);transition:all .2s ease}
        .btn-primary:hover{background:#264F11;transform:translateY(-1px)}
        .btn-ghost{background:#FFFFFF;color:#5A7A38;border:1px solid #DCE9CC;padding:14px 28px;border-radius:12px;font-size:15px;font-weight:600;font-family:'DM Sans',sans-serif;cursor:pointer;transition:all .2s ease}
        .btn-ghost:hover{background:#F4F9EF;border-color:#8BAA60}
        .premium-card{transition:transform .3s cubic-bezier(.16,1,.3,1),box-shadow .3s ease}
        .premium-card:hover{transform:translateY(-5px);box-shadow:0 22px 44px -12px rgba(30,61,10,.15)!important}
        .nav-tab{background:transparent;border:none;border-bottom:2px solid transparent;padding:0 4px;padding-bottom:6px;color:rgba(255,255,255,.65);font-size:13px;font-weight:500;font-family:'DM Sans',sans-serif;cursor:pointer;transition:all .2s;white-space:nowrap}
        .nav-tab:hover{color:#fff}
        .nav-tab.active{color:#8BAA60;border-bottom-color:#8BAA60}
        code{font-family:monospace}
      `}</style>

      {/* ══════════════ HEADER ══════════════ */}
      <header style={{ background:"rgba(26,54,9,0.88)", backdropFilter:"blur(16px)", WebkitBackdropFilter:"blur(16px)", position:"sticky", top:0, zIndex:100, boxShadow:"0 4px 30px rgba(0,0,0,0.12)", borderBottom:"1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ maxWidth:"1200px", margin:"0 auto", padding:"0 24px", height:"64px", display:"flex", alignItems:"center", gap:"24px" }}>
          <div style={{ display:"flex", alignItems:"center", gap:"10px", cursor:"pointer", flexShrink:0 }} onClick={()=>{reset();setActiveTab("diagnose")}}>
            <LeafMark size={28} fill={T.sageLt} stroke="rgba(220,233,204,0.9)"/>
            <span style={{ fontFamily:"'DM Serif Display',serif", fontSize:"20px", color:"#FFF", letterSpacing:".02em" }}>
              PlantWild <em style={{ color:T.sageLt, fontStyle:"italic", fontWeight:400 }}>AI</em>
            </span>
          </div>
          {/* Tab nav */}
          <nav style={{ display:"flex", alignItems:"center", gap:"20px", flex:1 }}>
            {TABS.map(t => (
              <button key={t.id} className={`nav-tab${activeTab===t.id?" active":""}`}
                onClick={()=>{ if(t.id==="diagnose"&&activeTab!=="diagnose") reset(); setActiveTab(t.id); }}>
                {t.label}
              </button>
            ))}
          </nav>
          {/* Right actions */}
          <div style={{ display:"flex", alignItems:"center", gap:"12px", flexShrink:0 }}>
            {DEMO_MODE && (
              <span style={{ fontSize:"11px", fontWeight:700, letterSpacing:"0.08em", background:"rgba(74,222,128,0.15)", color:T.green, padding:"4px 10px", borderRadius:"6px", border:"1px solid rgba(74,222,128,0.25)" }}>
                DEMO
              </span>
            )}
            {activeTab==="diagnose" && result && (
              <button onClick={reset} style={{ background:"rgba(255,255,255,0.1)", border:"1px solid rgba(255,255,255,0.18)", color:"#fff", padding:"7px 16px", borderRadius:"8px", cursor:"pointer", fontSize:"13px", fontWeight:600, fontFamily:"'DM Sans',sans-serif" }}>
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ══════════════ CONTENT ══════════════ */}
      {activeTab === "research"     && <ResearchMetricsTab/>}
      {activeTab === "architecture" && <ArchitectureTab/>}
      {activeTab === "paper"        && <ResearchPaperTab/>}

      {activeTab === "diagnose" && (
        <main style={{ maxWidth:"1200px", margin:"0 auto", padding:"60px 24px 80px" }}>

          {/* ── ERROR ── */}
          {error && (
            <div style={{ background:"#FEF2F2", border:"1px solid #FECACA", borderRadius:"12px", padding:"14px 18px", marginBottom:"24px", color:"#991B1B", fontSize:"14px", display:"flex", gap:"10px", alignItems:"flex-start" }}>
              <span>⚠️</span><span>{error}</span>
            </div>
          )}

          {/* ── UPLOAD ── */}
          {phase === "upload" && (
            <div className="fade-up" style={{ maxWidth:"580px", margin:"0 auto", textAlign:"center" }}>
              <div style={{ marginBottom:"40px" }}>
                <p style={{ fontSize:"12px", fontWeight:700, color:T.sage, letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:"12px" }}>Intelligent XAI Diagnostics</p>
                <h1 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"clamp(38px,6vw,58px)", color:T.forest, lineHeight:1.05, marginBottom:"16px" }}>
                  Diagnose Your<br/><em style={{ color:T.sage, fontStyle:"italic" }}>Plant Health</em>
                </h1>
                <p style={{ fontSize:"15px", color:T.muted, lineHeight:1.65, maxWidth:"460px", margin:"0 auto" }}>
                  Upload a leaf image. POWER-ProtoPNet will classify the disease, trace the decision back to real prototype patches, and generate GradCAM++ and SHAP explanations.
                </p>
              </div>
              <div className={`drop-zone${dragging?" active":""}`} style={{ borderRadius:"24px", padding:"48px 32px", marginBottom:"16px" }}
                onClick={()=>fileRef.current.click()} onDrop={onDrop}
                onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)}>
                <div style={{ width:"72px", height:"72px", borderRadius:"50%", background:T.sagePale, display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 20px", boxShadow:"inset 0 2px 8px rgba(255,255,255,.5)" }}>
                  <LeafMark size={36} fill="#C0D8A0" stroke={T.sage}/>
                </div>
                <p style={{ fontFamily:"'DM Serif Display',serif", fontSize:"24px", color:T.forest, marginBottom:"8px" }}>Drag & drop image here<br/>or click to browse</p>
                <p style={{ fontSize:"13px", color:T.dim, fontWeight:500 }}>JPG · PNG · WEBP · High Resolution Supported</p>
                <div style={{ marginTop:"28px", paddingTop:"20px", borderTop:`1px solid rgba(0,0,0,.05)` }}>
                  <p style={{ fontSize:"12px", color:T.muted }}>Model: <strong style={{color:T.forest}}>POWER-ProtoPNet</strong> · ConvNeXt-Tiny · 38 classes · 99.65% test accuracy</p>
                </div>
              </div>
              <input ref={fileRef} type="file" accept="image/*" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
              {DEMO_MODE && (
                <p style={{ fontSize:"12px", color:T.dim, marginTop:"8px" }}>
                  Demo mode active — any image will return a sample Cherry Powdery Mildew result. Set <code style={{color:T.sage}}>DEMO_MODE=false</code> and start the FastAPI backend for live inference.
                </p>
              )}
            </div>
          )}

          {/* ── PREVIEW ── */}
          {phase === "preview" && (
            <div className="fade-up" style={{ maxWidth:"520px", margin:"0 auto", textAlign:"center" }}>
              <h2 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"32px", color:T.forest, marginBottom:"24px" }}>Specimen Selected</h2>
              <div className="premium-card" style={{ ...cardBase, overflow:"hidden", marginBottom:"32px" }}>
                <img src={imageUrl} alt="Selected" style={{ width:"100%", maxHeight:"400px", objectFit:"contain", display:"block", background:T.paper }}/>
              </div>
              <div style={{ display:"flex", gap:"12px", justifyContent:"center" }}>
                <button className="btn-ghost" onClick={reset}>Cancel</button>
                <button className="btn-primary" onClick={analyze}>Run POWER-ProtoPNet</button>
              </div>
            </div>
          )}

          {/* ── LOADING ── */}
          {phase === "loading" && (
            <div className="fade-up" style={{ maxWidth:"520px", margin:"0 auto", textAlign:"center" }}>
              <h2 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"32px", color:T.forest, marginBottom:"24px" }}>Extracting Features…</h2>
              <div style={{ position:"relative", borderRadius:"20px", overflow:"hidden", marginBottom:"32px", boxShadow:"0 12px 36px -12px rgba(26,54,9,.12)", background:T.paper, border:`1px solid rgba(0,0,0,.04)` }}>
                <img src={imageUrl} alt="Analyzing" style={{ width:"100%", maxHeight:"400px", objectFit:"contain", display:"block" }}/>
                <div style={{ position:"absolute", inset:0, background:"rgba(26,54,9,0.08)" }}/>
                <div style={{ position:"absolute", left:0, right:0, height:"4px", background:`linear-gradient(90deg,transparent 0%,${T.scan} 40%,#A8CC70 60%,transparent 100%)`, boxShadow:`0 0 16px 6px rgba(139,170,96,.4)`, animation:"scanLine 1.8s linear infinite" }}/>
                <div style={{ position:"absolute", bottom:"20px", left:"50%", transform:"translateX(-50%)", background:"rgba(20,38,8,0.85)", backdropFilter:"blur(12px)", borderRadius:"99px", padding:"10px 20px", display:"flex", alignItems:"center", gap:"10px", whiteSpace:"nowrap" }}>
                  <div style={{ width:"8px", height:"8px", borderRadius:"50%", background:T.sageLt, animation:"pulse 1.2s ease infinite" }}/>
                  <span style={{ color:"#fff", fontSize:"13px", fontWeight:500 }}>Running POWER-ProtoPNet inference…</span>
                </div>
              </div>
              <p style={{ fontSize:"13px", color:T.muted }}>Computing prototype similarities, GradCAM++ maps, and SHAP values…</p>
            </div>
          )}

          {/* ── RESULTS ── */}
          {phase === "results" && result && (
            <div>
              {/* Header */}
              <div className="fade-up" style={{ marginBottom:"36px" }}>
                <p style={{ fontSize:"12px", fontWeight:700, color:T.sage, letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:"8px" }}>Analysis Complete</p>
                <h1 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"clamp(32px,5vw,48px)", color:T.forest }}>Diagnosis Report</h1>
              </div>

              {/* Row 1: Image + Gauge | Bar chart */}
              <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(320px,1fr))", gap:"24px", marginBottom:"24px" }}>
                {/* Left: Image + diagnosis */}
                <div className="fade-up fade-up-d1 premium-card" style={{ ...cardBase, padding:"28px", display:"flex", flexDirection:"column", gap:"24px" }}>
                  <div style={{ borderRadius:"14px", overflow:"hidden", border:`1px solid ${T.border}` }}>
                    <img src={imageUrl} alt="Analyzed" style={{ width:"100%", maxHeight:"220px", objectFit:"cover", display:"block" }}/>
                  </div>
                  <div>
                    <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"8px" }}>Primary Assessment</div>
                    <div style={{ fontFamily:"'DM Serif Display',serif", fontSize:"clamp(22px,3vw,30px)", color:T.forest, lineHeight:1.1 }}>
                      {result.prediction.class_name}
                    </div>
                    <div style={{ display:"flex", alignItems:"center", gap:"6px", marginTop:"8px" }}>
                      <div style={{ width:"6px", height:"6px", borderRadius:"50%", background: (SEV_STYLES[diseaseInfo.severity]||SEV_STYLES["Moderate"]).dot }}/>
                      <span style={{ fontSize:"12px", color:T.muted, fontWeight:500 }}>{diseaseInfo.scientific}</span>
                    </div>
                  </div>
                  <div style={{ borderTop:`1px solid ${T.border}`, paddingTop:"20px", display:"flex", justifyContent:"center" }}>
                    <ConfidenceGauge value={result.prediction.confidence}/>
                  </div>
                </div>

                {/* Right: Top-K bar chart */}
                <div className="fade-up fade-up-d2" style={{ display:"flex", flexDirection:"column", gap:"24px" }}>
                  <div className="premium-card" style={{ ...cardBase, padding:"28px", flex:1 }}>
                    <div style={{ marginBottom:"20px" }}>
                      <h3 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"22px", color:T.forest, marginBottom:"4px" }}>Probability Distribution</h3>
                      <p style={{ fontSize:"13px", color:T.muted }}>Top-5 class predictions ranked by model confidence.</p>
                    </div>
                    <div style={{ height:"220px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} layout="vertical" margin={{left:0,right:52,top:0,bottom:0}}>
                          <XAxis type="number" domain={[0,100]} hide/>
                          <YAxis type="category" dataKey="name" width={155} tick={{fontSize:11,fontFamily:"'DM Sans',sans-serif",fill:T.muted}} axisLine={false} tickLine={false}/>
                          <Tooltip content={<LightTooltip/>} cursor={{fill:T.cream}}/>
                          <Bar dataKey="value" radius={[0,6,6,0]} label={{position:"right",fontSize:11,fill:T.muted,formatter:v=>`${v.toFixed(2)}%`}}>
                            {chartData.map((_,i) => <Cell key={i} fill={BAR_COLORS[i]}/>)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              {/* Row 2: Disease info */}
              <div className="fade-up fade-up-d2" style={{ marginBottom:"24px" }}>
                <DiseaseInfoPanel info={diseaseInfo}/>
              </div>

              {/* Row 3: Prototype activations */}
              <div className="fade-up premium-card" style={{ ...cardBase, padding:"32px", marginBottom:"24px" }}>
                <div style={{ marginBottom:"28px", maxWidth:"640px" }}>
                  <div style={{ fontSize:"11px", fontWeight:700, color:T.sage, letterSpacing:"0.12em", textTransform:"uppercase", marginBottom:"8px" }}>Prototype Explainability</div>
                  <h2 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"26px", color:T.forest, marginBottom:"8px" }}>Why did the model predict this?</h2>
                  <p style={{ fontSize:"14px", color:T.muted, lineHeight:1.6 }}>
                    Each classification is grounded in real training image patches (prototypes). High cosine similarity scores indicate that a region of the uploaded image closely resembles the prototype's visual pattern.
                  </p>
                </div>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(155px,1fr))", gap:"16px" }}>
                  {result.prototypes.map(p => <PrototypeCard key={p.id} proto={p} maxSim={maxSim}/>)}
                </div>
                <div style={{ marginTop:"16px", padding:"12px 16px", background:T.cream, borderRadius:"10px", fontSize:"13px", color:T.muted }}>
                  <strong style={{color:T.forest}}>Prototype IDs</strong> are globally unique indices into the 190-prototype bank (5 per class × 38 classes). A high similarity score (≥0.85) indicates a near-identical visual pattern to the prototype's source training patch.
                </div>
              </div>

              {/* Row 4: GradCAM */}
              <div className="fade-up" style={{ marginBottom:"24px" }}>
                <GradCAMPanel imageUrl={imageUrl} gradcamUrl={result.gradcam_url}/>
              </div>

              {/* Row 5: SHAP */}
              <div className="fade-up" style={{ marginBottom:"24px" }}>
                <SHAPPanel shapUrl={result.shap_url}/>
              </div>

              {/* Model explanation footer */}
              <div className="fade-up premium-card" style={{ ...cardBase, padding:"24px", background:T.sagePale }}>
                <div style={{ fontSize:"11px", fontWeight:700, color:T.forestMd, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"8px" }}>Model Confidence Explanation</div>
                <p style={{ fontSize:"14px", color:T.forestMd, lineHeight:1.7 }}>
                  {diseaseInfo.scientific ? `The model matched the uploaded specimen against prototype #${result.prototypes[0]?.id} (similarity ${result.prototypes[0]?.similarity.toFixed(3)}) and ${result.prototypes.length-1} additional prototypes from class "${result.prediction.class_name}" with an aggregate cosine similarity score confirming the ${(result.prediction.confidence*100).toFixed(2)}% confidence prediction.` : "Prototype activations confirmed consistent visual patterns matching the predicted disease class."}
                </p>
              </div>
            </div>
          )}
        </main>
      )}

      {/* ══════════════ FOOTER ══════════════ */}
      <footer style={{ background:T.forest, borderTop:"1px solid rgba(255,255,255,0.05)", marginTop:"auto" }}>
        <div style={{ maxWidth:"1200px", margin:"0 auto", padding:"48px 24px 32px", display:"flex", flexWrap:"wrap", justifyContent:"space-between", gap:"40px" }}>
          <div style={{ flex:"1 1 300px" }}>
            <div style={{ display:"flex", alignItems:"center", gap:"10px", marginBottom:"14px" }}>
              <LeafMark size={28} fill={T.sageLt} stroke="rgba(220,233,204,0.8)"/>
              <span style={{ fontFamily:"'DM Serif Display',serif", fontSize:"20px", color:"#F6F2EA" }}>PlantWild <em style={{color:T.sageLt,fontStyle:"italic"}}>AI</em></span>
            </div>
            <p style={{ fontSize:"13px", color:"rgba(255,255,255,0.55)", lineHeight:1.7, maxWidth:"320px" }}>
              Transparent, prototype-based explainable AI for plant disease diagnostics. Powered by POWER-ProtoPNet — 99.65% test accuracy on PlantWild v3.
            </p>
          </div>
          <div style={{ flex:"0 1 200px" }}>
            <div style={{ fontSize:"11px", fontWeight:700, color:T.sageLt, letterSpacing:"0.12em", textTransform:"uppercase", marginBottom:"14px" }}>Research & Development</div>
            {["Milan Kundu","Nilanjan Pan"].map((name, i) => (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:"10px", marginBottom:"10px" }}>
                <div style={{ width:"24px", height:"24px", borderRadius:"50%", background:T.forestMd, border:"1px solid rgba(139,170,96,0.3)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:"9px", fontWeight:700, color:T.sageLt }}>
                  {name.split(" ").map(n=>n[0]).join("")}
                </div>
                <span style={{ fontSize:"13px", color:"rgba(255,255,255,0.75)" }}>{name}</span>
              </div>
            ))}
          </div>
          <div style={{ flex:"0 1 220px" }}>
            <div style={{ fontSize:"11px", fontWeight:700, color:T.sageLt, letterSpacing:"0.12em", textTransform:"uppercase", marginBottom:"14px" }}>Technical Stack</div>
            {["POWER-ProtoPNet · ConvNeXt-Tiny","SE Attention + SRM","190 Prototypes · 38 Classes","FastAPI · PyTorch · SHAP","GradCAM++ · KernelSHAP"].map((s,i)=>(
              <div key={i} style={{ fontSize:"12px", color:"rgba(255,255,255,0.5)", marginBottom:"6px", display:"flex", alignItems:"center", gap:"6px" }}>
                <div style={{ width:"3px", height:"3px", borderRadius:"50%", background:T.sageLt }}/>
                {s}
              </div>
            ))}
          </div>
        </div>
        <div style={{ borderTop:"1px solid rgba(255,255,255,0.06)", padding:"18px 24px", display:"flex", alignItems:"center", justifyContent:"center" }}>
          <span style={{ fontSize:"12px", color:"rgba(255,255,255,0.35)" }}>© {new Date().getFullYear()} PlantWild AI · POWER-ProtoPNet Research Platform · All Rights Reserved</span>
        </div>
      </footer>
    </div>
  );
}
