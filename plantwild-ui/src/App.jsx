import { useState, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer,
} from "recharts";

// ─── Mock Fetch (replace with real endpoint later) ────────────────────────────
const API_URL = "http://localhost:5000/predict";

const mockFetch = (_file) =>
  new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          predicted_class: "cherry powdery mildew",
          predicted_index: 24,
          confidence: 0.985432,
          top_k_classes: [
            "cherry powdery mildew",
            "grape leaf spot",
            "strawberry anthracnose",
            "apple scab",
            "peach bacterial spot",
          ],
          top_k_probs: [0.985432, 0.005123, 0.004211, 0.0021, 0.001034],
          prototype_scores: [1.45, 1.22, 0.89, 0.55, 0.41, 0.3, 0.12],
        }),
      2000
    )
  );

// ─── Utilities ────────────────────────────────────────────────────────────────
const toTitleCase = (str) =>
  str.replace(/\b\w/g, (c) => c.toUpperCase());

const confidenceMeta = (pct) => {
  if (pct >= 90) return { color: "#1A5E0A", track: "#22863A", bg: "#E6F4E0", label: "High Confidence" };
  if (pct >= 70) return { color: "#92400E", track: "#D97706", bg: "#FEF3C7", label: "Moderate Confidence" };
  return { color: "#991B1B", track: "#DC2626", bg: "#FEE2E2", label: "Low Confidence" };
};

// ─── Inline styles / token system ─────────────────────────────────────────────
const TOKEN = {
  cream:    "#F9F8F5",
  paper:    "#FFFFFF",
  forest:   "#1A3609",
  forestMd: "#264F11",
  sage:     "#5A7A38",
  sageLt:   "#8BAA60",
  sagePale: "#DCE9CC",
  border:   "#E6E2D8",
  muted:    "#8A8375",
  dim:      "#A8A194",
  ink:      "#16210A",
  scan:     "rgba(139,170,96,0.85)",
};

const card = {
  background: TOKEN.paper,
  border: `1px solid rgba(0,0,0,0.04)`,
  borderRadius: "20px",
  boxShadow: "0 12px 36px -12px rgba(26, 54, 9, 0.06)",
};

// ─── Leaf SVG mark ────────────────────────────────────────────────────────────
const LeafMark = ({ size = 32, fill = TOKEN.sageLt, stroke = TOKEN.forestMd }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path
      d="M16 4C16 4 7 9 7 17C7 23.627 11.373 28 16 28C20.627 28 25 23.627 25 17C25 9 16 4 16 4Z"
      fill={fill} stroke={stroke} strokeWidth="1.4"
    />
    <path
      d="M16 8L16 25M12 13C13.8 15.5 15 17.5 16 20C17 17.5 18.2 15.5 20 13"
      stroke={stroke} strokeWidth="1.3" strokeLinecap="round"
    />
  </svg>
);

// ─── Circular Confidence Gauge ────────────────────────────────────────────────
const ConfidenceGauge = ({ value }) => {
  const pct = Math.round(value * 100);
  const R = 68;
  const circ = 2 * Math.PI * R;
  const dash = (pct / 100) * circ;
  const meta = confidenceMeta(pct);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
      <svg width="176" height="176" viewBox="0 0 176 176" role="img">
        <circle cx="88" cy="88" r={R} fill="none" stroke="#F0EFEA" strokeWidth="13" />
        <circle
          cx="88" cy="88" r={R}
          fill="none"
          stroke={meta.track}
          strokeWidth="13"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          transform="rotate(-90 88 88)"
          style={{ transition: "stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1)" }}
        />
        <circle cx="88" cy="88" r="52" fill={meta.bg} />
        <text
          x="88" y="82"
          textAnchor="middle"
          fontFamily="'DM Serif Display', serif"
          fontSize="28" fontWeight="700"
          fill={meta.color}
        >
          {pct}%
        </text>
        <text
          x="88" y="102"
          textAnchor="middle"
          fontFamily="'DM Sans', sans-serif"
          fontSize="11"
          fill={meta.color}
          opacity="0.75"
        >
          CONFIDENCE
        </text>
      </svg>
      <span style={{
        display: "inline-flex", alignItems: "center", gap: "6px",
        background: meta.bg, color: meta.color,
        padding: "5px 14px", borderRadius: "99px",
        fontSize: "12px", fontWeight: "600",
      }}>
        <span style={{
          width: "7px", height: "7px", borderRadius: "50%",
          background: meta.track, display: "inline-block",
        }} />
        {meta.label}
      </span>
    </div>
  );
};

// ─── Feature Card (Formerly Prototype Card) ──────────────────────────────────
const FeatureCard = ({ index, score, maxScore }) => {
  const pct = Math.round((score / maxScore) * 100);
  const barW = `${pct}%`;

  return (
    <div className="premium-card" style={{ ...card, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          position: "relative",
          aspectRatio: "1 / 1",
          background: TOKEN.cream,
          overflow: "hidden",
        }}
      >
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: "6px",
        }}>
          <LeafMark size={36} fill="#D0DFB8" stroke={TOKEN.sageLt} />
          <span style={{ fontSize: "10px", color: TOKEN.dim, fontFamily: "'DM Sans', sans-serif", fontWeight: 500 }}>
            Visual Marker {index + 1}
          </span>
        </div>
        {index === 0 && (
          <div style={{
            position: "absolute", top: "7px", right: "7px",
            background: TOKEN.forestMd, color: "#fff",
            fontSize: "9px", fontWeight: 700,
            padding: "3px 7px", borderRadius: "99px",
            fontFamily: "'DM Sans', sans-serif", letterSpacing: "0.06em",
          }}>
            TOP
          </div>
        )}
      </div>

      <div style={{ padding: "10px 12px", borderTop: `1px solid ${TOKEN.border}` }}>
        <div style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "baseline", marginBottom: "6px",
        }}>
          <span style={{ fontSize: "10px", color: TOKEN.muted, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.07em", fontFamily: "'DM Sans', sans-serif" }}>
            Match Score
          </span>
          <span style={{ fontSize: "12px", color: TOKEN.forest, fontWeight: 700, fontFamily: "'DM Serif Display', serif" }}>
            {score.toFixed(2)}
          </span>
        </div>
        <div style={{
          background: TOKEN.sagePale, borderRadius: "99px",
          height: "5px", overflow: "hidden",
        }}>
          <div style={{
            height: "100%", width: barW,
            background: `linear-gradient(90deg, ${TOKEN.sageLt}, ${TOKEN.forestMd})`,
            borderRadius: "99px",
            transition: "width 1s cubic-bezier(.4,0,.2,1) 0.2s",
          }} />
        </div>
      </div>
    </div>
  );
};

// ─── Custom Bar Chart Tooltip ─────────────────────────────────────────────────
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: TOKEN.paper, border: `1px solid ${TOKEN.border}`,
      borderRadius: "8px", padding: "8px 12px",
      fontSize: "12px", color: TOKEN.ink,
      fontFamily: "'DM Sans', sans-serif",
      boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
    }}>
      <div style={{ fontWeight: 600, marginBottom: "2px", color: TOKEN.forest }}>{payload[0].payload.name}</div>
      <div style={{ color: TOKEN.muted }}>{payload[0].value.toFixed(2)}%</div>
    </div>
  );
};

// ─── Info Modal ───────────────────────────────────────────────────────────────
const InfoModal = ({ onClose }) => (
  <div
    onClick={onClose}
    style={{
      position: "fixed", inset: 0, zIndex: 200,
      background: "rgba(20,38,8,0.55)",
      backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: "20px",
    }}
  >
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        background: TOKEN.paper, borderRadius: "18px",
        border: `1px solid ${TOKEN.border}`,
        padding: "36px", maxWidth: "520px", width: "100%",
        boxShadow: "0 24px 48px rgba(0,0,0,0.2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
        <LeafMark size={36} fill={TOKEN.sagePale} stroke={TOKEN.sage} />
        <h2 style={{
          fontFamily: "'DM Serif Display', serif",
          fontSize: "24px", color: TOKEN.forest, margin: 0,
        }}>
          About the Technology
        </h2>
      </div>

      <div style={{ width: "36px", height: "2px", background: TOKEN.sageLt, marginBottom: "20px", borderRadius: "2px" }} />

      {[
        ["Intelligent Diagnostics", "This application utilizes state-of-the-art computer vision to analyze complex botanical structures and identify potential health issues in plants."],
        ["Explainable Decisions", "Rather than acting as a 'black box', the system highlights the specific visual regions and patterns that contributed to its final diagnosis, ensuring complete transparency."],
        ["Precision & Speed", "Images are processed through advanced feature-extraction algorithms, providing rapid, highly accurate assessments designed to assist in agricultural research."],
      ].map(([title, body]) => (
        <div key={title} style={{ marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", fontWeight: 700, color: TOKEN.sage, letterSpacing: "0.09em", textTransform: "uppercase", marginBottom: "4px", fontFamily: "'DM Sans', sans-serif" }}>
            {title}
          </div>
          <p style={{ fontSize: "14px", lineHeight: "1.65", color: TOKEN.muted, margin: 0, fontFamily: "'DM Sans', sans-serif" }}>{body}</p>
        </div>
      ))}

      <button
        onClick={onClose}
        style={{
          marginTop: "12px", background: TOKEN.forestMd, color: "#fff",
          border: "none", padding: "12px 28px", borderRadius: "8px",
          cursor: "pointer", fontSize: "14px", fontFamily: "'DM Sans', sans-serif",
          fontWeight: 600, transition: "background 0.2s",
        }}
      >
        Close
      </button>
    </div>
  </div>
);

// ─── App Shell ────────────────────────────────────────────────────────────────
export default function PlantWildApp() {
  const [phase, setPhase] = useState("upload"); 
  const [imageUrl, setImageUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const fileRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setPhase("preview");
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const analyze = async () => {
    setPhase("loading");
    const data = await mockFetch(imageFile);
    setResult(data);
    setPhase("results");
  };

  const reset = () => { setPhase("upload"); setImageUrl(null); setImageFile(null); setResult(null); };

  const pct     = result ? Math.round(result.confidence * 100) : 0;
  const maxProto = result ? Math.max(...result.prototype_scores) : 1;

  const chartData = result
    ? result.top_k_classes.map((c, i) => ({
        name: toTitleCase(c),
        value: parseFloat((result.top_k_probs[i] * 100).toFixed(3)),
      }))
    : [];

  const BAR_COLORS = ["#1A3609", "#264F11", "#5A7A38", "#8BAA60", "#B4CCAA"];

  return (
    <div style={{ minHeight: "100vh", background: `radial-gradient(ellipse at 50% -10%, #FFFFFF 0%, ${TOKEN.cream} 60%)`, fontFamily: "'DM Sans', sans-serif", color: TOKEN.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        @keyframes scanLine {
          0%   { top: -4px; }
          100% { top: 100%; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.55; transform: scale(0.9); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .fade-up { animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; }
        .fade-up-d1 { animation-delay: 0.1s; }
        .fade-up-d2 { animation-delay: 0.2s; }
        .fade-up-d3 { animation-delay: 0.3s; }

        .drop-zone {
          border: 1px dashed rgba(90, 122, 56, 0.35);
          background: linear-gradient(180deg, #FFFFFF 0%, #FDFCF8 100%);
          box-shadow: 0 16px 32px -10px rgba(26, 54, 9, 0.04);
          transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
          cursor: pointer;
        }
        .drop-zone:hover, .drop-zone.active {
          border: 1px solid rgba(90, 122, 56, 0.7);
          box-shadow: 0 24px 48px -12px rgba(90, 122, 56, 0.15);
          transform: translateY(-3px);
        }

        .btn-primary {
          background: #1A3609;
          color: #fff;
          border: none;
          padding: 14px 36px;
          border-radius: 12px;
          font-size: 15px;
          font-weight: 600;
          font-family: 'DM Sans', sans-serif;
          cursor: pointer;
          letter-spacing: 0.02em;
          box-shadow: 0 8px 20px -6px rgba(26, 54, 9, 0.4);
          transition: all 0.2s ease;
        }
        .btn-primary:hover { background: #264F11; transform: translateY(-1px); box-shadow: 0 10px 24px -6px rgba(26, 54, 9, 0.5); }
        .btn-primary:active { transform: scale(0.98); }

        .btn-ghost {
          background: #FFFFFF;
          color: #5A7A38;
          border: 1px solid #DCE9CC;
          padding: 14px 28px;
          border-radius: 12px;
          font-size: 15px;
          font-weight: 500;
          font-family: 'DM Sans', sans-serif;
          cursor: pointer;
          box-shadow: 0 4px 12px -4px rgba(0,0,0,0.03);
          transition: all 0.2s ease;
        }
        .btn-ghost:hover { border-color: #5A7A38; background: #FDFCF8; }

        .premium-card {
          transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .premium-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 22px 44px -12px rgba(30, 61, 10, 0.15) !important;
        }

        .team-avatar {
          transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), background 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }
        .team-row:hover .team-avatar {
          transform: scale(1.15);
          background: #5A7A38 !important;
          border-color: rgba(255,255,255,0.4) !important;
          color: #fff !important;
        }
        .nav-link {
          background: transparent; border: none; color: rgba(255,255,255,0.8);
          font-size: 13px; font-weight: 500; font-family: 'DM Sans', sans-serif;
          cursor: pointer; transition: color 0.2s;
        }
        .nav-link:hover { color: #fff; }
      `}</style>

      {showInfo && <InfoModal onClose={() => setShowInfo(false)} />}

      {/* ══════════════ PREMIUM HEADER ══════════════ */}
      <header style={{
        background: "rgba(26, 54, 9, 0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        position: "sticky", top: 0, zIndex: 100,
        boxShadow: "0 4px 30px rgba(0, 0, 0, 0.12)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
      }}>
        <div style={{
          maxWidth: "1200px", margin: "0 auto",
          padding: "0 24px", height: "64px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", cursor: "pointer" }} onClick={reset}>
            <LeafMark size={28} fill={TOKEN.sageLt} stroke={TOKEN.sagePale} />
            <span style={{
              fontFamily: "'DM Serif Display', serif",
              fontSize: "20px", color: "#FFFFFF", letterSpacing: "0.02em",
            }}>
              PlantWild <em style={{ color: TOKEN.sageLt, fontStyle: "italic", fontWeight: 400 }}>AI</em>
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
            <button className="nav-link" onClick={() => setShowInfo(true)}>
              About Technology
            </button>
            {result && (
              <button onClick={reset} style={{
                background: "rgba(255,255,255,0.12)",
                border: "1px solid rgba(255,255,255,0.2)",
                color: "#FFFFFF", padding: "8px 18px",
                borderRadius: "8px", cursor: "pointer",
                fontSize: "13px", fontWeight: 600, fontFamily: "'DM Sans', sans-serif",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.2)"}
              onMouseLeave={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.12)"}
              >
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ══════════════ MAIN ══════════════ */}
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "60px 24px 80px" }}>

        {/* ── UPLOAD ── */}
        {phase === "upload" && (
          <div className="fade-up" style={{ maxWidth: "580px", margin: "0 auto", textAlign: "center" }}>
            <div style={{ marginBottom: "40px" }}>
              <p style={{ fontSize: "12px", fontWeight: 700, color: TOKEN.sage, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "12px" }}>
                Intelligent Analysis
              </p>
              <h1 style={{
                fontFamily: "'DM Serif Display', serif",
                fontSize: "clamp(40px, 6vw, 60px)",
                color: TOKEN.forest, lineHeight: 1.05,
                marginBottom: "16px",
              }}>
                Diagnose Your<br />
                <em style={{ color: TOKEN.sage, fontStyle: "italic" }}>Plant Health</em>
              </h1>
              <p style={{ fontSize: "16px", color: TOKEN.muted, lineHeight: 1.6, maxWidth: "460px", margin: "0 auto" }}>
                Upload a botanical image. Our proprietary models will evaluate the specimen and visually highlight structural patterns driving the diagnosis.
              </p>
            </div>

            <div
              className={`drop-zone${dragging ? " active" : ""}`}
              style={{ borderRadius: "24px", padding: "50px 32px", marginBottom: "16px" }}
              onClick={() => fileRef.current.click()}
              onDrop={onDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
            >
              <div style={{
                width: "72px", height: "72px", borderRadius: "50%",
                background: TOKEN.sagePale,
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 20px",
                boxShadow: "inset 0 2px 8px rgba(255,255,255,0.5)",
              }}>
                <LeafMark size={36} fill="#C0D8A0" stroke={TOKEN.sage} />
              </div>

              <p style={{
                fontFamily: "'DM Serif Display', serif",
                fontSize: "24px", color: TOKEN.forest,
                marginBottom: "8px",
              }}>
                Drag & drop image here<br />or click to browse
              </p>
              <p style={{ fontSize: "13px", color: TOKEN.dim, fontWeight: 500 }}>
                JPG · PNG · WEBP &nbsp;—&nbsp; High Resolution Supported
              </p>

              <div style={{ 
                marginTop: "32px", 
                paddingTop: "24px", 
                borderTop: `1px solid rgba(0,0,0,0.05)`,
                display: "flex", flexDirection: "column", gap: "4px"
              }}>
                <p style={{ fontSize: "13px", color: TOKEN.forest, fontWeight: 600, letterSpacing: "0.02em" }}>
                  Project: PlantWild AI
                </p>
                <p style={{ fontSize: "13px", color: TOKEN.muted }}>
                
                </p>
              </div>
            </div>

            <input
              ref={fileRef} type="file" accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        )}

        {/* ── PREVIEW ── */}
        {phase === "preview" && (
          <div className="fade-up" style={{ maxWidth: "520px", margin: "0 auto", textAlign: "center" }}>
            <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "32px", color: TOKEN.forest, marginBottom: "24px" }}>
              Specimen Selected
            </h2>
            <div className="premium-card" style={{
              ...card,
              overflow: "hidden", marginBottom: "32px",
            }}>
              <img
                src={imageUrl} alt="Selected leaf"
                style={{ width: "100%", maxHeight: "400px", objectFit: "contain", display: "block", background: TOKEN.paper }}
              />
            </div>
            <div style={{ display: "flex", gap: "12px", justifyContent: "center" }}>
              <button className="btn-ghost" onClick={reset}>Cancel</button>
              <button className="btn-primary" onClick={analyze}>Analyze Specimen</button>
            </div>
          </div>
        )}

        {/* ── LOADING ── */}
        {phase === "loading" && (
          <div className="fade-up" style={{ maxWidth: "520px", margin: "0 auto", textAlign: "center" }}>
            <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "32px", color: TOKEN.forest, marginBottom: "24px" }}>
              Extracting Features...
            </h2>
            <div style={{
              position: "relative", borderRadius: "20px",
              overflow: "hidden", marginBottom: "32px",
              boxShadow: "0 12px 36px -12px rgba(26, 54, 9, 0.12)",
              background: TOKEN.paper, border: `1px solid rgba(0,0,0,0.04)`
            }}>
              <img
                src={imageUrl} alt="Leaf being analyzed"
                style={{ width: "100%", maxHeight: "400px", objectFit: "contain", display: "block" }}
              />
              <div style={{ position: "absolute", inset: 0, background: "rgba(26,54,9,0.08)" }} />
              <div style={{
                position: "absolute", left: 0, right: 0, height: "4px",
                background: `linear-gradient(90deg, transparent 0%, ${TOKEN.scan} 40%, #A8CC70 60%, transparent 100%)`,
                boxShadow: `0 0 16px 6px rgba(139,170,96,0.4)`,
                animation: "scanLine 1.8s linear infinite",
              }} />
              
              <div style={{
                position: "absolute", bottom: "20px", left: "50%",
                transform: "translateX(-50%)",
                background: "rgba(20,38,8,0.85)",
                backdropFilter: "blur(12px)",
                borderRadius: "99px", padding: "10px 20px",
                display: "flex", alignItems: "center", gap: "10px",
                whiteSpace: "nowrap", boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
              }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: TOKEN.sageLt, animation: "pulse 1.2s ease infinite" }} />
                <span style={{ color: "#fff", fontSize: "13px", fontWeight: 500 }}>Analyzing high-resolution features...</span>
              </div>
            </div>
          </div>
        )}

        {/* ── RESULTS ── */}
        {phase === "results" && result && (
          <div>
            <div className="fade-up" style={{ marginBottom: "36px" }}>
              <p style={{ fontSize: "12px", fontWeight: 700, color: TOKEN.sage, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "8px" }}>
                Analysis Complete
              </p>
              <h1 style={{
                fontFamily: "'DM Serif Display', serif",
                fontSize: "clamp(34px, 5vw, 48px)", color: TOKEN.forest,
              }}>
                Diagnosis Report
              </h1>
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "24px", marginBottom: "24px",
            }}>
              {/* LEFT */}
              <div className="fade-up fade-up-d1 premium-card" style={{ ...card, padding: "32px", display: "flex", flexDirection: "column", gap: "24px" }}>
                <div style={{
                  borderRadius: "14px", overflow: "hidden",
                  border: `1px solid ${TOKEN.border}`,
                }}>
                  <img
                    src={imageUrl} alt="Analyzed leaf"
                    style={{ width: "100%", maxHeight: "240px", objectFit: "cover", display: "block", background: TOKEN.cream }}
                  />
                </div>
                <div>
                  <div style={{
                    fontSize: "11px", fontWeight: 700, color: TOKEN.sage,
                    letterSpacing: "0.1em", textTransform: "uppercase",
                    marginBottom: "8px",
                  }}>
                    Primary Assessment
                  </div>
                  <div style={{
                    fontFamily: "'DM Serif Display', serif",
                    fontSize: "clamp(26px, 3vw, 36px)", color: TOKEN.forest, lineHeight: 1.1,
                  }}>
                    {toTitleCase(result.predicted_class)}
                  </div>
                </div>
                <div style={{ borderTop: `1px solid ${TOKEN.border}` }} />
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <ConfidenceGauge value={result.confidence} />
                </div>
              </div>

              {/* RIGHT */}
              <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
                <div className="fade-up fade-up-d2 premium-card" style={{ ...card, padding: "28px", flexGrow: 1 }}>
                  <div style={{ marginBottom: "20px" }}>
                    <h3 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "22px", color: TOKEN.forest, marginBottom: "4px" }}>
                      Probability Distribution
                    </h3>
                    <p style={{ fontSize: "13px", color: TOKEN.muted }}>Top nearest categorizations based on structural analysis.</p>
                  </div>
                  <div style={{ position: "relative", width: "100%", height: "220px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 48, top: 0, bottom: 0 }}>
                        <XAxis type="number" domain={[0, 100]} hide />
                        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 12, fontFamily: "'DM Sans', sans-serif", fill: TOKEN.muted }} axisLine={false} tickLine={false} />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: TOKEN.cream }} />
                        <Bar dataKey="value" radius={[0, 6, 6, 0]} label={{ position: "right", fontSize: 12, fill: TOKEN.muted, fontFamily: "'DM Sans', sans-serif", formatter: (v) => `${v.toFixed(1)}%` }}>
                          {chartData.map((_, i) => <Cell key={i} fill={BAR_COLORS[i]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>

            {/* VISUAL MARKERS */}
            <div className="fade-up premium-card" style={{ ...card, padding: "32px" }}>
              <div style={{ marginBottom: "32px", maxWidth: "600px" }}>
                <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "28px", color: TOKEN.forest, marginBottom: "8px" }}>
                  Visual Justification
                </h2>
                <p style={{ fontSize: "14px", color: TOKEN.muted, lineHeight: 1.6 }}>
                  The AI identifies distinct regions of interest on the specimen. Higher activation scores represent a stronger structural match with known disease patterns.
                </p>
              </div>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                gap: "16px",
              }}>
                {result.prototype_scores.map((score, i) => (
                  <FeatureCard key={i} index={i} score={score} maxScore={maxProto} />
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ══════════════ PREMIUM FOOTER ══════════════ */}
      <footer style={{
        background: TOKEN.forest,
        borderTop: "1px solid rgba(255,255,255,0.05)",
        marginTop: "auto"
      }}>
        <div style={{
          maxWidth: "1200px", margin: "0 auto",
          padding: "48px 24px 32px",
          display: "flex", flexDirection: "row", flexWrap: "wrap",
          justifyContent: "space-between", gap: "40px"
        }}>
          {/* Brand Col */}
          <div style={{ flex: "1 1 300px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
              <LeafMark size={28} fill={TOKEN.sageLt} stroke={TOKEN.sagePale} />
              <span style={{
                fontFamily: "'DM Serif Display', serif",
                fontSize: "20px", color: "#F6F2EA",
              }}>
                PlantWild <em style={{ color: TOKEN.sageLt, fontStyle: "italic", fontWeight: 400 }}>AI</em>
              </span>
            </div>
            <p style={{ fontSize: "13px", color: "rgba(255,255,255,0.6)", lineHeight: 1.7, maxWidth: "320px" }}>
              Delivering transparent, explainable machine learning diagnostics for modern botanical research and agriculture.
            </p>
          </div>

          {/* Team Col */}
          <div style={{ flex: "0 1 200px" }}>
            <div style={{ fontSize: "11px", fontWeight: 700, color: TOKEN.sageLt, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "16px" }}>
              Research & Development
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {[
                "Milan Kundu",
                "Nilanjan Pan",
                
              ].map((name, i) => (
                <div key={i} className="team-row" style={{
                  display: "flex", alignItems: "center", gap: "10px",
                  cursor: "pointer",
                }}>
                  <div className="team-avatar" style={{
                    width: "24px", height: "24px", borderRadius: "50%",
                    background: TOKEN.forestMd,
                    border: `1px solid rgba(139,170,96,0.3)`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: "9px", fontWeight: 700, color: TOKEN.sageLt,
                  }}>
                    {name.split(" ").map(n => n[0]).join("")}
                  </div>
                  <span style={{ fontSize: "13px", color: "rgba(255,255,255,0.75)", transition: "color 0.2s" }}>{name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div style={{
          borderTop: "1px solid rgba(255,255,255,0.06)",
          padding: "20px 24px",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.4)" }}>
            © {new Date().getFullYear()} PlantWild AI · All Rights Reserved
          </span>
        </div>
      </footer>
    </div>
  );
}