
import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const LATENT_DIM = 16;
const IMG_H = 96;
const IMG_W = 96;
const SCALE = 4;

// must match how you exported lstm_latent_step.onnx
const NUM_LAYERS = 2;
const HIDDEN_DIM = 128;

function LatentRolloutVisualizer() {
  const [decoderSession, setDecoderSession] = useState(null);
  const [lstmSession, setLstmSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [latent, setLatent] = useState(() => Array(LATENT_DIM).fill(0));
  const [hData, setHData] = useState(null); // Float32Array for h
  const [cData, setCData] = useState(null); // Float32Array for c

  const smallCanvasRef = useRef(null); // 96x96
  const bigCanvasRef = useRef(null);   // 384x384

  // Load both ONNX models once
  useEffect(() => {
    async function loadModels() {
      try {
        setLoading(true);
        const [dec, lstm] = await Promise.all([
          ort.InferenceSession.create("/vae_decoder16.onnx"),
          ort.InferenceSession.create("/lstm_latent_step.onnx"),
        ]);
        setDecoderSession(dec);
        setLstmSession(lstm);
      } catch (e) {
        console.error(e);
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }

    loadModels();
  }, []);

  // Decode current latent with VAE and draw image whenever latent changes
  useEffect(() => {
    async function runDecoder() {
      if (!decoderSession || !smallCanvasRef.current || !bigCanvasRef.current) return;

      const zData = new Float32Array(LATENT_DIM);
      for (let i = 0; i < LATENT_DIM; i++) {
        zData[i] = latent[i];
      }
      const zTensor = new ort.Tensor("float32", zData, [1, LATENT_DIM]);

      try {
        const outputs = await decoderSession.run({ z: zTensor });
        const xRecon = outputs["x_recon"]; // [1, 3, 96, 96]
        const data = xRecon.data;

        // draw to small canvas
        const smallCanvas = smallCanvasRef.current;
        const sctx = smallCanvas.getContext("2d");
        const imageData = sctx.createImageData(IMG_W, IMG_H);
        const rgba = imageData.data;
        const planeSize = IMG_H * IMG_W;

        for (let y = 0; y < IMG_H; y++) {
          for (let x = 0; x < IMG_W; x++) {
            const idxHW = y * IMG_W + x;

            const r = data[0 * planeSize + idxHW];
            const g = data[1 * planeSize + idxHW];
            const b = data[2 * planeSize + idxHW];

            const idxRGBA = idxHW * 4;
            rgba[idxRGBA + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
            rgba[idxRGBA + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
            rgba[idxRGBA + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
            rgba[idxRGBA + 3] = 255;
          }
        }

        sctx.putImageData(imageData, 0, 0);

        // upscale into big canvas
        const bigCanvas = bigCanvasRef.current;
        const bctx = bigCanvas.getContext("2d");
        bctx.imageSmoothingEnabled = false;
        bctx.clearRect(0, 0, bigCanvas.width, bigCanvas.height);
        bctx.drawImage(
          smallCanvas,
          0,
          0,
          smallCanvas.width,
          smallCanvas.height,
          0,
          0,
          bigCanvas.width,
          bigCanvas.height
        );
      } catch (e) {
        console.error(e);
        setError(String(e));
      }
    }

    runDecoder();
  }, [decoderSession, latent]);

  const handleSliderChange = (idx, value) => {
    setLatent((prev) => {
      const next = [...prev];
      next[idx] = value;
      return next;
    });
    // optionally: reset hidden state when you manually edit latent
    // setHData(null);
    // setCData(null);
  };

  const resetLatentAndHidden = () => {
    setLatent(Array(LATENT_DIM).fill(0));
    setHData(null);
    setCData(null);
  };

  const stepWithAction = async (actionVal) => {
    if (!lstmSession) return;

    try {
      const zArr = new Float32Array(LATENT_DIM);
      for (let i = 0; i < LATENT_DIM; i++) {
        zArr[i] = latent[i];
      }
      const latentTensor = new ort.Tensor("float32", zArr, [1, LATENT_DIM]);

      const actionTensor = new ort.Tensor(
        "float32",
        new Float32Array([actionVal]),
        [1, 1]
      );

      let hArr = hData;
      let cArr = cData;
      if (!hArr || !cArr) {
        hArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM); // zeros
        cArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM);
      }
      const hTensor = new ort.Tensor("float32", hArr, [NUM_LAYERS, 1, HIDDEN_DIM]);
      const cTensor = new ort.Tensor("float32", cArr, [NUM_LAYERS, 1, HIDDEN_DIM]);

      const outputs = await lstmSession.run({
        latent: latentTensor,
        action: actionTensor,
        h0: hTensor,
        c0: cTensor,
      });

      const nextLatentTensor = outputs["next_latent"]; // [1,16]
      const h1Tensor = outputs["h1"];                  // [2,1,128]
      const c1Tensor = outputs["c1"];                  // [2,1,128]

      // update latent sliders
      const nextLatentArr = Array.from(nextLatentTensor.data);
      setLatent(nextLatentArr);

      // store new hidden state
      setHData(new Float32Array(h1Tensor.data));
      setCData(new Float32Array(c1Tensor.data));
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  };

  const disabled = loading || !decoderSession || !lstmSession;

  return (
    <div style={{ padding: 16, fontFamily: "sans-serif" }}>
      <h2>LSTM</h2>

      {loading && <p>Loading ONNX models…</p>}
      {error && (
        <p style={{ color: "red", whiteSpace: "pre-wrap" }}>
          Error: {error}
        </p>
      )}

      <div style={{ display: "flex", gap: 32, alignItems: "flex-start" }}>
        {/* Sliders + buttons */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 52,
            maxHeight: "100%",
            maxWidth: "100%",
            overflowY: "auto",
          }}
        >
          {latent.map((value, i) => (
            <div key={i}>
              <label
                style={{
                  display: "block",
                  fontSize: 20,
                  marginBottom: 4,
                  fontFamily: "monospace",
                  color: "#000",
                }}
              >
                z[{i}] = {value.toFixed(2)}
              </label>
              <input
                type="range"
                min={-3}
                max={3}
                step={0.05}
                value={value}
                onChange={(e) =>
                  handleSliderChange(i, Number(e.target.value))
                }
                style={{ width: 200, accentColor: "#2563eb" }}
              />
            </div>
          ))}
        </div>

        {/* Image + controls */}
        <div>
          <p style={{ fontSize: 25, color: "#666", marginBottom: 8 }}>
            Decoded Image
          </p>
          {/* offscreen small canvas */}
          <canvas
            ref={smallCanvasRef}
            width={IMG_W}
            height={IMG_H}
            style={{ display: "none" }}
          />
          {/* visible upscaled canvas */}
          <canvas
            ref={bigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={{
              width: `${IMG_W * SCALE}px`,
              height: `${IMG_H * SCALE}px`,
              imageRendering: "pixelated",
              border: "1px solid #ccc",
              backgroundColor: "#000",
            }}
          />

          <div style={{ marginTop: 16, display: "flex", gap: 12 }}>
            <button
              onClick={() => stepWithAction(0)}
              disabled={disabled}
              style={{ padding: "8px 16px", fontSize: 22 }}
            >
              Action: Left
            </button>
            <button
              onClick={() => stepWithAction(1)}
              disabled={disabled}
              style={{ padding: "8px 16px", fontSize: 22 }}
            >
              Action: Right
            </button>
          </div>

          <button
            onClick={resetLatentAndHidden}
            style={{ marginTop: 12, padding: "8px 16px", fontSize: 22 }}
          >
            Reset latent & hidden state
          </button>
        </div>
      </div>
    </div>
  );
}

export default LatentRolloutVisualizer;
