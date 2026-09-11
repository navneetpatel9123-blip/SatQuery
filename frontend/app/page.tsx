import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.hero}>
      {/* ─── Navigation ─────────────────────────── */}
      <nav className={styles.nav}>
        <div className={styles.nav__brand}>
          <div className={styles.nav__logo}>S</div>
          <div className={styles.nav__name}>
            Sat<span>Query</span> AI
          </div>
        </div>

        <ul className={styles.nav__links}>
          <li className={styles.nav__link}>Platform</li>
          <li className={styles.nav__link}>Documentation</li>
          <li className={styles.nav__link}>API</li>
          <li className={styles.nav__link}>Research</li>
        </ul>

        <div className={styles.nav__actions}>
          <a href="/upload" className="btn btn-primary">
            Launch Analysis
          </a>
        </div>
      </nav>

      {/* ─── Hero Content ───────────────────────── */}
      <div className={styles.hero__content}>
        <div className={`${styles.hero__left} animate-fade-in-up`}>
          <div className={styles.hero__badge}>
            <span className={styles["hero__badge-dot"]} />
            Bi-Temporal Change VQA
          </div>

          <h1 className={styles.hero__title}>
            Understand{" "}
            <span className="text-gradient">satellite change</span>
            <br />
            with explainable AI
          </h1>

          <p className={styles.hero__subtitle}>
            Upload co-registered GeoTIFF image pairs. Our multi-model pipeline
            detects, classifies, and explains changes — with full execution
            trace for every conclusion.
          </p>

          <div className={styles.hero__actions}>
            <a href="/upload" className="btn btn-primary btn-lg">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload GeoTIFF Pair
            </a>
            <a href="/docs" className="btn btn-secondary btn-lg">
              View Documentation
            </a>
          </div>

          <div className={styles.hero__stats}>
            <div className={styles.hero__stat}>
              <span className={styles["hero__stat-value"]}>3-Stage</span>
              <span className={styles["hero__stat-label"]}>ML Pipeline</span>
            </div>
            <div className={styles.hero__stat}>
              <span className={styles["hero__stat-value"]}>Full</span>
              <span className={styles["hero__stat-label"]}>Execution Trace</span>
            </div>
            <div className={styles.hero__stat}>
              <span className={styles["hero__stat-value"]}>VQA</span>
              <span className={styles["hero__stat-label"]}>Natural Language</span>
            </div>
          </div>
        </div>

        {/* ─── Demo Card ──────────────────────────── */}
        <div className={`${styles.hero__right} animate-fade-in-up animate-delay-2`}>
          <div className={styles["demo-card"]}>
            <div className={styles["demo-card__header"]}>
              <div className={styles["demo-card__title"]}>
                <div className={styles["demo-card__icon"]}>S</div>
                SatQuery AI Analysis
              </div>
              <span className="badge badge--success">
                <span className="badge-dot" />
                Complete
              </span>
            </div>

            <div className={styles["demo-card__body"]}>
              {/* Terminal output */}
              <div className={styles["demo-terminal"]}>
                <div className={`${styles["demo-terminal__line"]} animate-fade-in-up animate-delay-1`}>
                  <span className={styles["demo-terminal__prefix"]}>Task:</span>
                  <span className={styles["demo-terminal__text"]}>Bi-temporal Change VQA</span>
                </div>
                <div className={`${styles["demo-terminal__line"]} animate-fade-in-up animate-delay-2`}>
                  <span className={styles["demo-terminal__prefix"]}>Input:</span>
                  <span className={styles["demo-terminal__text"]}>2 GeoTIFF images</span>
                </div>
                <div className={`${styles["demo-terminal__line"]} animate-fade-in-up animate-delay-3`}>
                  <span className={styles["demo-terminal__prefix"]}>Valid:</span>
                  <span className={styles["demo-terminal__text--accent"]}>
                    ✓ Co-registered ✓ Same AOI ✓ T1 &lt; T2
                  </span>
                </div>
              </div>

              {/* Pipeline steps */}
              <div className={styles["demo-pipeline"]}>
                <div className={`${styles["demo-step"]} animate-fade-in-up animate-delay-2`}>
                  <div className={`${styles["demo-step__indicator"]} ${styles["demo-step__indicator--success"]}`}>✓</div>
                  <span className={styles["demo-step__name"]}>Change Detection</span>
                  <span className={styles["demo-step__arrow"]}>→</span>
                </div>
                <div className={`${styles["demo-step"]} animate-fade-in-up animate-delay-3`}>
                  <div className={`${styles["demo-step__indicator"]} ${styles["demo-step__indicator--success"]}`}>✓</div>
                  <span className={styles["demo-step__name"]}>Change Understanding</span>
                  <span className={styles["demo-step__arrow"]}>→</span>
                </div>
                <div className={`${styles["demo-step"]} animate-fade-in-up animate-delay-4`}>
                  <div className={`${styles["demo-step__indicator"]} ${styles["demo-step__indicator--success"]}`}>✓</div>
                  <span className={styles["demo-step__name"]}>Evidence Extraction</span>
                  <span className={styles["demo-step__arrow"]}>✓</span>
                </div>
              </div>

              {/* Result */}
              <div className={`${styles["demo-result"]} animate-fade-in-up animate-delay-5`}>
                <div className={styles["demo-result__label"]}>Result</div>
                <div className={styles["demo-result__value"]}>Built-up area increased</div>
                <div className={styles["demo-result__evidence"]}>
                  Evidence: Change concentrated in the northern/eastern region
                </div>
                <div style={{ marginTop: '4px' }}>
                  <span className="badge badge--success" style={{ fontSize: '10px' }}>
                    Confidence: High
                  </span>
                </div>
              </div>
            </div>

            {/* Action buttons */}
            <div className={styles["demo-card__actions"]}>
              <button className={`${styles["demo-action-btn"]}`} type="button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                Change Map
              </button>
              <button className={styles["demo-action-btn"]} type="button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="2" y="3" width="20" height="18" rx="2" />
                  <line x1="12" y1="3" x2="12" y2="21" />
                </svg>
                T1 / T2
              </button>
              <button className={`${styles["demo-action-btn"]} ${styles["demo-action-btn--primary"]}`} type="button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
                Execution Trace
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Features Section ────────────────────── */}
      <section className={styles.features}>
        <div className={styles.features__header}>
          <h2 className={styles.features__title}>
            A pipeline you can <span className="text-gradient">trust</span>
          </h2>
          <p className={styles.features__subtitle}>
            Every conclusion is backed by a transparent chain of models,
            intermediate artifacts, and quantified evidence.
          </p>
        </div>

        <div className={styles.features__grid}>
          <div className={`glass-card ${styles["feature-card"]}`}>
            <div className={`${styles["feature-card__icon"]} ${styles["feature-card__icon--cyan"]}`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>
            <h3 className={styles["feature-card__title"]}>Change Detection</h3>
            <p className={styles["feature-card__desc"]}>
              Pixel-level and region-level change detection using spectral indices,
              thresholding, and connected component analysis across bi-temporal image pairs.
            </p>
          </div>

          <div className={`glass-card ${styles["feature-card"]}`}>
            <div className={`${styles["feature-card__icon"]} ${styles["feature-card__icon--purple"]}`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
            </div>
            <h3 className={styles["feature-card__title"]}>Change Understanding</h3>
            <p className={styles["feature-card__desc"]}>
              Classify the nature of change — built-up expansion, deforestation,
              water body shifts — with land cover analysis for both temporal snapshots.
            </p>
          </div>

          <div className={`glass-card ${styles["feature-card"]}`}>
            <div className={`${styles["feature-card__icon"]} ${styles["feature-card__icon--emerald"]}`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <h3 className={styles["feature-card__title"]}>Execution Trace</h3>
            <p className={styles["feature-card__desc"]}>
              Full DAG visualization of every pipeline step — inputs, outputs,
              confidence scores, and intermediate artifacts. The judge view.
            </p>
          </div>

          <div className={`glass-card ${styles["feature-card"]}`}>
            <div className={`${styles["feature-card__icon"]} ${styles["feature-card__icon--amber"]}`}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h3 className={styles["feature-card__title"]}>Visual QA</h3>
            <p className={styles["feature-card__desc"]}>
              Ask natural language questions about the changes. Every answer
              links back to the evidence and execution trace that produced it.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
