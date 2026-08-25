'use client';

import { useState, useEffect } from 'react';
import { Sparkles, Loader2, AlertTriangle } from 'lucide-react';
import { formatINR } from '@/lib/formatINR';
import styles from './SummaryPanel.module.css';

interface SummaryPanelProps {
  selectedSiteId: string | null;
  customResult?: any | null;
  customLoading?: boolean;
}

export default function SummaryPanel({ selectedSiteId, customResult, customLoading }: SummaryPanelProps) {
  const [text, setText] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    // When a custom point is selected, build the summary from the already-fetched result
    if (selectedSiteId === 'custom') {
      setText('');
      setError(null);
      return;
    }
    if (!selectedSiteId) return;

    let isMounted = true;
    const fetchSummary = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `http://localhost:8000/api/summary?site_id=${encodeURIComponent(selectedSiteId)}`
        );
        if (!res.ok) throw new Error('Failed to fetch summary');
        const json = await res.json();
        if (isMounted) setText(json.summary);
      } catch {
        if (isMounted) setError('Could not load executive summary.');
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchSummary();
    return () => { isMounted = false; };
  }, [selectedSiteId]);

  if (!selectedSiteId) return null;

  // Custom point summary
  if (selectedSiteId === 'custom') {
    return (
      <div className={`glass-panel ${styles.panel}`}>
        <div className={styles.iconWrapper}>
          <Sparkles size={24} />
        </div>
        <div className={styles.content}>
          <h2>Custom Location Score</h2>
          {customLoading && (
            <p style={{ color: '#94a3b8', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: 8 }}>
              <Loader2 size={14} style={{ animation: 'spin 0.8s linear infinite' }} />
              Scoring location…
            </p>
          )}
          {!customLoading && customResult && mounted && (
            <div className={styles.customSummary}>
              <div className={styles.customRow}>
                <span className={styles.customLabel}>ML Score</span>
                <span className={styles.customValue} style={{ color: '#10b981' }}>
                  {formatINR(customResult.predicted_score)}
                </span>
              </div>
              <div className={styles.customRow}>
                <span className={styles.customLabel}>Baseline</span>
                <span className={styles.customValue} style={{ color: '#94a3b8' }}>
                  {formatINR(customResult.baseline_score)}
                </span>
              </div>
              <div className={styles.customRow}>
                <span className={styles.customLabel}>Rank</span>
                <span className={styles.customValue} style={{ color: '#f0abfc' }}>
                  #{customResult.rank} of {customResult.total_candidates}
                </span>
              </div>

              {/* OOB warning badge */}
              {customResult.oob_features && customResult.oob_features.length > 0 && (
                <div className={styles.oobWarning}>
                  <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <div className={styles.oobTitle}>Extrapolated estimate</div>
                    <div className={styles.oobBody}>
                      {customResult.oob_features.length} feature
                      {customResult.oob_features.length > 1 ? 's' : ''} exceed the
                      50-site training range:
                      {' '}<strong>
                        {customResult.oob_features.map((f: any) => {
                          const label = f.feature
                            .replace('_1000m', ' (1km)')
                            .replace('_3000m', ' (3km)')
                            .replace(/_/g, ' ');
                          return `${label} (${f.side === 'high' ? '+' : '-'}${f.delta_pct.toFixed(0)}%)`;
                        }).join(', ')}
                      </strong>.
                      {' '}XGBoost cannot extrapolate beyond its training boundary —
                      the ML score may be conservative; the baseline formula
                      continues to extrapolate linearly.
                    </div>
                  </div>
                </div>
              )}

              <p style={{ marginTop: 10, color: '#94a3b8', fontSize: '0.82rem', lineHeight: 1.5 }}>
                This custom location would rank <strong style={{ color: '#f0abfc' }}>
                  #{customResult.rank}
                </strong> among the {customResult.total_candidates - 1} pre-screened candidates.
                The SHAP chart below shows which features drive this estimate.
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Standard candidate summary
  return (
    <div className={`glass-panel ${styles.panel}`}>
      <div className={styles.iconWrapper}>
        <Sparkles size={24} />
      </div>
      <div className={styles.content}>
        <h2>Executive Recommendation</h2>
        {loading && <p style={{ color: '#64748b', fontStyle: 'italic' }}>Generating analysis...</p>}
        {error && <p style={{ color: '#ef4444' }}>{error}</p>}
        {!loading && !error && text && <p>{text}</p>}
      </div>
    </div>
  );
}
