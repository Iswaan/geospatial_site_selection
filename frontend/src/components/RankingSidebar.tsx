'use client';

import { useEffect, useRef, useState } from 'react';
import styles from './RankingSidebar.module.css';
import ShapChart from './ShapChart';
import { formatINR } from '@/lib/formatINR';
import { MapPin, Loader2 } from 'lucide-react';

interface RankingSidebarProps {
  scores: any[];
  selectedSiteId: string | null;
  onSelectSite: (id: string) => void;
  customResult?: any | null;
  customLoading?: boolean;
}

export default function RankingSidebar({
  scores, selectedSiteId, onSelectSite, customResult, customLoading
}: RankingSidebarProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  // Auto-scroll to selected row when selectedSiteId changes (map → sidebar sync)
  useEffect(() => {
    if (selectedSiteId && selectedSiteId !== 'custom' && rowRefs.current[selectedSiteId] && listRef.current) {
      rowRefs.current[selectedSiteId]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [selectedSiteId]);

  const customShapData = customResult?.shap ?? null;

  return (
    <div className={styles.sidebar}>
      <div className={styles.header}>
        <h2>Candidate Rankings</h2>
        <p>Sorted by AI-Predicted Revenue</p>
      </div>

      <div className={styles.shapContainer}>
        <ShapChart
          selectedSiteId={selectedSiteId}
          customShapData={customShapData}
        />
      </div>

      <div className={styles.listContainer} ref={listRef}>

        {/* Pinned custom location card */}
        {(customResult || customLoading) && (
          <div
            className={`${styles.row} ${styles.customRow} ${selectedSiteId === 'custom' ? styles.rowSelected : ''}`}
            onClick={() => !customLoading && customResult && onSelectSite('custom')}
          >
            <div className={styles.rank} style={{ color: '#f0abfc' }}>
              <MapPin size={14} />
            </div>
            <div className={styles.content}>
              <div className={styles.siteId} style={{ color: '#f0abfc' }}>Custom Location</div>
              {customLoading ? (
                <div className={styles.scores}>
                  <span style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Loader2 size={12} style={{ animation: 'spin 0.8s linear infinite' }} />
                    Scoring…
                  </span>
                </div>
              ) : customResult && (
                <div className={styles.scores}>
                  <span className={styles.predicted} suppressHydrationWarning>
                    ML: {mounted ? formatINR(customResult.predicted_score) : '...'}
                  </span>
                  <span className={styles.baseline} suppressHydrationWarning>
                    Baseline: {mounted ? formatINR(customResult.baseline_score) : '...'}
                  </span>
                  <span style={{ fontSize: '0.72rem', color: '#f0abfc', fontWeight: 600 }}>
                    Rank #{customResult.rank} / {customResult.total_candidates}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 50 pre-computed candidate rows */}
        {scores.map((scoreObj) => {
          const isSelected = selectedSiteId === scoreObj.site_id;

          return (
            <div
              key={scoreObj.site_id}
              ref={(el) => { rowRefs.current[scoreObj.site_id] = el; }}
              className={`${styles.row} ${isSelected ? styles.rowSelected : ''}`}
              onClick={() => onSelectSite(scoreObj.site_id)}
            >
              <div className={styles.rank}>#{scoreObj.rank}</div>
              <div className={styles.content}>
                <div className={styles.siteId}>{scoreObj.site_id}</div>
                <div className={styles.scores}>
                  <span className={styles.predicted}>
                    ML: <span suppressHydrationWarning>{mounted ? formatINR(scoreObj.predicted_score) : '...'}</span>
                  </span>
                  <span className={styles.baseline}>
                    Baseline: <span suppressHydrationWarning>{mounted ? formatINR(scoreObj.baseline_score) : '...'}</span>
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
