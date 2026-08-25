'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { AlertCircle, MapPin, X } from 'lucide-react';
import styles from './page.module.css';

const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => <div className={styles.loadingOverlay}>Loading map...</div>,
});

import RankingSidebar from '@/components/RankingSidebar';
import SummaryPanel from '@/components/SummaryPanel';
import AddressSearch from '@/components/AddressSearch';
import ModelComparison from '@/components/ModelComparison';

const API_BASE = 'http://localhost:8000';

export default function Dashboard() {
  // --- Core data ---
  const [candidatesGeoJSON, setCandidatesGeoJSON] = useState<any>(null);
  const [scores, setScores] = useState<any[]>([]);
  const [selectedSiteId, setSelectedSiteId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // --- Custom scoring ---
  const [scoringMode, setScoringMode] = useState(false);
  const [customResult, setCustomResult] = useState<any | null>(null);
  const [customLoading, setCustomLoading] = useState(false);
  const [customError, setCustomError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [candidatesRes, scoresRes] = await Promise.all([
        fetch(`${API_BASE}/api/candidates`),
        fetch(`${API_BASE}/api/scores`),
      ]);
      if (!candidatesRes.ok || !scoresRes.ok) {
        throw new Error('Failed to fetch data from API');
      }
      const candidates = await candidatesRes.json();
      const scoresData = await scoresRes.json();
      setCandidatesGeoJSON(candidates);
      setScores(scoresData);
      if (scoresData.length > 0) {
        setSelectedSiteId(scoresData[0].site_id);
      }
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  // Score a custom point — called by map click OR address search
  const handleCustomScore = useCallback(async (lat: number, lon: number) => {
    setCustomLoading(true);
    setCustomError(null);
    setCustomResult(null);
    // Switch selection to custom so SHAP/summary panels show custom data
    setSelectedSiteId('custom');

    try {
      const res = await fetch(`${API_BASE}/api/score-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Scoring failed');
      }
      const data = await res.json();
      setCustomResult(data);
    } catch (err: any) {
      setCustomError(err.message || 'Could not reach scoring API');
      // Revert selection back to previous candidate on error
      setSelectedSiteId(prev => prev === 'custom' ? (scores[0]?.site_id ?? null) : prev);
    } finally {
      setCustomLoading(false);
    }
  }, [scores]);

  const exitScoringMode = () => {
    setScoringMode(false);
    setCustomResult(null);
    setCustomError(null);
    setCustomLoading(false);
    // Restore to top-ranked candidate
    if (scores.length > 0) setSelectedSiteId(scores[0].site_id);
  };

  if (loading) {
    return (
      <div className={styles.loadingOverlay}>
        <div className={styles.spinner}></div>
        <div className={styles.loadingText}>Initializing Geospatial Engine...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorState}>
        <AlertCircle size={48} className={styles.errorIcon} />
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={fetchData} className={styles.retryButton}>Retry Connection</button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.mapSection}>
        <MapComponent
          geoJson={candidatesGeoJSON}
          scores={scores}
          selectedSiteId={selectedSiteId}
          onSelectSite={(id) => { setSelectedSiteId(id); if (scoringMode) setScoringMode(false); }}
          scoringMode={scoringMode}
          onCustomScore={handleCustomScore}
          customResult={customResult}
          customLoading={customLoading}
        />

        {/* Mode toggle button */}
        <div className={styles.scoringToggle}>
          {!scoringMode ? (
            <button
              className={styles.scoreModeBtn}
              onClick={() => setScoringMode(true)}
              title="Click anywhere on the map to score a custom location"
            >
              <MapPin size={15} />
              Score a Location
            </button>
          ) : (
            <div className={styles.scoringActive}>
              <div className={styles.scoringActiveDot} />
              <span>Click map to score</span>
              <button className={styles.exitBtn} onClick={exitScoringMode} title="Exit scoring mode">
                <X size={14} />
              </button>
            </div>
          )}
        </div>

        {/* Address search — shown in scoring mode */}
        {scoringMode && (
          <div className={styles.addressSearchOverlay}>
            <AddressSearch
              onResult={(lat, lon) => handleCustomScore(lat, lon)}
              disabled={customLoading}
            />
          </div>
        )}

        {/* Custom scoring error toast */}
        {customError && (
          <div className={styles.errorToast}>
            <AlertCircle size={14} />
            <span>{customError}</span>
            <button onClick={() => setCustomError(null)} className={styles.toastClose}>
              <X size={12} />
            </button>
          </div>
        )}

        <div className={styles.summaryOverlay}>
          <SummaryPanel
            selectedSiteId={selectedSiteId}
            customResult={customResult}
            customLoading={customLoading}
          />
        </div>
      </div>

      <div className={styles.sidebarSection}>
        <RankingSidebar
          scores={scores}
          selectedSiteId={selectedSiteId}
          onSelectSite={(id) => { setSelectedSiteId(id); setScoringMode(false); }}
          customResult={customResult}
          customLoading={customLoading}
        />
      </div>
      
      <ModelComparison />
    </div>
  );
}
