'use client';

import { useState, useRef } from 'react';
import { Search, Loader2 } from 'lucide-react';
import styles from './AddressSearch.module.css';

interface AddressSearchProps {
  onResult: (lat: number, lon: number, label: string) => void;
  disabled?: boolean;
}

export default function AddressSearch({ onResult, disabled }: AddressSearchProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const geocode = async (q: string) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);

    try {
      // Bias results to Bengaluru viewbox; bounded=1 restricts to that box
      const url = new URL('https://nominatim.openstreetmap.org/search');
      url.searchParams.set('q', `${q}, Bengaluru`);
      url.searchParams.set('format', 'json');
      url.searchParams.set('limit', '1');
      url.searchParams.set('bounded', '1');
      // Bengaluru bounding box: west, south, east, north
      url.searchParams.set('viewbox', '77.35,12.75,77.85,13.20');

      const res = await fetch(url.toString(), {
        headers: {
          'User-Agent': 'geospatial-site-selection/1.0 (portfolio project)',
        },
      });

      if (!res.ok) throw new Error('Nominatim request failed');
      const results = await res.json();

      if (!results || results.length === 0) {
        setError('Address not found in Bengaluru');
        return;
      }

      const { lat, lon, display_name } = results[0];
      onResult(parseFloat(lat), parseFloat(lon), display_name);
      setQuery('');
    } catch (err: any) {
      setError('Could not reach geocoding service');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Respect Nominatim 1 req/sec policy via simple debounce
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => geocode(query), 300);
  };

  return (
    <div className={styles.wrapper}>
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.inputRow}>
          <input
            type="text"
            className={styles.input}
            placeholder="Search address in Bengaluru…"
            value={query}
            onChange={e => { setQuery(e.target.value); setError(null); }}
            disabled={disabled || loading}
          />
          <button
            type="submit"
            className={styles.button}
            disabled={disabled || loading || !query.trim()}
            aria-label="Search address"
          >
            {loading ? <Loader2 size={16} className={styles.spin} /> : <Search size={16} />}
          </button>
        </div>
        {error && <div className={styles.error}>{error}</div>}
      </form>
    </div>
  );
}
