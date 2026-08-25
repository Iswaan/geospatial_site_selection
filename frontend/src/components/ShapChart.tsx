'use client';

import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import styles from './ShapChart.module.css';
import { formatINR } from '@/lib/formatINR';

interface ShapChartProps {
  selectedSiteId: string | null;
  customShapData?: { base_value: number; features: Record<string, number> } | null;
}

export default function ShapChart({ selectedSiteId, customShapData }: ShapChartProps) {
  const [data, setData] = useState<any[]>([]);
  const [baseValue, setBaseValue] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  // When a custom point is selected, use the already-fetched SHAP data
  useEffect(() => {
    if (selectedSiteId === 'custom') {
      if (customShapData) {
        setBaseValue(customShapData.base_value);
        const features = Object.entries(customShapData.features)
          .map(([key, value]) => ({
            name: key
              .replace('_1000m', ' (1km)')
              .replace('_3000m', ' (3km)')
              .replace(/_/g, ' ')
              .replace(/\b\w/g, l => l.toUpperCase()),
            value: Number(value),
            abs_value: Math.abs(Number(value)),
          }))
          .sort((a, b) => b.abs_value - a.abs_value)
          .slice(0, 6);
        setData(features);
        setError(null);
        setLoading(false);
      }
      return;
    }
    if (!selectedSiteId) return;

    let isMounted = true;

    const fetchShap = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`http://localhost:8000/api/shap/${selectedSiteId}`);
        if (!res.ok) throw new Error('Failed to fetch SHAP data');

        const json = await res.json();
        if (isMounted) {
          setBaseValue(json.base_value);

          // Convert features dict → array, sort by |value| desc, top 6
          const features = Object.entries(json.features)
            .map(([key, value]) => {
              let label = key
                .replace('_1000m', ' (1km)')
                .replace('_3000m', ' (3km)')
                .replace(/_/g, ' ');
              // Title case each word
              label = label.replace(/\b\w/g, l => l.toUpperCase());
              return {
                name: label,
                value: Number(value),
                abs_value: Math.abs(Number(value)),
              };
            })
            .sort((a, b) => b.abs_value - a.abs_value)
            .slice(0, 6);

          setData(features);
        }
      } catch (err: any) {
        if (isMounted) setError(err.message || 'Error loading SHAP');
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchShap();
    return () => { isMounted = false; };
  }, [selectedSiteId]);

  if (!selectedSiteId) return <div className={styles.loading}>Select a site to view its drivers</div>;
  if (loading) return <div className={styles.loading}>Analyzing spatial drivers...</div>;
  if (error) return <div className={styles.error}>{error}</div>;

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Local Feature Impacts</h3>
      <p style={{ fontSize: '14px', color: '#94a3b8', margin: '0 0 16px 0' }}>
        Base Value:{' '}
        <strong style={{ color: 'white' }} suppressHydrationWarning>
          {mounted && baseValue !== null ? formatINR(baseValue) : '...'}
        </strong>
      </p>

      <div className={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <XAxis type="number" hide />
            <YAxis
              dataKey="name"
              type="category"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              width={120}
            />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)' }}
              formatter={(value: any) => [formatINR(Number(value)), 'Impact']}
            />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
