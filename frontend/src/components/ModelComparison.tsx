import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { ChevronUp, ChevronDown, CheckCircle2, Info } from 'lucide-react';
import styles from './ModelComparison.module.css';
import { formatINR } from '@/lib/formatINR';

export default function ModelComparison() {
  const [isOpen, setIsOpen] = useState(false);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch data only when first opened
  useEffect(() => {
    if (isOpen && !data && !loading && !error) {
      setLoading(true);
      fetch('http://localhost:8000/api/model-comparison')
        .then(res => {
          if (!res.ok) throw new Error('Failed to fetch comparison data');
          return res.json();
        })
        .then(json => {
          setData(json);
          setLoading(false);
        })
        .catch(err => {
          setError(err.message);
          setLoading(false);
        });
    }
  }, [isOpen, data, loading, error]);

  const toggleOpen = () => setIsOpen(!isOpen);

  return (
    <div className={`${styles.wrapper} ${isOpen ? styles.open : ''}`}>
      <button className={styles.toggleBtn} onClick={toggleOpen}>
        <div className={styles.toggleContent}>
          {isOpen ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
          <span>Model Comparison</span>
          {!isOpen && <span className={styles.badge}>4 Models</span>}
        </div>
      </button>

      {isOpen && (
        <div className={styles.content}>
          <div className={styles.header}>
            <h2>Leave-One-Out Cross Validation Results (n=50)</h2>
            <button className={styles.closeBtn} onClick={toggleOpen}>Close</button>
          </div>

          {loading && <div className={styles.loading}>Loading comparison data...</div>}
          {error && <div className={styles.error}>{error}</div>}

          {data && (
            <div className={styles.grid}>
              {/* Left Column: Chart & Table */}
              <div className={styles.leftCol}>
                <div className={styles.chartWrapper}>
                  <h3>R² Score by Model</h3>
                  <div style={{ height: 200, width: '100%', marginTop: 10 }}>
                    <ResponsiveContainer>
                      <BarChart data={data.models} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" domain={[0, 1]} stroke="#64748b" />
                        <YAxis type="category" dataKey="model" stroke="#94a3b8" width={90} />
                        <Tooltip
                          cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)' }}
                          formatter={(val: number) => val.toFixed(4)}
                        />
                        <Bar dataKey="r2" radius={[0, 4, 4, 0]} barSize={24}>
                          {data.models.map((entry: any) => (
                            <Cell key={entry.model} fill={entry.model === data.note.production_model ? '#f0abfc' : '#3b82f6'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className={styles.tableWrapper}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>R²</th>
                        <th>Min Error</th>
                        <th>Med Error</th>
                        <th>Max Error</th>
                        <th>Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.models.map((m: any) => (
                        <tr key={m.model} className={m.model === data.note.production_model ? styles.prodRow : ''}>
                          <td style={{ fontWeight: 500 }}>{m.model}</td>
                          <td>{m.r2.toFixed(4)}</td>
                          <td>{formatINR(m.mae_min)}</td>
                          <td>{formatINR(m.mae_median)}</td>
                          <td>{formatINR(m.mae_max)}</td>
                          <td>{formatINR(m.mae_std)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Right Column: Interpretation */}
              <div className={styles.rightCol}>
                <div className={styles.prodCard}>
                  <div className={styles.prodHeader}>
                    <CheckCircle2 size={18} color="#f0abfc" />
                    <h3>Production Model: {data.note.production_model}</h3>
                  </div>
                  <p>{data.note.selection_reason}</p>
                </div>

                <div className={styles.noteCard}>
                  <div className={styles.noteHeader}>
                    <Info size={16} color="#60a5fa" />
                    <h4>Statistical Caveat (n=50)</h4>
                  </div>
                  <p>{data.note.n_samples_caveat}</p>
                </div>

                <div className={styles.noteCard}>
                  <div className={styles.noteHeader}>
                    <Info size={16} color="#60a5fa" />
                    <h4>Structural Target Match</h4>
                  </div>
                  <p>{data.note.linear_structure_note}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
