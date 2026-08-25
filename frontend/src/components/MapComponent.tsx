'use client';

import { useMemo, useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents, Marker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { formatINR } from '@/lib/formatINR';

interface MapComponentProps {
  geoJson: any;
  scores: any[];
  selectedSiteId: string | null;
  onSelectSite: (id: string) => void;
  scoringMode: boolean;
  onCustomScore: (lat: number, lng: number) => void;
  customResult: any | null;
  customLoading: boolean;
}

// MapController: pan to selected candidate (sidebar → map sync)
function MapController({ selectedSiteId, siteData }: { selectedSiteId: string | null; siteData: any[] }) {
  const map = useMap();
  useEffect(() => {
    if (selectedSiteId && selectedSiteId !== 'custom' && siteData.length > 0) {
      const site = siteData.find(s => s.properties.site_id === selectedSiteId);
      if (site) {
        const [lng, lat] = site.geometry.coordinates;
        map.flyTo([lat, lng], 14, { duration: 0.5 });
      }
    }
  }, [selectedSiteId, siteData, map]);
  return null;
}

// ClickHandler: fires onCustomScore when scoring mode is active
function ClickHandler({ active, onCustomScore }: { active: boolean; onCustomScore: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      if (active) {
        onCustomScore(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

// Pulsing diamond marker for custom location
const customDivIcon = L.divIcon({
  className: '',
  html: `<div style="
    width: 20px; height: 20px;
    background: #f0abfc;
    border: 3px solid #ffffff;
    border-radius: 4px;
    transform: rotate(45deg);
    box-shadow: 0 0 0 3px rgba(240,171,252,0.4), 0 0 16px rgba(240,171,252,0.6);
    animation: customPulse 1.4s ease-in-out infinite;
  "></div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

const customLoadingDivIcon = L.divIcon({
  className: '',
  html: `<div style="
    width: 20px; height: 20px;
    background: #94a3b8;
    border: 3px solid #ffffff;
    border-radius: 4px;
    transform: rotate(45deg);
    opacity: 0.6;
    animation: customPulse 0.8s ease-in-out infinite;
  "></div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

export default function MapComponent({
  geoJson, scores, selectedSiteId, onSelectSite,
  scoringMode, onCustomScore, customResult, customLoading
}: MapComponentProps) {
  const [mounted, setMounted] = useState(false);
  const [pendingLatLng, setPendingLatLng] = useState<[number, number] | null>(null);

  useEffect(() => { setMounted(true); }, []);

  // Track where the user clicked so we can show the loading marker immediately
  const handleCustomScore = useCallback((lat: number, lng: number) => {
    setPendingLatLng([lat, lng]);
    onCustomScore(lat, lng);
  }, [onCustomScore]);

  // When custom result arrives, clear the pending state
  useEffect(() => {
    if (customResult && !customLoading) {
      setPendingLatLng(null);
    }
  }, [customResult, customLoading]);

  const siteData = useMemo(() => {
    if (!geoJson?.features || !scores) return [];
    return geoJson.features.map((f: any) => {
      const scoreObj = scores.find(s => s.site_id === f.properties.site_id);
      return {
        ...f,
        properties: {
          ...f.properties,
          predicted_score: scoreObj?.predicted_score ?? 0,
          rank: scoreObj?.rank ?? 999,
        },
      };
    });
  }, [geoJson, scores]);

  const { minScore, maxScore } = useMemo(() => {
    if (!scores || scores.length === 0) return { minScore: 0, maxScore: 1 };
    const vals = scores.map(s => s.predicted_score);
    return { minScore: Math.min(...vals), maxScore: Math.max(...vals) };
  }, [scores]);

  const getColor = (score: number) => {
    if (maxScore === minScore) return '#10b981';
    const ratio = Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
    const r = Math.round(239 + ratio * (16 - 239));
    const g = Math.round(68 + ratio * (185 - 68));
    const b = Math.round(68 + ratio * (129 - 68));
    return `rgb(${r}, ${g}, ${b})`;
  };

  if (siteData.length === 0) return null;

  const center: [number, number] = [12.9716, 77.5946];

  // Determine where to show the custom marker
  const customMarkerPos: [number, number] | null =
    customLoading && pendingLatLng
      ? pendingLatLng
      : customResult
      ? [customResult.lat, customResult.lon]
      : null;

  return (
    <div style={{ height: '100%', width: '100%', background: '#0f172a', cursor: scoringMode ? 'crosshair' : 'default' }}>
      {/* Inject custom pulse animation */}
      <style>{`
        @keyframes customPulse {
          0%, 100% { box-shadow: 0 0 0 3px rgba(240,171,252,0.4), 0 0 16px rgba(240,171,252,0.6); }
          50% { box-shadow: 0 0 0 6px rgba(240,171,252,0.1), 0 0 28px rgba(240,171,252,0.8); }
        }
      `}</style>
      <MapContainer
        center={center}
        zoom={12}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        <MapController selectedSiteId={selectedSiteId} siteData={siteData} />
        <ClickHandler active={scoringMode} onCustomScore={handleCustomScore} />

        {/* 50 pre-computed candidate markers */}
        {siteData.map((site: any) => {
          const [lng, lat] = site.geometry.coordinates;
          const isSelected = selectedSiteId === site.properties.site_id;
          const color = getColor(site.properties.predicted_score);

          return (
            <CircleMarker
              key={site.properties.site_id}
              center={[lat, lng]}
              radius={isSelected ? 12 : 8}
              pathOptions={{
                fillColor: color,
                fillOpacity: isSelected ? 1 : 0.7,
                color: isSelected ? '#ffffff' : '#000000',
                weight: isSelected ? 3 : 1,
              }}
              eventHandlers={{ click: () => onSelectSite(site.properties.site_id) }}
            >
              <Popup>
                <div style={{ fontFamily: 'Inter, sans-serif', textAlign: 'center', minWidth: 120 }}>
                  <strong>{site.properties.site_id}</strong>
                  <div>Rank: #{site.properties.rank}</div>
                  <div style={{ color: '#10b981', fontWeight: 600 }} suppressHydrationWarning>
                    {mounted ? formatINR(site.properties.predicted_score) : '...'}
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Custom location diamond marker */}
        {customMarkerPos && (
          <Marker
            position={customMarkerPos}
            icon={customLoading ? customLoadingDivIcon : customDivIcon}
          >
            {customResult && !customLoading && (
              <Popup>
                <div style={{ fontFamily: 'Inter, sans-serif', textAlign: 'center', minWidth: 140 }}>
                  <strong style={{ color: '#d946ef' }}>Custom Location</strong>
                  <div>Rank: #{customResult.rank} / {customResult.total_candidates}</div>
                  <div style={{ color: '#10b981', fontWeight: 600 }} suppressHydrationWarning>
                    {mounted ? formatINR(customResult.predicted_score) : '...'}
                  </div>
                </div>
              </Popup>
            )}
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}
