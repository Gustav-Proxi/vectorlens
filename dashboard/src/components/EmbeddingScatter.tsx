'use client';

import React, { useRef, useState, useEffect } from 'react';
import { EmbeddingPoint, fetchEmbeddings } from '../lib/api';

interface EmbeddingScatterProps {
  sessionId: string;
  onPointClick?: (point: EmbeddingPoint) => void;
  className?: string;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  point: EmbeddingPoint | null;
}

interface ZoomPanState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export const EmbeddingScatter: React.FC<EmbeddingScatterProps> = ({
  sessionId,
  onPointClick,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<EmbeddingPoint[]>([]);
  const [explainedVariance, setExplainedVariance] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    point: null,
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoomPan, setZoomPan] = useState<ZoomPanState>({
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  });

  // Fetch embedding data
  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const response = await fetchEmbeddings(sessionId);
        setData(response.points);
        setExplainedVariance(response.explained_variance);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load embeddings');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [sessionId]);

  // Get unique types present in data
  const presentTypes = Array.from(new Set(data.map((p) => p.type)));

  // Color scheme by type
  const typeColors: Record<string, string> = {
    chunk: '#60a5fa', // blue
    graphrag_community: '#f59e0b', // amber
    cag_document: '#8b5cf6', // violet
    query: '#10b981', // emerald
    hallucination: '#ef4444', // red
  };

  const typeLabels: Record<string, string> = {
    chunk: 'Chunk',
    graphrag_community: 'GraphRAG Community',
    cag_document: 'CAG Document',
    query: 'Query',
    hallucination: 'Hallucination',
  };

  // SVG dimensions
  const width = 800;
  const height = 600;
  const padding = 40;
  const plotWidth = width - 2 * padding;

  // Map data coordinates [-1, 1] to SVG pixel space
  const dataToSvg = (dataCoord: number): number => {
    return padding + ((dataCoord + 1) / 2) * plotWidth;
  };

  // Calculate point size based on attribution score
  const getPointSize = (score: number): number => {
    return 4 + score * 8; // 4-12px range
  };

  // Create star polygon points (5-pointed star)
  const createStarPoints = (cx: number, cy: number, radius: number): string => {
    const points = [];
    for (let i = 0; i < 10; i++) {
      const angle = (i * Math.PI) / 5 - Math.PI / 2;
      const r = i % 2 === 0 ? radius : radius * 0.4;
      points.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
    }
    return points.map((p) => p.join(',')).join(' ');
  };

  // Create diamond (rotated square)
  const createDiamondPath = (cx: number, cy: number, size: number): string => {
    const offset = size / Math.sqrt(2);
    return `M${cx},${cy - offset} L${cx + offset},${cy} L${cx},${cy + offset} L${cx - offset},${cy} Z`;
  };

  // Handle point click
  const handlePointClick = (point: EmbeddingPoint) => {
    setSelectedId(point.id);
    if (onPointClick) {
      onPointClick(point);
    }
  };

  // Handle tooltip on hover
  const handlePointHover = (
    point: EmbeddingPoint,
    e: React.MouseEvent<SVGCircleElement | SVGPathElement | SVGPolygonElement>,
  ) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    const pageX = e.clientX;
    const pageY = e.clientY;

    setTooltip({
      visible: true,
      x: pageX,
      y: pageY,
      point,
    });
  };

  const handleMouseLeave = () => {
    setTooltip({ ...tooltip, visible: false });
  };

  // Handle scroll zoom
  const handleWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomPan((prev) => ({
      ...prev,
      scale: Math.max(1, Math.min(5, prev.scale * delta)),
    }));
  };

  // Handle pan drag
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    // Only pan on empty area (not on points)
    if ((e.target as SVGElement).tagName === 'svg') {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isDragging) {
      const dx = (e.clientX - dragStart.x) / zoomPan.scale;
      const dy = (e.clientY - dragStart.y) / zoomPan.scale;
      setZoomPan((prev) => ({
        ...prev,
        offsetX: prev.offsetX + dx,
        offsetY: prev.offsetY + dy,
      }));
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Render point based on type
  const renderPoint = (point: EmbeddingPoint) => {
    const svgX = dataToSvg(point.x) + zoomPan.offsetX;
    const svgY = dataToSvg(point.y) + zoomPan.offsetY;
    const size = getPointSize(point.attribution_score);
    const color = typeColors[point.type] || '#9ca3af';
    const opacity = point.caused_hallucination ? 1 : 0.6;
    const isSelected = selectedId === point.id;

    const commonProps = {
      key: `point-${point.id}`,
      onMouseEnter: (e: React.MouseEvent<any>) => handlePointHover(point, e),
      onMouseLeave: handleMouseLeave,
      onClick: () => handlePointClick(point),
      className: 'cursor-pointer transition-all',
      style: {
        filter: isSelected ? 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.6))' : 'none',
      },
    };

    if (point.type === 'chunk') {
      return (
        <circle
          cx={svgX}
          cy={svgY}
          r={size}
          fill={color}
          opacity={opacity}
          {...commonProps}
        />
      );
    } else if (point.type === 'graphrag_community') {
      return (
        <path
          d={createDiamondPath(svgX, svgY, size)}
          fill={color}
          opacity={opacity}
          {...commonProps}
        />
      );
    } else if (point.type === 'cag_document') {
      return (
        <rect
          x={svgX - size}
          y={svgY - size}
          width={size * 2}
          height={size * 2}
          fill={color}
          opacity={opacity}
          {...commonProps}
        />
      );
    } else if (point.type === 'query') {
      return (
        <polygon
          points={createStarPoints(svgX, svgY, size)}
          fill={color}
          opacity={opacity}
          {...commonProps}
        />
      );
    } else if (point.type === 'hallucination') {
      return (
        <g key={`point-${point.id}`}>
          <circle cx={svgX} cy={svgY} r={size} fill={color} opacity={opacity} />
          <circle
            cx={svgX}
            cy={svgY}
            r={size + 2}
            fill="none"
            stroke={color}
            strokeWidth={2}
            opacity={opacity}
            {...commonProps}
          />
        </g>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div
        className={`flex items-center justify-center ${className}`}
        style={{ width: '100%', height: '600px' }}
      >
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 bg-blue-500 rounded-full animate-pulse"></div>
          <p className="text-gray-400 text-sm">Loading embedding space...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={`flex items-center justify-center ${className}`}
        style={{ width: '100%', height: '600px' }}
      >
        <div className="text-center">
          <p className="text-red-500 text-sm font-medium">Error loading embeddings</p>
          <p className="text-gray-500 text-xs mt-2">{error}</p>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-zinc-900 rounded-lg border border-zinc-800 ${className}`}
        style={{ width: '100%', height: '600px' }}
      >
        <p className="text-gray-400 text-sm">Run a query to see embedding space</p>
      </div>
    );
  }

  return (
    <div className={`relative bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden ${className}`}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full cursor-grab active:cursor-grabbing bg-zinc-950"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          transform: `scale(${zoomPan.scale})`,
          transformOrigin: 'center',
        }}
      >
        {/* Grid background */}
        <defs>
          <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path
              d="M 50 0 L 0 0 0 50"
              fill="none"
              stroke="#27272a"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect width={width} height={height} fill="url(#grid)" />

        {/* Axes */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#52525b" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#52525b" strokeWidth="1" />

        {/* Axis labels */}
        <text x={width - padding - 20} y={height - padding + 20} fontSize="12" fill="#a1a1aa" textAnchor="end">
          PC1
        </text>
        <text x={padding - 20} y={padding + 15} fontSize="12" fill="#a1a1aa" textAnchor="end">
          PC2
        </text>

        {/* Data points */}
        {data.map((point) => renderPoint(point))}
      </svg>

      {/* Explained variance (top-right) */}
      {explainedVariance.length >= 2 && (
        <div className="absolute top-4 right-4 bg-zinc-800/80 backdrop-blur px-3 py-2 rounded text-xs text-gray-300 font-mono border border-zinc-700">
          PC1: {(explainedVariance[0] * 100).toFixed(1)}% | PC2: {(explainedVariance[1] * 100).toFixed(1)}%
        </div>
      )}

      {/* Legend (bottom-left) */}
      {presentTypes.length > 0 && (
        <div className="absolute bottom-4 left-4 bg-zinc-800/80 backdrop-blur rounded text-xs text-gray-300 border border-zinc-700 p-2 space-y-1">
          <p className="font-semibold text-gray-200 mb-2">Legend</p>
          {presentTypes.map((type) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: typeColors[type] || '#9ca3af' }}
              ></div>
              <span>{typeLabels[type] || type}</span>
            </div>
          ))}
        </div>
      )}

      {/* Tooltip */}
      {tooltip.visible && tooltip.point && (
        <div
          className="fixed bg-zinc-800/95 backdrop-blur border border-zinc-700 rounded px-3 py-2 text-xs text-gray-200 pointer-events-none max-w-xs z-50"
          style={{
            left: `${Math.min(tooltip.x + 10, window.innerWidth - 320)}px`,
            top: `${Math.max(tooltip.y - 60, 10)}px`,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          }}
        >
          <p className="font-semibold text-gray-100 mb-1">{tooltip.point.label}</p>
          <p className="text-gray-400 line-clamp-3 mb-2">{tooltip.point.text}</p>
          <div className="flex justify-between text-gray-500">
            <span>Attribution: {(tooltip.point.attribution_score * 100).toFixed(1)}%</span>
            {tooltip.point.caused_hallucination && <span className="text-red-400">Hallucination</span>}
          </div>
        </div>
      )}
    </div>
  );
};
