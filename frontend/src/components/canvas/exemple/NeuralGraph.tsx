"use client";

import { useRef, useMemo, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { Line, Sphere } from "@react-three/drei";
import * as THREE from "three";
import BrainOverlay from "./BrainOverlay";

// Seeded random for deterministic results
function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function Node({ position, active }: { position: [number, number, number]; active: boolean }) {
  const ref = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ref.current && active) {
      const t = state.clock.getElapsedTime();
      ref.current.scale.setScalar(1 + Math.sin(t * 10) * 0.2);
    }
  });

  return (
    <Sphere ref={ref} args={[0.06, 16, 16]} position={position}>
      <meshStandardMaterial
        color={active ? "#f59e0b" : "#06b6d4"}
        emissive={active ? "#f59e0b" : "#06b6d4"}
        emissiveIntensity={active ? 3 : 1.2}
        roughness={0.2}
        metalness={0.8}
        toneMapped={false}
      />
    </Sphere>
  );
}

function Synapse({ start, end, active }: { start: [number, number, number]; end: [number, number, number]; active: boolean }) {
  return (
    <Line
      points={[start, end]}
      color={active ? "#f59e0b" : "#0e7490"}
      lineWidth={active ? 2.5 : 1.5}
      transparent
      opacity={active ? 0.9 : 0.4}
      toneMapped={false}
      dashed={false}
    />
  );
}

// Pre-generated deterministic node positions
const INITIAL_NODES: [number, number, number][] = Array.from({ length: 20 }, (_, i) => [
  (seededRandom(i * 3) - 0.5) * 4,
  (seededRandom(i * 3 + 1) - 0.5) * 4,
  (seededRandom(i * 3 + 2) - 0.5) * 2,
]);

// Pre-generated deterministic connections
const INITIAL_CONNECTIONS: [number, number][] = (() => {
  const c: [number, number][] = [];
  for (let i = 0; i < 20; i++) {
    for (let j = i + 1; j < 20; j++) {
      if (seededRandom(i * 20 + j) > 0.8) {
        c.push([i, j]);
      }
    }
  }
  return c;
})();

export default function NeuralGraph() {
  // Use pre-generated deterministic data
  const nodes = useMemo(() => INITIAL_NODES, []);
  const connections = useMemo(() => INITIAL_CONNECTIONS, []);

  return (
    <group>
      {/* Cérebro holográfico no centro - conexões parecem acontecer dentro dele */}
      <Suspense fallback={null}>
        <BrainOverlay />
      </Suspense>

      {/* Nós neurais orbitando ao redor do cérebro */}
      {nodes.map((pos, i) => (
        <Node key={i} position={pos} active={i % 5 === 0} />
      ))}

      {/* Sinapses conectando os nós */}
      {connections.map(([startIdx, endIdx], i) => (
        <Synapse
          key={`link-${i}`}
          start={nodes[startIdx]}
          end={nodes[endIdx]}
          active={i % 7 === 0}
        />
      ))}
    </group>
  );
}
