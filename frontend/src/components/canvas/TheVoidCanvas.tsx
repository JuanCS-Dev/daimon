"use client";

import { Canvas } from "@react-three/fiber";
import TheVoid from "./TheVoid";

/**
 * TheVoidCanvas - Wrapper component for TheVoid with Canvas
 * Separated to avoid require() in dynamic import
 */
export default function TheVoidCanvas() {
  return (
    <div className="absolute inset-0 -z-10">
      <Canvas camera={{ position: [0, 0, 1] }} dpr={[1, 2]}>
        <TheVoid />
      </Canvas>
    </div>
  );
}
