import { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF, Center } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

interface Props {
  meshUrl: string;
}

function Model({ url }: { url: string }) {
  // useGLTF caches by url and suspends until loaded -- the Suspense
  // boundary in MeshViewer below shows a fallback while this resolves.
  const { scene } = useGLTF(url);
  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.3, 8, 8]} />
      <meshBasicMaterial color="#E14B36" wireframe />
    </mesh>
  );
}

export default function MeshViewer({ meshUrl }: Props) {
  const controlsRef = useRef<OrbitControlsImpl>(null);

  function resetView() {
    controlsRef.current?.reset();
  }

  return (
    <div className="relative aspect-[3/4] w-full overflow-hidden rounded-2xl border border-ink/10 bg-white">
      <Canvas camera={{ position: [0, 1.0, 3.2], fov: 35 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 5, 4]} intensity={1.1} />
        <directionalLight position={[-3, 2, -4]} intensity={0.4} />
        <Suspense fallback={<LoadingFallback />}>
          <Model url={meshUrl} />
          <Environment preset="studio" />
        </Suspense>
        <OrbitControls
          ref={controlsRef}
          enablePan
          enableZoom
          enableRotate
          minDistance={1.2}
          maxDistance={8}
          target={[0, 0.9, 0]}
        />
      </Canvas>

      <button
        type="button"
        onClick={resetView}
        className="absolute bottom-3 right-3 rounded-full bg-ink/85 px-4 py-2 font-mono text-xs uppercase tracking-widest text-mist hover:bg-thread"
      >
        Reset view
      </button>
    </div>
  );
}
