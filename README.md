{
  "name": "axiomes-base-labs",
  "private": true,
  "version": "4.2.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "tsc --noEmit",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.344.0",
    "motion": "^12.4.7",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"

export const DEVELOPER_BIO = {
  name: "Axiomes Base Labs",
  title: "Consultant en Architecture des Systèmes Vivants & Noétique Appliquée, Conférencière & Conceptrice — Meta-Cognitiviste",
  handle: "contact@axiomes-baselabs.org",
  location: "Axiomes Base Labs // Réseau Décentralisé",
  clearance: "AGENCE & CONSULTANCE NOÉTIQUE",
  specialty: "Investigation Phénoménologique, Stratégie des Communs & Gouvernance Auto-organisée",
  summary: "Spécialiste de la convergence entre l'intelligence biologique et la cybernétique avancée. Concepteur d'architectures réseau résilientes à ultra-basse latence, d'interfaces synaptiques directes et de kernels sécurisés post-quantique.",
  tagline: "Méta-Cognition, Systèmes Vivants & Gouvernance Autopoïétique",
  pillars: [
    { title: "Investigation Phénoménologique", desc: "Analyse fine des structures d'expérience vécue, perception et cadres attentionnels." },
    { title: "Stratégie des Communs", desc: "Co-conception de ressources partagées, règles d'usage et régulation soutenant l'autonomie." },
    { title: "Gouvernance Auto-organisée", desc: "Architecture décisionnelle décentralisée, boucles de rétroaction et résilience systémique." },
    { title: "Noétique Appliquée", desc: "Ingénierie des processus cognitifs complexes, modèles mentaux et réseaux d'intelligence collective." }
  ],
  stats: [
    { label: "Pôles d'Investigation", value: "04" },
    { label: "Communs Modélisés", value: "28" },
    { label: "Cohérence Phénoménologique", value: "99.94%" },
    { label: "Réseaux Auto-organisés", value: "100%" }
  ]
};

import React, { useState } from 'react';
import { DisplayMode } from './types';
import { TerminalHeader } from './components/TerminalHeader';
import { CyberHUDView } from './components/CyberHUDView';
import { InteractiveTerminal } from './components/InteractiveTerminal';
import { MatrixRain } from './components/MatrixRain';
import { SynthwaveAudioControls } from './components/SynthwaveAudioControls';

export default function App() {
  const [displayMode, setDisplayMode] = useState<DisplayMode>('hud');
  const [matrixActive, setMatrixActive] = useState<boolean>(true);
  const [scanlinesActive, setScanlinesActive] = useState<boolean>(true);

  return (
    <div className={`min-h-screen bg-[#03070E] text-gray-200 font-sans relative selection:bg-[#00F0FF] selection:text-black ${scanlinesActive ? 'scanlines' : ''}`}>
      {matrixActive && <MatrixRain opacity={0.15} />}

      <TerminalHeader
        displayMode={displayMode}
        setDisplayMode={setDisplayMode}
        matrixActive={matrixActive}
        setMatrixActive={setMatrixActive}
        scanlinesActive={scanlinesActive}
        setScanlinesActive={setScanlinesActive}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 relative z-10">
        {displayMode === 'hud' ? (
          <CyberHUDView onSwitchToTerminal={() => setDisplayMode('cli')} />
        ) : (
          <InteractiveTerminal />
        )}
      </main>
    </div>
  );
}




  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.11",
    "typescript": "^5.5.3",
    "vite": "^5.4.2"
  }
}# berry
Mon serveur personnel


