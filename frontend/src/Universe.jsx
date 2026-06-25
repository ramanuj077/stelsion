import React, { useState, useEffect, useRef } from 'react';
import { 
  Globe, Compass, Eye, Activity, ShieldCheck, Gamepad2, Award, 
  Layers, Database, Camera, Rocket, Radio, BookOpen, Clock, Sun,
  Sliders, Star, Zap, Cpu, RefreshCw, ChevronRight, ChevronLeft, Volume2, Info
} from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

// --- ENCYCLOPEDIA DATA ---
const PLANET_ENCYCLOPEDIA = [
  {
    name: "STELSION-1b",
    mass: "1.25 x Earth",
    radius: "1.12 x Earth",
    temp: "288 K (Habitable)",
    dist: "320 Light Years",
    host: "G-Type Solar Twin",
    fact: "Atmospheric scan suggests high silicate concentrations with potential water vapour bands.",
    desc: "A rocky planet located squarely within the liquid-water habitable zone of its parent solar twin. Detected via automated wavelet-attention filters.",
    habitability: 89,
    composition: "Silicate crust, nickel-iron core"
  },
  {
    name: "Kepler-22b",
    mass: "8.3 x Earth",
    radius: "2.40 x Earth",
    temp: "250 K (Mild Cold)",
    dist: "620 Light Years",
    host: "G-Type Star",
    fact: "The first planet discovered by Kepler in the habitable zone of a Sun-like star.",
    desc: "Often classified as a water world or gas dwarf, Kepler-22b has a significant atmosphere and might possess a global ocean.",
    habitability: 68,
    composition: "Volatiles, high water content, gas envelope"
  },
  {
    name: "TRAPPIST-1e",
    mass: "0.69 x Earth",
    radius: "0.92 x Earth",
    temp: "251 K (Temperate)",
    dist: "40 Light Years",
    host: "M-Dwarf Red Star",
    fact: "Tidally locked to its host star, meaning one side experiences eternal day and the other eternal night.",
    desc: "One of the most promising rocky candidates. ESI metrics suggest a compact terrestrial iron core and water ice layers.",
    habitability: 95,
    composition: "Iron core, silicate mantle, superficial ice/water"
  }
];

const MUSEUM_ROOMS = [
  { id: "astronomy", name: "Hall of Astronomy", desc: "Interactive exhibits showing Keplerian orbital dynamics, stellar classification, and gravitational transit alignment angles." },
  { id: "missions", name: "Hall of Space Missions", desc: "Trace historical satellites from CoRoT and Kepler to TESS, JWST, and the upcoming Ariel spectroscopy mission." },
  { id: "exoplanets", name: "Hall of Exoplanets", desc: "Detailed 3D models comparing super-Earths, hot Jupiters, ocean worlds, and gaseous sub-Neptunes." },
  { id: "ai", name: "Hall of AI", desc: "Exhibits on deep learning, convolution filter scans, multi-head self-attention arrays, and neural confidence calibration." }
];

export default function Universe({ playClick, playSuccess, playNotification }) {
  // Tabs for the Universe operating system
  // 'space-map', 'explorer', 'museum', 'academy', 'telescope', 'detective', 'flight', 'achievements'
  const [activeTab, setActiveTab] = useState('space-map');

  // Space Map & Physics simulation states
  const [timePeriod, setTimePeriod] = useState(2026); // Timeline slider
  const [selectedPlanet, setSelectedPlanet] = useState(PLANET_ENCYCLOPEDIA[0]);
  const [activeMuseumRoom, setActiveMuseumRoom] = useState(MUSEUM_ROOMS[0]);

  // Cosmic events simulation trigger states
  const [cosmicEvent, setCosmicEvent] = useState('');
  const [eventProgress, setEventProgress] = useState(0);

  // Game 1: Exoplanet Detective Game
  const [detectiveStar, setDetectiveStar] = useState({ name: "KIC-9023", noise: "High", period: 12 });
  const [detectiveGuess, setDetectiveGuess] = useState('');
  const [detectiveScore, setDetectiveScore] = useState(0);

  // Game 2: Space Flight Simulator Game
  const [spacecraftPosition, setSpacecraftPosition] = useState({ x: 50, y: 50 });
  const [fuel, setFuel] = useState(100);
  const [scannedStars, setScannedStars] = useState(0);

  // Telescope Simulator States
  const [telescopeAngle, setTelescopeAngle] = useState(180);
  const [photonsCollected, setPhotonsCollected] = useState(0);
  const [isScanningTelescope, setIsScanningTelescope] = useState(false);

  // Photo Mode Settings
  const [bloomSetting, setBloomSetting] = useState(50);
  const [dofSetting, setDofSetting] = useState(30);

  // Trigger cosmic event simulation
  const triggerCosmicEvent = (type) => {
    if (playNotification) playNotification();
    setCosmicEvent(type);
    setEventProgress(0);
    const interval = setInterval(() => {
      setEventProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5;
      });
    }, 100);
  };

  // Detective Verification
  const makeDetectiveGuess = (guess) => {
    if (guess === 'planet') {
      if (playSuccess) playSuccess();
      setDetectiveGuess('CORRECT! High-confidence transit detected matching orbital criteria.');
      setDetectiveScore(prev => prev + 10);
    } else {
      if (playClick) playClick();
      setDetectiveGuess('FAILED. The signal features secondary eclipses indicating a binary star system.');
    }
  };

  // Spacecraft Flight Simulator Logic
  const moveSpacecraft = (dir) => {
    if (fuel <= 0) return;
    if (playClick) playClick();
    setSpacecraftPosition((prev) => {
      let nextX = prev.x;
      let nextY = prev.y;
      if (dir === 'up') nextY = Math.max(0, prev.y - 8);
      if (dir === 'down') nextY = Math.min(100, prev.y + 8);
      if (dir === 'left') nextX = Math.max(0, prev.x - 8);
      if (dir === 'right') nextX = Math.min(100, prev.x + 8);
      
      setFuel(f => Math.max(0, f - 2));
      // Randomly scan stars if coordinates align
      if (Math.random() > 0.6) {
        if (playSuccess) playSuccess();
        setScannedStars(s => s + 1);
      }
      return { x: nextX, y: nextY };
    });
  };

  // Telescope Scan simulator
  const startTelescopeScan = () => {
    if (playNotification) playNotification();
    setIsScanningTelescope(true);
    setPhotonsCollected(0);
    const interval = setInterval(() => {
      setPhotonsCollected((prev) => {
        if (prev >= 2000) {
          clearInterval(interval);
          setIsScanningTelescope(false);
          if (playSuccess) playSuccess();
          return 2000;
        }
        return prev + 100;
      });
    }, 150);
  };

  return (
    <div className="w-full min-h-screen bg-transparent text-[#f8fafc] flex flex-col font-sans relative overflow-hidden">
      
      {/* HUD Universe Header Tab selection bar */}
      <div className="flex flex-wrap gap-2 border-b border-cyan-500/10 pb-4 mb-8">
        {[
          { id: 'space-map', label: '1. Cosmic space Map', icon: Globe },
          { id: 'explorer', label: '2. Planet Explorer', icon: Database },
          { id: 'museum', label: '3. Space Museum', icon: BookOpen },
          { id: 'academy', label: '4. Academy Lessons', icon: Clock },
          { id: 'telescope', label: '5. Telescope simulator', icon: Sliders },
          { id: 'detective', label: '6. Detective Laboratory', icon: ShieldCheck },
          { id: 'flight', label: '7. Flight Exploration', icon: Rocket },
          { id: 'achievements', label: '8. Achievements & photo', icon: Award },
          { id: 'solar-system', label: '9. 3D Solar System', icon: Sun }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              if (playClick) playClick();
              setActiveTab(tab.id);
            }}
            className={`px-4 py-2.5 rounded-lg font-mono text-xs font-bold border flex items-center space-x-2 transition-all duration-300 ${
              activeTab === tab.id
                ? 'bg-cyan-500/10 border-cyan-500 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                : 'bg-white/5 border-transparent text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* ================= TAB 1: COSMIC SPACE MAP ================= */}
      {activeTab === 'space-map' && (
        <SolarSystemView />
      )}

      {/* ================= TAB 2: PLANET EXPLORER ENCYCLOPEDIA ================= */}
      {activeTab === 'explorer' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Planet list */}
          <div className="lg:col-span-4 space-y-4">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Encyclopedia Database</span>
            <h3 className="text-2xl font-bold text-white">Exoplanet Index</h3>

            <div className="space-y-3">
              {PLANET_ENCYCLOPEDIA.map((planet) => (
                <button
                  key={planet.name}
                  onClick={() => {
                    if (playClick) playClick();
                    setSelectedPlanet(planet);
                  }}
                  className={`w-full text-left p-4 rounded-lg border transition-all duration-200 ${
                    selectedPlanet.name === planet.name
                      ? 'border-cyan-500 bg-cyan-950/20 text-white'
                      : 'border-white/5 bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <div className="font-bold text-xs font-mono">{planet.name}</div>
                  <div className="text-[10px] text-gray-500 mt-1">Host Star: {planet.host} | Distance: {planet.dist}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Planet Details Card */}
          <div className="lg:col-span-8 glass-panel p-6 rounded-xl border border-cyan-500/10 flex flex-col justify-between">
            <div className="flex justify-between items-start border-b border-white/5 pb-4">
              <div>
                <h4 className="text-2xl font-extrabold text-white font-mono">{selectedPlanet.name} System</h4>
                <p className="text-xs text-gray-400 italic mt-1 font-light">{selectedPlanet.desc}</p>
              </div>
              <span className="text-xs font-mono text-cyan-400 bg-cyan-950/30 px-3 py-1 rounded border border-cyan-500/20">
                Habitability: {selectedPlanet.habitability}%
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs my-6">
              <div className="bg-black/40 p-3 rounded border border-white/5">
                <span className="text-[10px] text-gray-500 block">PLANETARY MASS</span>
                <span className="text-white font-bold">{selectedPlanet.mass}</span>
              </div>
              <div className="bg-black/40 p-3 rounded border border-white/5">
                <span className="text-[10px] text-gray-500 block">PLANETARY RADIUS</span>
                <span className="text-white font-bold">{selectedPlanet.radius}</span>
              </div>
              <div className="bg-black/40 p-3 rounded border border-white/5">
                <span className="text-[10px] text-gray-500 block">STABILITY TEMP</span>
                <span className="text-white font-bold">{selectedPlanet.temp}</span>
              </div>
              <div className="bg-black/40 p-3 rounded border border-white/5">
                <span className="text-[10px] text-gray-500 block">COMPOSITION</span>
                <span className="text-cyan-400 font-bold">{selectedPlanet.composition}</span>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-cyan-950/15 border border-cyan-500/20 font-mono text-xs">
              <span className="text-cyan-400 block font-bold mb-1">Key Scientific Observation:</span>
              <p className="text-gray-300 leading-normal">{selectedPlanet.fact}</p>
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 3: SPACE MUSEUM ================= */}
      {activeTab === 'museum' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Museum Room Selector */}
          <div className="lg:col-span-4 space-y-4">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Museum Exhibits</span>
            <h3 className="text-2xl font-bold text-white">Stelsion Science Museum</h3>

            <div className="space-y-3">
              {MUSEUM_ROOMS.map((room) => (
                <button
                  key={room.id}
                  onClick={() => setActiveMuseumRoom(room)}
                  className={`w-full text-left p-4 rounded-lg border transition-all duration-200 ${
                    activeMuseumRoom.id === room.id
                      ? 'border-cyan-500 bg-cyan-950/20 text-white'
                      : 'border-white/5 bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <div className="font-bold text-xs font-mono">{room.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Exhibition Display */}
          <div className="lg:col-span-8 glass-panel p-6 rounded-xl border border-cyan-500/10 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-mono text-cyan-400 block mb-2">ACTIVE EXHIBIT HALL</span>
              <h4 className="text-2xl font-bold text-white font-mono">{activeMuseumRoom.name}</h4>
              <p className="text-gray-300 font-light text-sm mt-4 leading-relaxed">
                {activeMuseumRoom.desc}
              </p>
            </div>

            <div className="bg-black/60 p-4 rounded-lg border border-white/5 font-mono text-xs space-y-2 mt-6">
              <span className="text-cyan-400 block font-bold">Interactive Lesson Content:</span>
              <p className="text-gray-400 leading-normal">
                Observe the diagrams and calibrate telemetry levels using the sliders located in the Preprocessing Labs tab to see results.
              </p>
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 4: ACADEMY LESSONS ================= */}
      {activeTab === 'academy' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { title: "What is Gravity?", desc: "Gravity shapes orbital trajectories. Kepler's third law links orbital distance and period.", icon: Sliders },
            { title: "How does Light Travel?", desc: "Photons travel light-years before capture. Ingress transit shadows decrease starlight amplitude.", icon: Star },
            { title: "What is Spectroscopy?", desc: "Atmosphere molecules absorb specific wavelengths, creating chemical absorption signatures.", icon: Globe }
          ].map((lesson) => (
            <div key={lesson.title} className="glass-panel p-5 rounded-xl border border-white/5 space-y-4">
              <div className="flex justify-between items-center border-b border-white/5 pb-2">
                <h4 className="font-bold text-sm font-mono text-white">{lesson.title}</h4>
                <lesson.icon className="w-4 h-4 text-cyan-400" />
              </div>
              <p className="text-xs text-gray-400 leading-relaxed font-light">{lesson.desc}</p>
            </div>
          ))}
        </div>
      )}

      {/* ================= TAB 5: TELESCOPE SIMULATOR ================= */}
      {activeTab === 'telescope' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Controls */}
          <div className="lg:col-span-4 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Telescope Simulator</span>
            <h3 className="text-2xl font-bold text-white">Target Star Photometer</h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between font-mono text-xs">
                  <span>Rotate Telescope Angle</span>
                  <span className="text-cyan-400">{telescopeAngle}°</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="360"
                  value={telescopeAngle}
                  onChange={(e) => setTelescopeAngle(Number(e.target.value))}
                  className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <button
                onClick={startTelescopeScan}
                disabled={isScanningTelescope}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-black font-mono font-bold text-xs rounded transition-all duration-200"
              >
                {isScanningTelescope ? "COLLECTING PHOTONS..." : "START SCANNING TARGET"}
              </button>
            </div>
          </div>

          {/* Telemetry Display */}
          <div className="lg:col-span-8 glass-panel p-6 rounded-xl border border-cyan-500/10 flex flex-col justify-between min-h-[300px]">
            <div>
              <span className="text-[10px] font-mono text-cyan-400">PHOTONS INGESTION STREAM</span>
              <div className="h-44 flex flex-col justify-center items-center">
                <Activity className={`w-12 h-12 text-cyan-400 ${isScanningTelescope ? 'animate-bounce' : 'animate-pulse'}`} />
                <div className="font-mono text-xs text-white mt-4">
                  Photons Ingested: <span className="text-cyan-400 font-bold">{photonsCollected} / 2000</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 6: DETECTIVE LABORATORY ================= */}
      {activeTab === 'detective' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Controls */}
          <div className="lg:col-span-5 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Game Mode // Exoplanet Detective</span>
            <h3 className="text-2xl font-bold text-white">Analyze Target System</h3>
            <p className="text-xs text-gray-400 leading-relaxed font-light">
              Stellar Target: {detectiveStar.name} presents a cyclic dip profile with strong flares. Decide whether this represents a transiting exoplanet or a binary star flare.
            </p>

            <div className="flex gap-2">
              <button
                onClick={() => makeDetectiveGuess('planet')}
                className="flex-1 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold text-xs rounded"
              >
                CONFIRM EXOPLANET
              </button>
              <button
                onClick={() => makeDetectiveGuess('binary')}
                className="flex-1 py-2 bg-red-600 hover:bg-red-500 text-white font-mono font-bold text-xs rounded"
              >
                CONFIRM BINARY STAR
              </button>
            </div>

            {detectiveGuess && (
              <div className="p-3 bg-black/40 rounded border border-white/5 font-mono text-[11px] text-gray-300">
                {detectiveGuess}
              </div>
            )}
          </div>

          {/* Scope details */}
          <div className="lg:col-span-7 glass-panel p-6 rounded-xl border border-white/5 flex flex-col justify-between">
            <span className="text-[10px] font-mono text-cyan-400">DETECTIVE SCORE INDEX</span>
            <div className="text-center py-8">
              <div className="text-4xl font-extrabold text-white font-mono">{detectiveScore}</div>
              <span className="text-xs text-gray-500 font-mono">SCIENTIST RANK POINTS</span>
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 7: SPACECRAFT FLIGHT SIMULATOR ================= */}
      {activeTab === 'flight' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Flight Controls */}
          <div className="lg:col-span-4 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Flight Game // Orbit Grid</span>
            <h3 className="text-2xl font-bold text-white">Space Flight Navigation</h3>
            
            <div className="grid grid-cols-3 gap-2 max-w-[180px] mx-auto text-center font-mono">
              <div />
              <button onClick={() => moveSpacecraft('up')} className="bg-white/5 hover:bg-white/10 p-2 border border-white/5 rounded text-xs">▲</button>
              <div />
              <button onClick={() => moveSpacecraft('left')} className="bg-white/5 hover:bg-white/10 p-2 border border-white/5 rounded text-xs">◀</button>
              <button onClick={() => moveSpacecraft('down')} className="bg-white/5 hover:bg-white/10 p-2 border border-white/5 rounded text-xs">▼</button>
              <button onClick={() => moveSpacecraft('right')} className="bg-white/5 hover:bg-white/10 p-2 border border-white/5 rounded text-xs">▶</button>
            </div>

            <div className="space-y-2 text-xs font-mono text-gray-400">
              <div className="flex justify-between"><span>Ship Fuel:</span> <span className="text-cyan-400 font-bold">{fuel}%</span></div>
              <div className="flex justify-between"><span>Stars Scanned:</span> <span className="text-white">{scannedStars}</span></div>
            </div>
          </div>

          {/* Visual Grid Radar */}
          <div className="lg:col-span-8 glass-panel p-6 rounded-xl border border-cyan-500/10 min-h-[300px] relative flex items-center justify-center">
            <div className="absolute inset-0 bg-grid-lines opacity-10" />
            
            {/* The Spacecraft */}
            <div 
              className="absolute w-6 h-6 rounded-full bg-cyan-500 border border-cyan-300 flex items-center justify-center transition-all duration-100 shadow-[0_0_15px_rgba(6,182,212,0.8)]"
              style={{
                left: `${spacecraftPosition.x}%`,
                top: `${spacecraftPosition.y}%`
              }}
            >
              <Rocket className="w-3.5 h-3.5 text-black" />
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 8: ACHIEVEMENTS & PHOTO MODE ================= */}
      {activeTab === 'achievements' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 text-left">
          
          {/* Achievements list */}
          <div className="glass-panel p-6 rounded-xl border border-white/5 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase block font-bold">Explorer Badges</span>
            <h4 className="text-lg font-bold text-white font-mono">Mission Achievements</h4>

            <div className="space-y-3 font-mono text-xs">
              <div className="flex justify-between items-center bg-black/40 p-3 rounded border border-white/5">
                <div>
                  <span className="text-white block font-bold">Exoplanet Hunter Badge</span>
                  <span className="text-[10px] text-gray-500">Discover your first confirmed planet using STELSION.</span>
                </div>
                <Check className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="flex justify-between items-center bg-black/40 p-3 rounded border border-white/5">
                <div>
                  <span className="text-white block font-bold">Signal Preprocessor Badge</span>
                  <span className="text-[10px] text-gray-500">Run Wavelet Denoising filters at level 8.</span>
                </div>
                <Check className="w-4 h-4 text-emerald-400" />
              </div>
            </div>
          </div>

          {/* Photo Mode settings */}
          <div className="glass-panel p-6 rounded-xl border border-white/5 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase block font-bold">Cinematic Capture</span>
            <h4 className="text-lg font-bold text-white font-mono">Poster Photo Settings</h4>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between font-mono text-xs">
                  <span>Bloom intensity</span>
                  <span>{bloomSetting}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={bloomSetting}
                  onChange={(e) => setBloomSetting(Number(e.target.value))}
                  className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between font-mono text-xs">
                  <span>Depth of Field (DOF)</span>
                  <span>{dofSetting}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={dofSetting}
                  onChange={(e) => setDofSetting(Number(e.target.value))}
                  className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <button
                onClick={() => alert("Cinematic poster screenshot exported to system directory.")}
                className="w-full py-2.5 bg-cyan-500 hover:bg-cyan-400 text-black font-mono font-bold text-xs rounded transition-all duration-200"
              >
                EXPORT CINEMATIC WALLPAPER
              </button>
            </div>
          </div>

        </div>
      )}

      {/* ================= TAB 9: INTERACTIVE GALAXY GRID ================= */}
      {activeTab === 'solar-system' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left panel: Galaxy Map details */}
          <div className="lg:col-span-4 space-y-6">
            <span className="text-cyan-400 font-mono text-xs uppercase tracking-widest block font-bold">Universe Explorer // Galaxy Map</span>
            <h3 className="text-3xl font-extrabold text-white font-mono">Milky Way Galaxy Grid</h3>
            <p className="text-gray-400 font-light text-sm">
              Freely trace positions across solar systems. Travel through timeline periods using the slider to observe spatial evolution.
            </p>

            <div className="space-y-2">
              <div className="flex justify-between font-mono text-xs text-gray-400">
                <span>Select Space Era</span>
                <span className="text-cyan-400 font-bold">{timePeriod} AD</span>
              </div>
              <input
                type="range"
                min="1610"
                max="2026"
                value={timePeriod}
                onChange={(e) => setTimePeriod(Number(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
            </div>

            {/* Trigger cosmic event simulations */}
            <div className="glass-panel p-4 rounded-lg border border-white/5 space-y-3 font-mono text-xs">
              <span className="text-cyan-400 block font-bold">Simulate Cosmic Events</span>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={() => triggerCosmicEvent('supernova')} className="bg-white/5 hover:bg-white/10 p-2 rounded border border-white/5 text-[10px]">Supernova Explosion</button>
                <button onClick={() => triggerCosmicEvent('blackhole')} className="bg-white/5 hover:bg-white/10 p-2 rounded border border-white/5 text-[10px]">Gravitational Lensing</button>
              </div>
            </div>
          </div>

          {/* Right panel: 3D interactive Orbit Grid */}
          <div className="lg:col-span-8 glass-panel p-6 rounded-xl border border-cyan-500/10 min-h-[400px] flex flex-col justify-between relative overflow-hidden">
            <div className="absolute inset-0 bg-grid-lines opacity-10" />
            
            <div className="flex justify-between items-center text-xs font-mono text-cyan-400 z-10">
              <span>TARGET COORDINATES SYSTEM</span>
              <span>ERA STATUS: {timePeriod < 1990 ? 'PRE-DISCOVERY ERA' : 'MODERN DEEP SPACE SURVEY'}</span>
            </div>

            {/* Orbit animation rendering */}
            <div className="w-full h-64 flex items-center justify-center relative py-6">
              
              {/* Giant Sun */}
              <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-yellow-400 via-orange-500 to-red-600 shadow-[0_0_50px_rgba(234,179,8,0.5)] relative flex items-center justify-center">
                
                {/* Simulated Supernova particles */}
                {cosmicEvent === 'supernova' && (
                  <div 
                    className="absolute w-40 h-40 rounded-full bg-cyan-500/30 border border-cyan-400 animate-ping"
                    style={{ scale: `${eventProgress}%` }}
                  />
                )}
                {/* Black hole lensing */}
                {cosmicEvent === 'blackhole' && (
                  <div className="absolute w-36 h-36 border border-white/20 rounded-full animate-spin" />
                )}
              </div>

              {/* Orbiting circles */}
              <div className="absolute w-44 h-[60px] border border-white/10 rounded-full rotate-12" />
              <div className="absolute w-60 h-[100px] border border-dashed border-white/5 rounded-full -rotate-12" />
            </div>

            <div className="font-mono text-[10px] text-gray-500 z-10 text-center">
              DRAG OR SELECT RADIAL VECTORS TO ROTATE GALAXY PATHS
            </div>
          </div>

        </div>
      )}

    </div>
  );
}

const SOLAR_PLANETS = [
  { name: "Mercury", dist: "0.39 AU", period: "88 Days", temp: "440 K", color: "#9e9e9e", size: 4.5, orbitRadius: 50, speed: 0.04, desc: "The smallest planet in our solar system and closest to the Sun." },
  { name: "Venus", dist: "0.72 AU", period: "225 Days", temp: "737 K", color: "#e5a93b", size: 6.5, orbitRadius: 75, speed: 0.025, desc: "Second planet from the Sun. It has a thick, toxic atmosphere." },
  { name: "Earth", dist: "1.00 AU", period: "365 Days", temp: "288 K", color: "#2d7dd2", size: 7, orbitRadius: 105, speed: 0.018, desc: "Our home planet. The only place in the universe known to harbor life." },
  { name: "Mars", dist: "1.52 AU", period: "687 Days", temp: "210 K", color: "#d24d2d", size: 5.5, orbitRadius: 140, speed: 0.014, desc: "A dusty, cold, desert world with a very thin atmosphere." },
  { name: "Jupiter", dist: "5.20 AU", period: "12 Years", temp: "165 K", color: "#dfc19c", size: 14, orbitRadius: 185, speed: 0.008, desc: "The largest planet in our solar system, more than twice as massive as all others combined." },
  { name: "Saturn", dist: "9.58 AU", period: "29 Years", temp: "134 K", color: "#e5c158", size: 11, orbitRadius: 235, speed: 0.005, hasRings: true, desc: "Adorned with a dazzling, complex system of icy rings." },
  { name: "Uranus", dist: "19.2 AU", period: "84 Years", temp: "76 K", color: "#a5f3fc", size: 8, orbitRadius: 285, speed: 0.003, desc: "An ice giant with a unique sideways tilt of nearly 90 degrees." },
  { name: "Neptune", dist: "30.1 AU", period: "165 Years", temp: "72 K", color: "#4f46e5", size: 7.5, orbitRadius: 335, speed: 0.002, desc: "The most distant solar planet. Dark, cold, and whipped by supersonic winds." }
];

function SolarSystemView() {
  const canvasRef = useRef(null);
  const [yaw, setYaw] = useState(0.4);
  const [pitch, setPitch] = useState(0.6);
  const [zoom, setZoom] = useState(1.1);
  const [speed, setSpeed] = useState(1.0);
  const [showOrbits, setShowOrbits] = useState(true);
  const [selectedPlanet, setSelectedPlanet] = useState(SOLAR_PLANETS[2]);

  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let t = 0;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      ctx.fillStyle = '#020206';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(6, 182, 212, 0.03)';
      ctx.lineWidth = 1;
      for (let r = 50; r < 400; r += 50) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, r * zoom, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, centerY); ctx.lineTo(width, centerY);
      ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height);
      ctx.stroke();

      const project = (x, y, z) => {
        const x1 = x * Math.cos(yaw) - z * Math.sin(yaw);
        const z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
        const y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
        const z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
        const D = 600;
        const perspective = D / (D + z2);
        const scale = zoom * perspective;
        return {
          x: centerX + x1 * scale,
          y: centerY + y2 * scale,
          zDepth: z2,
          rScale: scale
        };
      };

      const objects = [];

      objects.push({
        type: 'sun',
        x: 0, y: 0, z: 0,
        size: 26,
        color: '#ff9800',
        projected: project(0, 0, 0)
      });

      SOLAR_PLANETS.forEach((p) => {
        const angle = t * p.speed * speed;
        const planetX = p.orbitRadius * Math.cos(angle);
        const planetZ = p.orbitRadius * Math.sin(angle);
        const proj = project(planetX, 0, planetZ);

        objects.push({
          type: 'planet',
          name: p.name,
          color: p.color,
          size: p.size,
          hasRings: p.hasRings,
          projected: proj,
          orbitRadius: p.orbitRadius,
          planetData: p
        });

        if (p.name === 'Earth') {
          const moonAngle = t * 0.25;
          const moonX = planetX + 15 * Math.cos(moonAngle);
          const moonZ = planetZ + 15 * Math.sin(moonAngle);
          const moonProj = project(moonX, 0, moonZ);
          objects.push({
            type: 'moon',
            color: '#b0bec5',
            size: 1.8,
            projected: moonProj
          });
        }
      });

      objects.sort((a, b) => b.projected.zDepth - a.projected.zDepth);

      if (showOrbits) {
        ctx.lineWidth = 1.2;
        SOLAR_PLANETS.forEach((p) => {
          ctx.beginPath();
          ctx.strokeStyle = selectedPlanet?.name === p.name ? 'rgba(6, 182, 212, 0.45)' : 'rgba(255, 255, 255, 0.08)';
          for (let i = 0; i <= 72; i++) {
            const theta = (i / 72) * Math.PI * 2;
            const ox = p.orbitRadius * Math.cos(theta);
            const oz = p.orbitRadius * Math.sin(theta);
            const oproj = project(ox, 0, oz);
            if (i === 0) {
              ctx.moveTo(oproj.x, oproj.y);
            } else {
              ctx.lineTo(oproj.x, oproj.y);
            }
          }
          ctx.stroke();
        });
      }

      objects.forEach((obj) => {
        const { x, y, rScale } = obj.projected;

        if (obj.type === 'sun') {
          const glowRad = obj.size * 2 * rScale;
          const sunGlow = ctx.createRadialGradient(x, y, 2 * rScale, x, y, glowRad);
          sunGlow.addColorStop(0, '#ffffff');
          sunGlow.addColorStop(0.2, '#ffea00');
          sunGlow.addColorStop(0.5, 'rgba(244,143,177,0.3)');
          sunGlow.addColorStop(1, 'rgba(230,81,0,0)');
          ctx.fillStyle = sunGlow;
          ctx.beginPath();
          ctx.arc(x, y, glowRad, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = '#ffeb3b';
          ctx.beginPath();
          ctx.arc(x, y, obj.size * rScale, 0, Math.PI * 2);
          ctx.fill();
        } 
        else if (obj.type === 'planet') {
          const size = obj.size * rScale;

          if (obj.hasRings) {
            ctx.strokeStyle = 'rgba(229,193,88,0.5)';
            ctx.lineWidth = 3.5 * rScale;
            ctx.beginPath();
            for (let i = 0; i <= 36; i++) {
              const theta = (i / 36) * Math.PI * 2;
              const rx = (obj.size * 1.9) * Math.cos(theta);
              const rz = (obj.size * 1.9) * Math.sin(theta);
              const pAngle = t * obj.planetData.speed * speed;
              const planetX = obj.orbitRadius * Math.cos(pAngle);
              const planetZ = obj.orbitRadius * Math.sin(pAngle);
              const ringProj = project(planetX + rx, 0, planetZ + rz);
              if (i === 0) {
                ctx.moveTo(ringProj.x, ringProj.y);
              } else {
                ctx.lineTo(ringProj.x, ringProj.y);
              }
            }
            ctx.stroke();
          }

          const grad = ctx.createRadialGradient(x - size/3, y - size/3, size/10, x, y, size);
          grad.addColorStop(0, '#ffffff');
          grad.addColorStop(0.2, obj.color);
          grad.addColorStop(0.8, 'rgba(0,0,0,0.85)');
          grad.addColorStop(1, 'rgba(0,0,0,0.98)');
          
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fill();

          if (selectedPlanet?.name === obj.name) {
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(x, y, size + 6, 0, Math.PI * 2);
            ctx.stroke();
            
            ctx.fillStyle = '#06b6d4';
            ctx.font = 'bold 9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(obj.name.toUpperCase(), x, y - size - 12);
          } else {
            ctx.fillStyle = 'rgba(255,255,255,0.45)';
            ctx.font = '8px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(obj.name, x, y - size - 6);
          }
        } 
        else if (obj.type === 'moon') {
          ctx.fillStyle = obj.color;
          ctx.beginPath();
          ctx.arc(x, y, obj.size * rScale, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      t += 0.5;
      animationFrameId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [yaw, pitch, zoom, speed, showOrbits, selectedPlanet]);

  const handleMouseDown = (e) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    dragStart.current = { x: e.clientX, y: e.clientY };
    setYaw((prev) => prev + dx * 0.007);
    setPitch((prev) => Math.max(0.1, Math.min(Math.PI / 2 - 0.05, prev - dy * 0.007)));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 text-left">
      <div className="lg:col-span-8 flex flex-col space-y-4">
        <div className="flex justify-between items-center bg-black/40 p-4 rounded-xl border border-white/5 font-mono text-xs">
          <div>
            <span className="text-cyan-400 block font-bold">MILKY WAY SYSTEM // SOL SYSTEM</span>
            <span className="text-[10px] text-gray-500">Drag viewport to rotate orbit paths. Use controls to zoom.</span>
          </div>
          <div className="flex space-x-3">
            <button 
              onClick={() => setShowOrbits(!showOrbits)}
              className={`px-3 py-1 rounded border transition-colors ${showOrbits ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/35' : 'bg-white/5 text-gray-400 border-transparent'}`}
            >
              ORBITS
            </button>
            <button 
              onClick={() => { setYaw(0.4); setPitch(0.6); setZoom(1.1); }}
              className="bg-white/5 hover:bg-white/10 px-3 py-1 rounded border border-white/5 text-white"
            >
              RESET VIEW
            </button>
          </div>
        </div>

        <div className="relative w-full aspect-video bg-[#020206] rounded-2xl border border-cyan-500/15 overflow-hidden shadow-2xl">
          <canvas 
            ref={canvasRef}
            width={800}
            height={480}
            className="w-full h-full cursor-grab active:cursor-grabbing block"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={() => { isDragging.current = false; }}
            onMouseLeave={() => { isDragging.current = false; }}
          />

          <div className="absolute right-4 bottom-4 glass-panel px-4 py-3 rounded-xl border border-white/10 space-y-2.5 w-44 font-mono text-[10px]">
            <div className="space-y-1">
              <div className="flex justify-between text-gray-400"><span>Zoom:</span> <span className="text-white font-bold">{(zoom * 100).toFixed(0)}%</span></div>
              <input 
                type="range" min="0.4" max="2.5" step="0.05" value={zoom} 
                onChange={(e) => setZoom(parseFloat(e.target.value))}
                className="w-full h-0.5 bg-gray-800 rounded appearance-none cursor-pointer accent-cyan-400"
              />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-gray-400"><span>Orbit Speed:</span> <span className="text-white font-bold">{speed.toFixed(1)}x</span></div>
              <input 
                type="range" min="0.0" max="3.0" step="0.1" value={speed} 
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full h-0.5 bg-gray-800 rounded appearance-none cursor-pointer accent-cyan-400"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-4 flex flex-col justify-between space-y-6">
        <div className="glass-panel p-5 rounded-2xl border border-cyan-500/15 text-left flex-1 flex flex-col justify-between">
          <div>
            <span className="text-[10px] text-gray-500 font-mono block border-b border-white/5 pb-2">SOLAR TELEMETRY</span>
            <div className="mt-4 flex flex-col space-y-4">
              <div className="grid grid-cols-4 gap-1.5">
                {SOLAR_PLANETS.map((p) => (
                  <button 
                    key={p.name}
                    onClick={() => setSelectedPlanet(p)}
                    className={`py-1 rounded font-mono text-[9px] font-bold border transition-all ${
                      selectedPlanet?.name === p.name 
                        ? 'bg-cyan-500/10 border-cyan-500 text-cyan-400' 
                        : 'bg-white/5 border-transparent text-gray-400 hover:text-white'
                    }`}
                  >
                    {p.name.slice(0, 3).toUpperCase()}
                  </button>
                ))}
              </div>

              {selectedPlanet && (
                <div className="space-y-3.5 font-mono text-xs pt-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-3.5 h-3.5 rounded-full" style={{ backgroundColor: selectedPlanet.color }} />
                    <h3 className="text-lg font-bold text-white uppercase">{selectedPlanet.name}</h3>
                  </div>
                  <p className="text-[11px] text-gray-400 font-light leading-relaxed">{selectedPlanet.desc}</p>
                  
                  <div className="space-y-2 border-t border-white/5 pt-3">
                    <div className="flex justify-between"><span className="text-gray-500">Distance from Sun:</span> <span className="text-white font-bold">{selectedPlanet.dist}</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">Orbital Period:</span> <span className="text-white font-bold">{selectedPlanet.period}</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">Average Temp:</span> <span className="text-cyan-400 font-bold">{selectedPlanet.temp}</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="border-t border-white/5 pt-4 mt-6">
            <div className="flex items-center space-x-2 bg-purple-950/15 border border-purple-500/20 p-3.5 rounded-xl">
              <Sliders className="w-5 h-5 text-purple-400" />
              <div className="font-mono text-[10px]">
                <span className="text-white block font-bold">Stellar Physicist Info</span>
                <span className="text-gray-400">Exoplanet models calibrate based on this system as ground truth.</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
