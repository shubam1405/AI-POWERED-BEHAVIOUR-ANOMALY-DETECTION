import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Shield, AlertTriangle, Activity, Database, Server, Cpu, Terminal, 
  Play, Pause, RotateCcw, SkipForward, Send, Layers, GitBranch, 
  FileText, BarChart2, Bell, CheckCircle, RefreshCw, LogOut, Info,
  MapPin, Clock, Search, ChevronRight, MessageSquare, Check, Zap, User
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, LineChart, Line, PieChart, Pie
} from 'recharts';

export default function App() {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Dashboard & global stats
  const [stats, setStats] = useState({
    totalSessions: 13015,
    normalSessions: 11447,
    anomalousSessions: 1568,
    highSeverityAlerts: 3,
    activeCampaigns: 180,
    modelHealth: 'Optimal',
    llmProvider: 'TemplateLLMClient'
  });

  // Simulated live cache
  const [simulatedSessions, setSimulatedSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState('SES-000614');
  const [sessionDetail, setSessionDetail] = useState(null);
  const [campaigns, setCampaigns] = useState([]);
  const [selectedCampaign, setSelectedCampaign] = useState(null);

  // Simulation pipeline state
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationStage, setSimulationStage] = useState(0);
  const [simulatedMetrics, setSimulatedMetrics] = useState({
    riskScore: 0,
    anomalyScore: 0,
    confidence: 0
  });

  // Multi-select attack state
  const [selectedBehaviours, setSelectedBehaviours] = useState([]);

  // Demo mode state
  const [demoActive, setDemoActive] = useState(false);
  const [demoIntervalId, setDemoIntervalId] = useState(null);
  const [demoIndex, setDemoIndex] = useState(0);

  // Chat Q&A state
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'AI Security Copilot initialized. Ask me anything about the active incident.' }
  ]);
  const [chatLoading, setChatLoading] = useState(false);

  // Toast notifications stack
  const [toasts, setToasts] = useState([]);

  // Auto scroll chat ref
  const chatEndRef = useRef(null);

  // Attack categories for simulation controls
  const attackTypes = [
    { label: 'Normal User', value: 'normal_user', category: 'Baseline' },
    { label: 'Brute Force', value: 'brute_force', category: 'Access' },
    { label: 'Credential Stuffing', value: 'credential_stuffing', category: 'Access' },
    { label: 'Device Spoofing', value: 'device_spoofing', category: 'Credential' },
    { label: 'Impossible Travel', value: 'impossible_travel', category: 'Network' },
    { label: 'Insider Drift', value: 'insider_drift', category: 'Insider' },
    { label: 'Lateral Movement', value: 'lateral_movement', category: 'Subnet' },
    { label: 'Privilege Escalation', value: 'privilege_escalation', category: 'Credential' },
    { label: 'Suspicious PowerShell', value: 'suspicious_powershell', category: 'Execution' },
    { label: 'Data Exfiltration', value: 'data_exfiltration', category: 'Egress' },
    { label: 'Low-Slow Exfiltration', value: 'low_slow_exfiltration', category: 'Egress' },
    { label: 'USB Data Theft', value: 'usb_data_theft', category: 'Insider' },
    { label: 'Off-hours Access', value: 'off_hours_access', category: 'Temporal' },
    { label: 'Malware Execution', value: 'malware_execution', category: 'Execution' }
  ];

  // Load baseline statistics and campaigns on mount
  useEffect(() => {
    fetchHealth();
    fetchCampaigns();
    // Load default session report
    loadSessionDetails('SES-000614');
  }, []);

  // Keep simulatedMetrics synchronized with sessionDetail when not actively simulating
  useEffect(() => {
    if (sessionDetail && !isSimulating) {
      setSimulatedMetrics({
        riskScore: sessionDetail.risk_score ?? 0,
        anomalyScore: sessionDetail.anomaly_score ?? 0,
        confidence: sessionDetail.confidence != null ? sessionDetail.confidence * 100 : 0
      });
    }
  }, [sessionDetail, isSimulating]);

  // Scroll to bottom of chat whenever messages update
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Demo Mode automation loop
  useEffect(() => {
    if (demoActive) {
      const runNextDemo = () => {
        const nextAttack = attackTypes[demoIndex % attackTypes.length];
        setDemoIndex(prev => prev + 1);
        addToast(`[Demo Mode] Launching simulation: ${nextAttack.label}`, 'info');
        triggerSingleSimulation(nextAttack.value);
      };
      
      // Run immediately on activate
      runNextDemo();
      
      const id = setInterval(runNextDemo, 25000);
      setDemoIntervalId(id);
    } else {
      if (demoIntervalId) {
        clearInterval(demoIntervalId);
        setDemoIntervalId(null);
      }
    }
    return () => {
      if (demoIntervalId) clearInterval(demoIntervalId);
    };
  }, [demoActive, demoIndex]);

  // Fetch functions
  const fetchHealth = async () => {
    try {
      const res = await axios.get('/health');
      setStats(prev => ({
        ...prev,
        llmProvider: res.data.provider,
        activeCampaigns: res.data.active_campaigns
      }));
    } catch (e) {
      console.error("Health check api call failed.", e);
    }
  };

  const fetchCampaigns = async () => {
    try {
      const res = await axios.get('/campaigns');
      setCampaigns(res.data);
      if (res.data.length > 0) {
        setSelectedCampaign(res.data[0]);
      }
    } catch (e) {
      console.error("Fetch campaigns api call failed.", e);
    }
  };

  const loadSessionDetails = async (sessionId) => {
    try {
      const [sessRes, repRes, recRes] = await Promise.all([
        axios.get(`/session/${sessionId}`),
        axios.get(`/report/${sessionId}`),
        axios.get(`/recommendations/${sessionId}`)
      ]);
      
      const details = {
        ...sessRes.data,
        report: repRes.data,
        recommendations: recRes.data
      };
      
      setSessionDetail(details);
      setSelectedSessionId(sessionId);
      
      // Seed default chatbot messages
      setChatMessages([
        { role: 'assistant', content: `AI Security Copilot initialized for session ${sessionId}. Mapped attack technique: ${details.mitre?.technique || 'None'}. Ask me questions such as "what should I investigate first?" or "why is the severity ${details.severity}?"` }
      ]);
    } catch (e) {
      console.error("Load session details failed.", e);
    }
  };

  // Toast alerts helper
  const addToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 6000);
  };

  // --- Shared pipeline animation helper ---
  const runPipelineAnimation = async (sessionData) => {
    setSimulatedSessions(prev => [sessionData, ...prev]);
    await delay(800); setSimulationStage(2);
    await delay(850); setSimulationStage(3);
    animateCounter('anomalyScore', 0, sessionData.anomaly_score, 800);
    await delay(900); setSimulationStage(4);
    animateCounter('confidence', 0, sessionData.confidence * 100, 800);
    await delay(850); setSimulationStage(5);
    animateCounter('riskScore', 0, sessionData.risk_score, 900);
    await delay(950); setSimulationStage(6);
    setSessionDetail(sessionData);
    setSelectedSessionId(sessionData.session_id);
    setStats(prev => ({
      ...prev,
      totalSessions: prev.totalSessions + 1,
      anomalousSessions: sessionData.attack_type !== 'Normal' ? prev.anomalousSessions + 1 : prev.anomalousSessions,
      normalSessions: sessionData.attack_type === 'Normal' ? prev.normalSessions + 1 : prev.normalSessions,
      highSeverityAlerts: ['High', 'Critical'].includes(sessionData.severity) ? prev.highSeverityAlerts + 1 : prev.highSeverityAlerts
    }));
    if (sessionData.copilot_context?.campaign_id) fetchCampaigns();
    if (['High', 'Critical'].includes(sessionData.severity)) {
      addToast(`ALERT: [${sessionData.severity}] ${sessionData.attack_type} detected in session ${sessionData.session_id}!`, 'error');
    } else {
      addToast(`Triage completed: ${sessionData.attack_type} (${sessionData.severity})`, 'success');
    }
    setChatMessages([
      { role: 'assistant', content: `AI Security Copilot context loaded for simulated session ${sessionData.session_id} (${sessionData.attack_type}). How can I assist you with this threat triage?` }
    ]);
  };

  // Multi-select: toggle a behaviour card
  const toggleBehaviour = (value) => {
    setSelectedBehaviours(prev =>
      prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]
    );
  };

  // Generate combined multi-behaviour attack
  const triggerCombinedSimulation = async () => {
    if (isSimulating || selectedBehaviours.length === 0) return;
    setIsSimulating(true);
    setSimulationStage(1);
    setSimulatedMetrics({ riskScore: 0, anomalyScore: 0, confidence: 0 });
    try {
      const res = await axios.post('/simulate', {
        behaviours: selectedBehaviours,
        employee_id: sessionDetail?.employee_id || null,
      });
      await runPipelineAnimation(res.data);
    } catch (e) {
      console.error('Combined simulation failed', e);
      addToast('Combined simulation pipeline encountered an error', 'error');
    } finally {
      setIsSimulating(false);
    }
  };

  // Legacy single-type simulation (for demo mode and Next Attack button)
  const triggerSingleSimulation = async (attackType) => {
    if (isSimulating) return;
    setIsSimulating(true);
    setSimulationStage(1);
    setSimulatedMetrics({ riskScore: 0, anomalyScore: 0, confidence: 0 });
    try {
      const res = await axios.post(`/simulate/${attackType}`);
      await runPipelineAnimation(res.data);
    } catch (e) {
      console.error('Simulation failed', e);
      addToast('Simulation pipeline encountered an error', 'error');
    } finally {
      setIsSimulating(false);
    }
  };

  // Keep backward-compat alias used by old demo mode
  const triggerSimulation = triggerSingleSimulation;

  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  // Animates counter values incrementally
  const animateCounter = (field, start, end, duration) => {
    const range = end - start;
    let current = start;
    const increment = range / (duration / 30);
    const stepTime = Math.abs(Math.floor(duration / (range || 1)));
    
    const timer = setInterval(() => {
      current += increment;
      if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
        clearInterval(timer);
        setSimulatedMetrics(prev => ({ ...prev, [field]: end }));
      } else {
        setSimulatedMetrics(prev => ({ ...prev, [field]: current }));
      }
    }, 30);
  };

  // Chat query execution
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading || !sessionDetail) return;

    const userQuestion = chatInput;
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userQuestion }]);
    setChatLoading(true);

    try {
      const res = await axios.post(`/chat/${selectedSessionId}`, { question: userQuestion });
      setChatMessages(prev => [...prev, { role: 'assistant', content: res.data.answer }]);
    } catch (err) {
      console.error(err);
      setChatMessages(prev => [...prev, { role: 'assistant', content: "Failed to communicate with the Copilot server." }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Reset local state statistics
  const resetDashboard = () => {
    setSimulatedSessions([]);
    fetchHealth();
    fetchCampaigns();
    loadSessionDetails('SES-000614');
    addToast("Dashboard state reset successfully.", "info");
  };

  // Global feature importance chart values (SHAP baseline indicators)
  const chartData = [
    { name: 'failed_login', shap: 0.255 },
    { name: 'idle_time', shap: 0.239 },
    { name: 'admin_access', shap: 0.216 },
    { name: 'process_pos', shap: 0.172 },
    { name: 'resource_ratio', shap: 0.141 },
    { name: 'powershell', shap: 0.131 },
    { name: 'upload_bytes', shap: 0.130 },
    { name: 'device_dev', shap: 0.127 }
  ];

  // Helper to format values
  const formatValue = (val, field) => {
    if (field === 'confidence') return `${Math.round(val)}%`;
    if (field === 'anomalyScore') return val.toFixed(4);
    return val.toFixed(1);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* Toast Alert Stack */}
      <div className="absolute top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
        {toasts.map(t => (
          <div 
            key={t.id} 
            className={`p-4 rounded-lg shadow-xl border flex items-start gap-3 transition-all duration-300 transform translate-x-0 ${
              t.type === 'error' ? 'bg-red-950/80 border-red-800 text-red-200' :
              t.type === 'success' ? 'bg-emerald-950/80 border-emerald-800 text-emerald-200' :
              'bg-slate-900/90 border-slate-700 text-slate-200'
            }`}
          >
            {t.type === 'error' ? <AlertTriangle className="h-5 w-5 text-red-400 shrink-0" /> :
             t.type === 'success' ? <CheckCircle className="h-5 w-5 text-emerald-400 shrink-0" /> :
             <Info className="h-5 w-5 text-sky-400 shrink-0" />}
            <span className="text-sm font-medium">{t.message}</span>
          </div>
        ))}
      </div>

      {/* Side Navigation Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-slate-900/50 flex flex-col justify-between shrink-0">
        <div>
          {/* Header */}
          <div className="p-6 border-b border-slate-800 flex items-center gap-3">
            <Shield className="h-8 w-8 text-emerald-400" />
            <div>
              <h1 className="font-bold text-lg tracking-wide text-slate-50">CYBER CAGE</h1>
              <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">XDR & UEBA SOC</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-1">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Layers },
              { id: 'timeline', label: 'Threat Timeline', icon: Clock },
              { id: 'campaigns', label: 'Campaign Graph', icon: GitBranch },
              { id: 'reports', label: 'Incident Reports', icon: FileText },
              { id: 'copilot', label: 'AI Copilot Chat', icon: MessageSquare },
              { id: 'analytics', label: 'Analytics', icon: BarChart2 }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab.id 
                      ? 'bg-slate-800 text-emerald-400 border-l-2 border-emerald-500' 
                      : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* System Settings & Provider Status */}
        <div className="p-6 border-t border-slate-800 bg-slate-950/40 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500 font-medium">LLM Provider</span>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-emerald-400 font-mono text-[10px]">{stats.llmProvider}</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500 font-medium">Model Status</span>
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              {stats.modelHealth}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="flex-1 flex flex-col overflow-hidden">
        
        {/* Top Navbar */}
        <header className="h-16 border-b border-slate-800 px-8 flex items-center justify-between bg-slate-900/10 shrink-0">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-bold capitalize text-slate-100">{activeTab} Interface</h2>
            {isSimulating && (
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium animate-pulse flex items-center gap-1">
                <RefreshCw className="h-3 w-3 animate-spin" />
                Processing pipeline...
              </span>
            )}
          </div>

          {/* Demo Controls */}
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setDemoActive(!demoActive)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                demoActive 
                  ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' 
                  : 'bg-slate-900 border-slate-800 hover:bg-slate-800 hover:text-slate-100'
              }`}
            >
              {demoActive ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
              {demoActive ? 'Pause Demo' : 'Start Demo'}
            </button>
            
            <button 
              onClick={() => {
                setDemoIndex(prev => prev + 1);
                const nextAttack = attackTypes[demoIndex % attackTypes.length];
                triggerSingleSimulation(nextAttack.value);
              }}
              disabled={isSimulating}
              className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-slate-100 disabled:opacity-50"
              title="Next Attack"
            >
              <SkipForward className="h-4 w-4" />
            </button>

            <button 
              onClick={resetDashboard}
              className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-slate-100"
              title="Reset State"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Dynamic Panels container */}
        <div className="flex-1 overflow-y-auto p-8">
          
          {/* Global statistics overview banner */}
          <section className="grid grid-cols-5 gap-4 mb-8">
            {[
              { label: 'Total Logs', value: stats.totalSessions, icon: Database, color: 'text-slate-400 bg-slate-900/50' },
              { label: 'Baseline normal', value: stats.normalSessions, icon: CheckCircle, color: 'text-emerald-400 bg-emerald-950/20' },
              { label: 'Anomalies Detected', value: stats.anomalousSessions, icon: Activity, color: 'text-amber-400 bg-amber-950/20' },
              { label: 'Severe Incidents', value: stats.highSeverityAlerts, icon: AlertTriangle, color: 'text-red-400 bg-red-950/20' },
              { label: 'Correlated Campaigns', value: stats.activeCampaigns, icon: GitBranch, color: 'text-purple-400 bg-purple-950/20' }
            ].map((card, idx) => {
              const Icon = card.icon;
              return (
                <div key={idx} className={`p-4 rounded-xl border border-slate-800/80 flex items-center gap-4 ${card.color}`}>
                  <div className="p-3 rounded-lg bg-slate-950/40 shrink-0">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <span className="text-xs text-slate-400 font-semibold">{card.label}</span>
                    <h3 className="text-xl font-bold tracking-tight mt-0.5 font-mono">{card.value}</h3>
                  </div>
                </div>
              );
            })}
          </section>

          {/* Tab Pages Routing */}
          
          {/* TABS 1: Dashboard Page */}
          {activeTab === 'dashboard' && (
            <div className="grid grid-cols-3 gap-8">
              
              {/* Simulation buttons panel */}
              <div className="col-span-2 space-y-6">
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-slate-100 flex items-center gap-2">
                      <Cpu className="h-5 w-5 text-emerald-400" />
                      Attack Simulation console
                    </h3>
                    <span className="text-xs text-slate-400">Trigger simulated payloads in real time</span>
                  </div>
                  
                  {/* Multi-select Attack Card Grid */}
                  <div className="grid grid-cols-3 gap-2.5">
                    {attackTypes.map(type => {
                      const isSelected = selectedBehaviours.includes(type.value);
                      return (
                        <button
                          key={type.value}
                          onClick={() => toggleBehaviour(type.value)}
                          disabled={isSimulating}
                          className={`py-3 px-4 rounded-xl border text-xs font-semibold tracking-wide text-left transition-all relative ${
                            isSimulating
                              ? 'bg-slate-900 border-slate-800 text-slate-500 cursor-not-allowed'
                              : isSelected
                                ? 'bg-emerald-950/40 border-emerald-500 text-emerald-300 shadow-[0_0_10px_rgba(52,211,153,0.15)]'
                                : 'bg-slate-900/60 border-slate-800 hover:border-slate-600 text-slate-300 hover:text-slate-100 hover:scale-[1.01]'
                          }`}
                        >
                          {isSelected && (
                            <span className="absolute top-1.5 right-1.5 h-4 w-4 rounded-full bg-emerald-500 flex items-center justify-center">
                              <Check className="h-2.5 w-2.5 text-slate-950" />
                            </span>
                          )}
                          <span className="text-[10px] text-slate-500 block uppercase mb-1 tracking-wider">{type.category}</span>
                          {type.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Selected behaviours chip strip */}
                  <div className="mt-2">
                    {selectedBehaviours.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {selectedBehaviours.map(val => {
                          const t = attackTypes.find(a => a.value === val);
                          return (
                            <span
                              key={val}
                              className="px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-[10px] font-semibold flex items-center gap-1.5 cursor-pointer hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-300"
                              onClick={() => toggleBehaviour(val)}
                              title="Click to deselect"
                            >
                              <Check className="h-2.5 w-2.5" />
                              {t?.label || val}
                            </span>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500 italic">Select one or more attack behaviours above, then click Generate Combined Attack.</p>
                    )}
                  </div>

                  {/* Generate Combined Attack button */}
                  <button
                    onClick={triggerCombinedSimulation}
                    disabled={isSimulating || selectedBehaviours.length === 0}
                    className={`w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${
                      selectedBehaviours.length === 0 || isSimulating
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-[0_0_20px_rgba(52,211,153,0.25)] hover:shadow-[0_0_30px_rgba(52,211,153,0.4)]'
                    }`}
                  >
                    {isSimulating
                      ? <><RefreshCw className="h-4 w-4 animate-spin" /> Running inference pipeline...</>
                      : <><Zap className="h-4 w-4" /> Generate Combined Attack ({selectedBehaviours.length} behaviour{selectedBehaviours.length !== 1 ? 's' : ''})</>}
                  </button>

                </div>

                {/* Pipeline visualizer panel */}
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>
                  
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <Layers className="h-5 w-5 text-emerald-400" />
                    Cyber Cage pipeline status
                  </h3>

                  <div className="flex flex-col gap-4 relative pl-8 border-l border-slate-800/80">
                    {[
                      { step: 1, label: 'Enterprise Logs Collection', desc: 'Sinks raw employee access and shell audits.' },
                      { step: 2, label: 'Feature Engineering', desc: 'Standardizes behavior baselines and sequential vectors.' },
                      { step: 3, label: 'GRU Autoencoder Reconstruction', desc: 'Rebuilds sequence sequences to score behavioral anomalies.' },
                      { step: 4, label: 'XGBoost Attack Classifier', desc: 'Maps indicators to MITRE signatures with class confidence.' },
                      { step: 5, label: 'SHAP Explainability Engine', desc: 'Quantifies positive and negative behavioral indicators.' },
                      { step: 6, label: 'AI Security Copilot Report', desc: 'Constructs incident summaries and 4-tier playbooks.' }
                    ].map(step => {
                      const isActive = simulationStage === step.step;
                      const isCompleted = simulationStage > step.step;
                      const isWaiting = simulationStage < step.step;

                      return (
                        <div key={step.step} className="relative flex items-center justify-between">
                          {/* Dot connector */}
                          <div className={`absolute -left-[41px] h-6 w-6 rounded-full border-2 flex items-center justify-center text-xs font-bold transition-all ${
                            isActive ? 'bg-slate-950 border-emerald-400 text-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.3)] scale-110' :
                            isCompleted ? 'bg-emerald-500/20 border-emerald-400 text-emerald-400' :
                            'bg-slate-950 border-slate-800 text-slate-500'
                          }`}>
                            {isCompleted ? '✓' : step.step}
                          </div>

                          <div className="ml-2">
                            <h4 className={`text-sm font-semibold transition-all ${isActive ? 'text-emerald-400' : isCompleted ? 'text-slate-200' : 'text-slate-500'}`}>
                              {step.label}
                            </h4>
                            <p className="text-xs text-slate-400 mt-0.5">{step.desc}</p>
                          </div>

                          {isActive && (
                            <span className="text-[10px] font-mono text-emerald-400 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 uppercase tracking-widest font-semibold animate-pulse">
                              Processing
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Sidebar result stats */}
              <div className="space-y-6">
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-6">
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <Activity className="h-5 w-5 text-emerald-400" />
                    Triage metrics
                  </h3>

                  {/* Score meters */}
                  <div className="space-y-4">
                    {[
                      { label: 'Risk Score', value: isSimulating ? simulatedMetrics.riskScore : (sessionDetail?.risk_score ?? simulatedMetrics.riskScore), max: 100, field: 'riskScore', desc: 'Cumulative severity weights' },
                      { label: 'Anomaly score', value: isSimulating ? simulatedMetrics.anomalyScore : (sessionDetail?.anomaly_score ?? simulatedMetrics.anomalyScore), max: 2, field: 'anomalyScore', desc: 'GRU reconstruction error threshold' },
                      { label: 'Classifier confidence', value: isSimulating ? simulatedMetrics.confidence : (sessionDetail?.confidence != null ? sessionDetail.confidence * 100 : simulatedMetrics.confidence), max: 100, field: 'confidence', desc: 'XGBoost class mapping confidence' }
                    ].map((metric, idx) => (
                      <div key={idx} className="space-y-1.5 p-3.5 rounded-xl bg-slate-950/40 border border-slate-800/40">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400 font-semibold">{metric.label}</span>
                          <span className="font-mono font-bold text-slate-200">{formatValue(metric.value, metric.field)}</span>
                        </div>
                        <div className="h-2 w-full bg-slate-800/80 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-400 transition-all duration-300 rounded-full"
                            style={{ width: `${(metric.value / metric.max) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-[10px] text-slate-500">{metric.desc}</span>
                      </div>
                    ))}
                  </div>

                  {/* Summary Card */}
                  {sessionDetail && (
                    <div className="p-4 rounded-xl border border-slate-800/60 bg-slate-950/30 space-y-3">
                      <div>
                        <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">Prediction classification</span>
                        <h4 className="text-lg font-bold text-slate-100 mt-0.5">{sessionDetail.attack_type}</h4>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div>
                          <span className="text-slate-500">Incident ID</span>
                          <span className="block font-mono font-semibold text-slate-300">{sessionDetail.session_id}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Severity</span>
                          <span className={`block font-semibold ${
                            sessionDetail.severity === 'Critical' ? 'text-red-400' :
                            sessionDetail.severity === 'High' ? 'text-amber-400' :
                            'text-sky-400'
                          }`}>{sessionDetail.severity}</span>
                        </div>
                      </div>

                      {sessionDetail.mitre && (
                        <div className="border-t border-slate-800 pt-3">
                          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">MITRE mapping</span>
                          <span className="text-xs text-slate-300 block font-medium mt-0.5">
                            {sessionDetail.mitre.technique_id} - {sessionDetail.mitre.technique}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Quick actions recommendation */}
                {sessionDetail?.recommendations && (
                  <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                    <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-emerald-400" />
                      Priority action playbook
                    </h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Based on top SHAP indicators, execute the following containment step first:
                    </p>
                    <div className="p-3.5 rounded-xl border border-amber-500/25 bg-amber-500/5 text-amber-300 text-xs font-semibold flex items-center gap-3">
                      <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0" />
                      {sessionDetail.recommendations.priority_action}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TABS 2: Threat Timeline */}
          {activeTab === 'timeline' && (
            <div className="grid grid-cols-3 gap-8">
              {/* Timeline list */}
              <div className="col-span-2 space-y-6">
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-6">
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <Clock className="h-5 w-5 text-emerald-400" />
                    Chronological activity log
                  </h3>
                  
                  {sessionDetail ? (
                    <div className="relative pl-6 border-l border-slate-800 space-y-6 py-2">
                      {/* Use event_timeline from API if available, otherwise fallback to SHAP contributors */}
                      {(sessionDetail.event_timeline || []).length > 0 ? (
                        sessionDetail.event_timeline.map((evt, idx) => (
                          <div key={idx} className="relative">
                            <div className={`absolute -left-[31px] top-1.5 h-3.5 w-3.5 rounded-full border-2 ${
                              evt.type === 'detection' ? 'bg-red-400 border-slate-900 shadow-[0_0_8px_rgba(248,113,113,0.5)]' :
                              evt.type === 'attack'    ? 'bg-amber-400 border-slate-900' :
                              'bg-slate-800 border-emerald-400'
                            }`}></div>
                            <span className="text-[10px] text-slate-500 font-bold block tracking-wider">{evt.time}</span>
                            <h4 className={`text-sm font-semibold mt-0.5 ${
                              evt.type === 'detection' ? 'text-red-300' :
                              evt.type === 'attack'    ? 'text-amber-300' :
                              'text-slate-300'
                            }`}>{evt.event}</h4>
                            <p className="text-xs text-slate-400 mt-0.5">{evt.detail}</p>
                          </div>
                        ))
                      ) : (
                        <>
                          {/* Fallback: static entry + SHAP contributor slice */}
                          <div className="relative">
                            <div className="absolute -left-[31px] top-1.5 h-3.5 w-3.5 rounded-full bg-slate-800 border-2 border-emerald-400"></div>
                            <span className="text-[10px] text-slate-500 font-bold block tracking-wider">09:00:01</span>
                            <h4 className="text-sm font-semibold text-slate-300 mt-0.5">Session Authentication</h4>
                            <p className="text-xs text-slate-400 mt-0.5">User {sessionDetail.employee_id} authenticated successfully.</p>
                          </div>
                          {sessionDetail.positive_contributors.slice(0, 3).map((contrib, idx) => (
                            <div key={idx} className="relative">
                              <div className="absolute -left-[31px] top-1.5 h-3.5 w-3.5 rounded-full bg-slate-800 border-2 border-amber-400"></div>
                              <span className="text-[10px] text-slate-500 font-bold block tracking-wider">09:00:0{idx * 4 + 4}</span>
                              <h4 className="text-sm font-semibold text-slate-300 mt-0.5">{contrib.feature.replace(/_/g, ' ')} anomaly</h4>
                              <p className="text-xs text-slate-400 mt-0.5">
                                Behavioral anomaly flag: value of {contrib.value} was measured as significantly high (SHAP={contrib.shap_value.toFixed(4)}).
                              </p>
                            </div>
                          ))}
                          <div className="relative">
                            <div className="absolute -left-[31px] top-1.5 h-3.5 w-3.5 rounded-full bg-red-400 border-2 border-slate-900 shadow-[0_0_8px_rgba(248,113,113,0.5)]"></div>
                            <span className="text-[10px] text-slate-500 font-bold block tracking-wider">09:00:15</span>
                            <h4 className="text-sm font-semibold text-red-300 mt-0.5">Incident Triage Complete</h4>
                            <p className="text-xs text-slate-400 mt-0.5">XGBoost prediction finalized as **{sessionDetail.attack_type}**.</p>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="py-8 text-center text-slate-500 text-sm">No active session selected. Simulate an attack first.</div>
                  )}
                </div>
              </div>

              {/* Simulated session history list */}
              <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                <h3 className="font-bold text-slate-100 flex items-center gap-2">
                  <Database className="h-5 w-5 text-emerald-400" />
                  Simulated session history
                </h3>
                <p className="text-xs text-slate-400">Select any simulated session to load context</p>

                <div className="space-y-2 overflow-y-auto max-h-[400px] pr-2">
                  {simulatedSessions.map(sess => (
                    <button
                      key={sess.session_id}
                      onClick={() => loadSessionDetails(sess.session_id)}
                      className={`w-full p-3 rounded-xl border text-left flex justify-between items-center transition-all ${
                        selectedSessionId === sess.session_id 
                          ? 'bg-slate-800/80 border-emerald-500/40 text-slate-200' 
                          : 'bg-slate-950/50 border-slate-800/50 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div>
                        <span className="text-xs font-mono font-bold block text-slate-300">{sess.session_id}</span>
                        <span className="text-[10px] block mt-0.5 text-slate-500">{sess.prediction}</span>
                      </div>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                        sess.severity === 'Critical' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                        sess.severity === 'High' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                        'bg-sky-500/10 text-sky-400 border border-sky-500/20'
                      }`}>{sess.severity}</span>
                    </button>
                  ))}
                  {simulatedSessions.length === 0 && (
                    <div className="text-xs text-slate-500 text-center py-6">No simulated sessions yet. Trigger simulations from the dashboard.</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TABS 3: Campaign Graph */}
          {activeTab === 'campaigns' && (
            <div className="grid grid-cols-3 gap-8">
              {/* Campaigns list */}
              <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                <h3 className="font-bold text-slate-100 flex items-center gap-2">
                  <GitBranch className="h-5 w-5 text-emerald-400" />
                  Correlated campaigns
                </h3>
                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                  {campaigns.map(camp => (
                    <button
                      key={camp.campaign_id}
                      onClick={() => setSelectedCampaign(camp)}
                      className={`w-full p-4 rounded-xl border text-left space-y-2 transition-all ${
                        selectedCampaign?.campaign_id === camp.campaign_id
                          ? 'bg-slate-800/80 border-purple-500/40 text-slate-200'
                          : 'bg-slate-950/50 border-slate-800/50 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-bold font-mono text-purple-400">{camp.campaign_id}</span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                          camp.overall_severity === 'Critical' ? 'bg-red-500/10 text-red-400' :
                          camp.overall_severity === 'High' ? 'bg-amber-500/10 text-amber-400' :
                          'bg-sky-500/10 text-sky-400'
                        }`}>{camp.overall_severity}</span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">{camp.summary}</p>
                      <div className="flex items-center gap-4 text-[10px] text-slate-500 font-medium">
                        <span>{camp.session_count} Sessions</span>
                        <span>{camp.affected_employees.length} Users</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Campaign graph: render campaign_chain if present, else legacy SVG */}
              <div className="col-span-2 p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-6">
                <h3 className="font-bold text-slate-100 flex items-center gap-2">
                  <GitBranch className="h-5 w-5 text-emerald-400" />
                  Campaign relationship graph
                </h3>

                {sessionDetail?.campaign_chain && sessionDetail.campaign_chain.length > 0 ? (
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">Live attack chain from the latest combined simulation — showing the full AI processing pipeline.</p>
                    {/* Vertical node chain */}
                    <div className="flex flex-col gap-0">
                      {sessionDetail.campaign_chain.map((node, idx) => {
                        const isFirst = idx === 0;
                        const isLast = idx === sessionDetail.campaign_chain.length - 1;
                        const nodeColor =
                          node.type === 'employee'  ? 'border-sky-500 bg-sky-950/40 text-sky-300' :
                          node.type === 'behaviour' ? 'border-amber-500 bg-amber-950/30 text-amber-300' :
                          node.type === 'model'     ? 'border-emerald-500 bg-emerald-950/30 text-emerald-300' :
                          'border-red-500 bg-red-950/30 text-red-300';
                        const dotColor =
                          node.type === 'employee'  ? 'bg-sky-400' :
                          node.type === 'behaviour' ? 'bg-amber-400' :
                          node.type === 'model'     ? 'bg-emerald-400' :
                          'bg-red-400';
                        return (
                          <div key={node.id} className="flex flex-col items-center">
                            <div className={`w-full max-w-sm px-4 py-2.5 rounded-xl border text-xs font-semibold flex items-center gap-3 ${nodeColor}`}>
                              <span className={`h-2.5 w-2.5 rounded-full shrink-0 ${dotColor}`}></span>
                              {node.label}
                            </div>
                            {!isLast && (
                              <div className="h-5 w-px bg-slate-700 my-0.5"></div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : selectedCampaign ? (
                  <div className="space-y-6">
                    {/* Legacy SVG campaign graph */}
                    <div className="h-48 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-center justify-center p-6 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-3xl"></div>
                      <svg className="w-full h-full max-w-lg" viewBox="0 0 500 120">
                        <line x1="50" y1="60" x2="180" y2="60" stroke="#475569" strokeWidth="2" strokeDasharray="4 4" />
                        <line x1="180" y1="60" x2="310" y2="30" stroke="#a855f7" strokeWidth="2" />
                        <line x1="180" y1="60" x2="310" y2="90" stroke="#a855f7" strokeWidth="2" />
                        <line x1="310" y1="30" x2="440" y2="60" stroke="#a855f7" strokeWidth="2" />
                        <line x1="310" y1="90" x2="440" y2="60" stroke="#a855f7" strokeWidth="2" />
                        <circle cx="50" cy="60" r="18" fill="#1e293b" stroke="#64748b" strokeWidth="2" />
                        <text x="50" y="64" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="bold">IP</text>
                        <text x="50" y="92" textAnchor="middle" fill="#64748b" fontSize="8">{selectedCampaign.common_source_ips?.[0] || '10.10.X.X'}</text>
                        <circle cx="180" cy="60" r="22" fill="#701a75" stroke="#a855f7" strokeWidth="2" />
                        <text x="180" y="64" textAnchor="middle" fill="#f5f3ff" fontSize="9" fontWeight="bold">{selectedCampaign.campaign_id}</text>
                        <circle cx="310" cy="30" r="18" fill="#0f172a" stroke="#a855f7" strokeWidth="2" />
                        <text x="310" y="34" textAnchor="middle" fill="#e9d5ff" fontSize="9">User</text>
                        <text x="310" y="8" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold">{selectedCampaign.affected_employees?.[0]}</text>
                        <circle cx="310" cy="90" r="18" fill="#0f172a" stroke="#a855f7" strokeWidth="2" />
                        <text x="310" y="94" textAnchor="middle" fill="#e9d5ff" fontSize="9">Device</text>
                        <text x="310" y="118" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold">{selectedCampaign.common_devices?.[0] || 'DEV-001'}</text>
                        <circle cx="440" cy="60" r="18" fill="#7f1d1d" stroke="#ef4444" strokeWidth="2" />
                        <text x="440" y="64" textAnchor="middle" fill="#fecaca" fontSize="9">Alert</text>
                      </svg>
                    </div>
                    <div className="grid grid-cols-2 gap-6 text-sm">
                      <div className="space-y-1">
                        <span className="text-slate-500 text-xs font-semibold uppercase tracking-wider block">Attack chain path</span>
                        <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-950/40 text-purple-300 font-mono text-xs">
                          {selectedCampaign.is_attack_chain ? selectedCampaign.chain_description : 'Clustered behavioral alert matches'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-slate-500 text-xs font-semibold uppercase tracking-wider block">Impact summary</span>
                        <p className="text-xs text-slate-400 leading-relaxed">{selectedCampaign.summary}</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <span className="text-slate-500 text-xs font-semibold uppercase tracking-wider block">Coordinated response actions</span>
                      <div className="space-y-2">
                        {selectedCampaign.recommended_actions?.map((act, idx) => (
                          <div key={idx} className="p-3 rounded-xl border border-slate-850 bg-slate-900/40 text-xs text-slate-300 flex items-start gap-3">
                            <span className="text-purple-400 font-bold font-mono">[{idx + 1}]</span>
                            <span>{act}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-500 text-sm">Run a combined simulation to see the AI pipeline chain graph, or select a correlated campaign.</div>
                )}
              </div>
            </div>
          )}

          {/* TABS 4: Incident Reports */}
          {activeTab === 'reports' && (
            <div className="grid grid-cols-3 gap-8">
              {/* Reports list */}
              <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                <h3 className="font-bold text-slate-100 flex items-center gap-2">
                  <FileText className="h-5 w-5 text-emerald-400" />
                  Incident report logs
                </h3>
                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                  {simulatedSessions.map(sess => (
                    <button
                      key={sess.session_id}
                      onClick={() => loadSessionDetails(sess.session_id)}
                      className={`w-full p-4 rounded-xl border text-left space-y-2 transition-all ${
                        selectedSessionId === sess.session_id
                          ? 'bg-slate-800/80 border-emerald-500/40 text-slate-200'
                          : 'bg-slate-950/50 border-slate-800/50 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-bold font-mono text-slate-300">{sess.session_id}</span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                          sess.severity === 'Critical' ? 'bg-red-500/10 text-red-400' :
                          sess.severity === 'High' ? 'bg-amber-500/10 text-amber-400' :
                          'bg-sky-500/10 text-sky-400'
                        }`}>{sess.severity}</span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-1">{sess.prediction}</p>
                    </button>
                  ))}
                  {simulatedSessions.length === 0 && (
                    <button
                      onClick={() => loadSessionDetails('SES-000614')}
                      className={`w-full p-4 rounded-xl border text-left space-y-2 transition-all ${
                        selectedSessionId === 'SES-000614' ? 'bg-slate-800/80 border-emerald-500/40 text-slate-200' : 'bg-slate-950/50 border-slate-800/50 text-slate-400'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-bold font-mono">SES-000614</span>
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400">High</span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-1">Privilege Escalation</p>
                    </button>
                  )}
                </div>
              </div>

              {/* View pane */}
              <div className="col-span-2 p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-6">
                {sessionDetail?.report ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                      <h3 className="font-bold text-slate-100 flex items-center gap-2">
                        <FileText className="h-5 w-5 text-emerald-400" />
                        SOC Report Markdown
                      </h3>
                      <button 
                        onClick={() => {
                          const element = document.createElement("a");
                          const file = new Blob([sessionDetail.report.report_text_markdown], {type: 'text/markdown'});
                          element.href = URL.createObjectURL(file);
                          element.download = `incident_${selectedSessionId}.md`;
                          document.body.appendChild(element);
                          element.click();
                        }}
                        className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-850 text-xs font-semibold text-slate-300 hover:bg-slate-800"
                      >
                        Download MD
                      </button>
                    </div>

                    <div className="h-[450px] overflow-y-auto bg-slate-950/80 border border-slate-850 rounded-xl p-6 font-mono text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                      {sessionDetail.report.report_text_markdown}
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-500 text-sm">Select an incident to view its completed report.</div>
                )}
              </div>
            </div>
          )}

          {/* TABS 5: AI Copilot Chat */}
          {activeTab === 'copilot' && (
            <div className="grid grid-cols-4 gap-8 h-[550px]">
              
              {/* Context checklist sidebar */}
              <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4 flex flex-col justify-between">
                <div className="space-y-4">
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <Terminal className="h-5 w-5 text-emerald-400" />
                    Security context
                  </h3>
                  {sessionDetail ? (
                    <div className="space-y-3 text-xs">
                      <div>
                        <span className="text-slate-500 block">Incident ID</span>
                        <span className="font-mono text-slate-300 font-bold">{sessionDetail.session_id}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Target Employee</span>
                        <span className="font-mono text-slate-300">{sessionDetail.employee_id}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Prediction Class</span>
                        <span className="text-emerald-400 font-semibold">{sessionDetail.attack_type}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Classification Severity</span>
                        <span className="text-amber-400 font-semibold">{sessionDetail.severity}</span>
                      </div>
                    </div>
                  ) : (
                    <span className="text-slate-500 text-xs">No active session selected.</span>
                  )}
                </div>

                {/* Prompt templates suggestions */}
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">Suggested queries</span>
                  <div className="flex flex-col gap-1.5">
                    {[
                      "Why is this classified as a threat?",
                      "What features contributed most?",
                      "Provide the MITRE mapping details.",
                      "What containment steps do I execute?"
                    ].map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => setChatInput(q)}
                        className="w-full p-2.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-950/40 text-left text-xs text-slate-400 hover:text-slate-200"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Chat Interface pane */}
              <div className="col-span-3 p-6 rounded-2xl border border-slate-800 bg-slate-900/25 flex flex-col justify-between h-full relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>

                {/* Messages scroll box */}
                <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                  {chatMessages.map((msg, idx) => (
                    <div 
                      key={idx} 
                      className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}
                    >
                      <div className={`p-3.5 rounded-2xl text-xs leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-emerald-500/10 border border-emerald-500/25 text-emerald-100'
                          : 'bg-slate-950/80 border border-slate-850 text-slate-300'
                      }`}>
                        {msg.content}
                      </div>
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="flex gap-3 max-w-[85%]">
                      <div className="p-3.5 rounded-2xl bg-slate-950/80 border border-slate-850 text-xs text-slate-500 flex items-center gap-2">
                        <RefreshCw className="h-4 w-4 animate-spin text-emerald-400" />
                        Thinking...
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Input box */}
                <form onSubmit={handleChatSubmit} className="flex gap-2 shrink-0">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask the AI Security Copilot a question..."
                    disabled={chatLoading || !sessionDetail}
                    className="flex-1 bg-slate-950/80 border border-slate-850 rounded-xl px-4 py-3 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/40 disabled:opacity-50"
                  />
                  <button
                    type="submit"
                    disabled={chatLoading || !sessionDetail}
                    className="px-4 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-xs flex items-center justify-center disabled:opacity-50"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* TABS 6: Analytics */}
          {activeTab === 'analytics' && (
            <div className="space-y-8">
              <div className="grid grid-cols-2 gap-8">
                
                {/* Feature importance chart */}
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <BarChart2 className="h-5 w-5 text-emerald-400" />
                    Global Feature Importance (SHAP)
                  </h3>
                  <p className="text-xs text-slate-400">Mean absolute SHAP value impact across all 13,015 triage iterations</p>

                  <div className="h-64 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                        <XAxis type="number" stroke="#475569" fontSize={10} />
                        <YAxis dataKey="name" type="category" stroke="#475569" fontSize={10} width={80} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} labelStyle={{ color: '#f8fafc' }} />
                        <Bar dataKey="shap" fill="#10b981" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#10b981' : '#34d399'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Anomaly score chart */}
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/25 space-y-4">
                  <h3 className="font-bold text-slate-100 flex items-center gap-2">
                    <Activity className="h-5 w-5 text-emerald-400" />
                    GRU Reconstruction anomaly thresholds
                  </h3>
                  <p className="text-xs text-slate-400">Reconstruction error baseline (threshold: 0.05)</p>

                  <div className="h-64 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={[
                        { tick: 1, error: 0.012 },
                        { tick: 2, error: 0.015 },
                        { tick: 3, error: 0.024 },
                        { tick: 4, error: 0.089 }, // Anomaly
                        { tick: 5, error: 0.042 },
                        { tick: 6, error: 0.011 }
                      ]} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                        <XAxis dataKey="tick" stroke="#475569" fontSize={10} />
                        <YAxis stroke="#475569" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                        <Line type="monotone" dataKey="error" stroke="#10b981" strokeWidth={2} />
                        {/* Threshold line */}
                        <Line type="monotone" dataKey={() => 0.05} stroke="#f87171" strokeDasharray="5 5" dot={false} strokeWidth={1.5} name="Threshold" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
