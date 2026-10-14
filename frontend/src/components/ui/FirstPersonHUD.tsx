import React from 'react';
import { Agent, ViewMode } from '../../types/agent';
import { Compass, ArrowLeft, User, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Mic, MicOff } from 'lucide-react';
import { AgentReport } from '../3d/AgentScene';

interface FirstPersonHUDProps {
  viewMode: ViewMode;
  onExitMode: () => void;
  targetedAgent: Agent | null;
  targetDistance: number | null;
  pendingReports?: AgentReport[];
  activeReportingAgentId?: string | null;
  interactive?: boolean;
  isListening?: boolean;
  onToggleListening?: () => void;
  lastTranscript?: string;
}

export const FirstPersonHUD: React.FC<FirstPersonHUDProps> = ({
  viewMode,
  onExitMode,
  targetedAgent,
  targetDistance,
  pendingReports = [],
  activeReportingAgentId = null,
  interactive = false,
  isListening = false,
  onToggleListening,
  lastTranscript = '',
}) => {
  if (viewMode === 'isometric') return null;

  const triggerKey = (key: string, isDown: boolean) => {
    const eventType = isDown ? 'keydown' : 'keyup';
    window.dispatchEvent(new KeyboardEvent(eventType, { key, bubbles: true }));
  };

  const targetedReport = targetedAgent
    ? pendingReports.find(r => r.agentId === targetedAgent.id)
    : null;
  const isTargetReporting = targetedAgent && activeReportingAgentId === targetedAgent.id;

  return (
    <div className="absolute inset-0 pointer-events-none z-30 font-mono select-none overflow-hidden">
      {viewMode === 'first-person' && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
          <div
            className={`rounded-full transition-all duration-200 ${
              targetedAgent
                ? targetedReport
                  ? 'w-4 h-4 bg-amber-400 ring-4 ring-amber-500/40 animate-pulse'
                  : 'w-3.5 h-3.5 bg-emerald-400 ring-4 ring-emerald-500/30'
                : 'w-2 h-2 bg-white/70 shadow'
            }`}
          />
        </div>
      )}

      {viewMode === 'vr-stereo' && (
        <>
          <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-slate-800/80 -translate-x-1/2" />
          <div className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2">
            <div className="w-2 h-2 rounded-full bg-white/70" />
          </div>
          <div className="absolute top-1/2 left-3/4 -translate-x-1/2 -translate-y-1/2">
            <div className="w-2 h-2 rounded-full bg-white/70" />
          </div>
        </>
      )}

      <div className="absolute top-16 left-3 pointer-events-auto">
        <button
          onClick={onExitMode}
          className="px-3 py-1.5 rounded-lg bg-slate-950/80 border border-slate-700 text-slate-200 text-xs flex items-center gap-1.5 hover:bg-slate-800"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Sair do {viewMode === 'vr-stereo' ? 'VR' : 'FPS'}
        </button>
      </div>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2">
        {targetedAgent ? (
          <div className="px-3 py-2 rounded-xl bg-slate-950/85 border border-slate-600 text-xs text-slate-200 backdrop-blur-md max-w-xs text-center">
            <div className="flex items-center justify-center gap-1.5 text-emerald-300">
              <User className="w-3.5 h-3.5" />
              {targetedAgent.name}
            </div>
            <div className="text-slate-400 mt-0.5">{targetedAgent.role}</div>
            {targetDistance != null && (
              <div className="text-[10px] text-slate-500 mt-1">{targetDistance.toFixed(1)}m</div>
            )}
            {isTargetReporting && targetedReport && (
              <p className="mt-2 text-amber-200/90 line-clamp-3">{targetedReport.content}</p>
            )}
            {lastTranscript && (
              <p className="mt-2 text-slate-300 italic line-clamp-2">“{lastTranscript}”</p>
            )}
            <p className="mt-2 text-[10px] text-slate-500">
              {interactive ? (onToggleListening ? (isListening ? 'Microfone aberto — fale com esta pessoa' : 'Microfone fechado') : 'Voz indisponível nesta redação') : 'Somente observação'}
            </p>
            {interactive && onToggleListening && (
              <button
                type="button"
                onClick={onToggleListening}
                className={`mt-2 pointer-events-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] ${
                  isListening
                    ? 'bg-amber-500 text-slate-950 border-amber-300'
                    : 'bg-slate-900 text-slate-200 border-slate-600'
                }`}
              >
                {isListening ? <Mic className="w-3.5 h-3.5" /> : <MicOff className="w-3.5 h-3.5" />}
                {isListening ? 'Ouvindo' : 'Falar'}
              </button>
            )}
          </div>
        ) : (
          <div className="px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700 text-[11px] text-slate-400 flex items-center gap-1.5">
            <Compass className="w-3 h-3" />
            WASD para andar · olhe para um agente
          </div>
        )}

        <div className="grid grid-cols-3 gap-1 pointer-events-auto opacity-80">
          <div />
          <button
            className="w-10 h-10 rounded-lg bg-slate-900/80 border border-slate-600 text-white flex items-center justify-center"
            onPointerDown={() => triggerKey('w', true)}
            onPointerUp={() => triggerKey('w', false)}
            onPointerLeave={() => triggerKey('w', false)}
          >
            <ChevronUp className="w-4 h-4" />
          </button>
          <div />
          <button
            className="w-10 h-10 rounded-lg bg-slate-900/80 border border-slate-600 text-white flex items-center justify-center"
            onPointerDown={() => triggerKey('a', true)}
            onPointerUp={() => triggerKey('a', false)}
            onPointerLeave={() => triggerKey('a', false)}
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            className="w-10 h-10 rounded-lg bg-slate-900/80 border border-slate-600 text-white flex items-center justify-center"
            onPointerDown={() => triggerKey('s', true)}
            onPointerUp={() => triggerKey('s', false)}
            onPointerLeave={() => triggerKey('s', false)}
          >
            <ChevronDown className="w-4 h-4" />
          </button>
          <button
            className="w-10 h-10 rounded-lg bg-slate-900/80 border border-slate-600 text-white flex items-center justify-center"
            onPointerDown={() => triggerKey('d', true)}
            onPointerUp={() => triggerKey('d', false)}
            onPointerLeave={() => triggerKey('d', false)}
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
