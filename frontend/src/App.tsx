import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Agent, ViewMode } from './types/agent';
import { AgentScene, DeskMonitorFrame, AgentReport } from './components/3d/AgentScene';
import { Navbar } from './components/ui/Navbar';
import { WhiteboardModal, WhiteboardNote } from './components/ui/WhiteboardModal';
import { FirstPersonHUD } from './components/ui/FirstPersonHUD';
import { soundFX } from './utils/audio';
import { liveVoice, type LiveTranscript } from './utils/liveVoice';
import { Eye, Mic, MicOff } from 'lucide-react';

const BALLOON_AUTO_DISMISS_MS = 6000;
const BOARD_LABELS = new Set(['Task', 'Note', 'Handoff', 'Result', 'Job']);

function boardColor(label: string, status?: string): WhiteboardNote['color'] {
  if (status === 'delivered' || status === 'completed' || label === 'Result') return 'emerald';
  if (status === 'failed' || status === 'blocked' || status === 'waiting') return 'pink';
  if (status === 'in_progress' || status === 'ready_to_close' || label === 'Handoff' || label === 'Job') {
    return 'cyan';
  }
  if (label === 'Note') return 'yellow';
  return 'yellow';
}

type LiveLine = { sender: 'user' | 'agent'; text: string; streaming?: boolean };

function applyLiveTranscript(
  prev: Record<string, LiveLine[]>,
  agentId: string,
  item: LiveTranscript,
): Record<string, LiveLine[]> {
  const chunk = item.text || '';
  const list = [...(prev[agentId] || [])];
  const last = list[list.length - 1];
  const sender = item.role === 'user' ? 'user' : 'agent';
  if (last && last.sender === sender && last.streaming) {
    list[list.length - 1] = {
      sender,
      text: chunk ? `${last.text}${chunk}` : last.text,
      streaming: !item.finished,
    };
    return { ...prev, [agentId]: list };
  }
  if (!chunk.trim()) return prev;
  list.push({ sender, text: chunk, streaming: !item.finished });
  return { ...prev, [agentId]: list.slice(-8) };
}

function nodeToNote(node: any): WhiteboardNote | null {
  if (!node || node.archived || !BOARD_LABELS.has(node.label)) return null;
  const status = String(node.properties?.status || (node.label === 'Note' ? 'note' : 'open'));
  const assignee = String(node.properties?.assignee_name || node.author || '');
  const assigneeId = node.properties?.assignee ? String(node.properties.assignee) : undefined;
  const result = String(node.properties?.result || '');
  const jobId = String(node.properties?.job_id || (node.label === 'Job' ? node.id : '') || '');

  let title = node.title || node.label;
  if (node.label === 'Task') title = `${status.replace('_', ' ').toUpperCase()} · ${assignee}`;
  else if (node.label === 'Handoff') title = `DELEGAÇÃO · ${node.title || (node.author + ' → ' + assignee)}`;
  else if (node.label === 'Result') title = `RESULTADO · ${assignee || node.author}`;
  else if (node.label === 'Job') {
    const done = node.properties?.done_count;
    const total = node.properties?.task_count;
    const progress = total != null ? ` · ${done || 0}/${total}` : '';
    title = `JOB · ${status.replace('_', ' ').toUpperCase()}${progress}`;
  }

  let content = result || node.content || '';
  if (node.label === 'Task' && result && node.content && result !== node.content) {
    content = `${node.content}\n\n✓ ${result}`;
  }
  if (node.label === 'Job' && result) {
    content = `${node.content || ''}\n\n✓ Entrega: ${result}`.trim();
  }

  let author = node.author || '';
  if (node.label === 'Task') author = `Atribuído por ${node.author || 'Office'}`;
  else if (node.label === 'Handoff') author = `Encaminhado por ${node.author}`;
  else if (node.label === 'Result') author = `Concluído por ${assignee || node.author}`;
  else if (node.label === 'Job') author = `Aberto por ${node.author || 'Office'}`;

  return {
    id: node.id,
    title,
    content,
    author,
    kind: node.label,
    assigneeId,
    status,
    jobId: jobId || undefined,
    color: boardColor(node.label, status),
    pinned: true,
  };
}

export default function App() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('isometric');
  const [isWhiteboardOpen, setIsWhiteboardOpen] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [officeName, setOfficeName] = useState('The Office');
  const [articlesPerDay, setArticlesPerDay] = useState(3);
  const [blogUrl, setBlogUrl] = useState<string | undefined>();
  const [readOnly, setReadOnly] = useState(true);
  // Live voice is Gemini Live. The server reports false when Gemini is not configured.
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [whiteboardNotes, setWhiteboardNotes] = useState<WhiteboardNote[]>([]);
  const [agentThinkingMap, setAgentThinkingMap] = useState<Record<string, boolean>>({});
  const [agentTaskCounts, setAgentTaskCounts] = useState<Record<string, number>>({});
  const [pendingReports, setPendingReports] = useState<AgentReport[]>([]);
  const [activeReportingAgentId, setActiveReportingAgentId] = useState<string | null>(null);
  const [targetedAgent, setTargetedAgent] = useState<Agent | null>(null);
  const [targetDistance, setTargetDistance] = useState<number | null>(null);
  const [liveAgentId, setLiveAgentId] = useState<string | null>(null);
  const [liveLines, setLiveLines] = useState<Record<string, LiveLine[]>>({});
  const [fpsMicEnabled, setFpsMicEnabled] = useState(true);
  const [lastTranscript, setLastTranscript] = useState('');
  const monitorsRef = useRef<Record<string, DeskMonitorFrame>>({});
  const [monitorAgentId, setMonitorAgentId] = useState<string | null>(null);
  const [monitorFrame, setMonitorFrame] = useState<DeskMonitorFrame | null>(null);
  const seenNodesRef = useRef<Set<string>>(new Set());
  const agentsRef = useRef<Agent[]>(agents);
  agentsRef.current = agents;

  useEffect(() => {
    soundFX.setEnabled(soundEnabled);
  }, [soundEnabled]);

  // Server roster is authoritative — no local hire / edit.
  useEffect(() => {
    let cancelled = false;
    fetch('/api/office')
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (cancelled || !data) return;
        const serverAgents = Array.isArray(data.agents) ? data.agents : [];
        setAgents(serverAgents);
        if (data.config?.name) setOfficeName(String(data.config.name));
        if (data.config?.articles_per_day) setArticlesPerDay(Number(data.config.articles_per_day) || 3);
        if (data.config?.blog_url) setBlogUrl(String(data.config.blog_url));
        setReadOnly(data.read_only !== false);
        setVoiceAvailable(data.voice_available === true);
        const notes: WhiteboardNote[] = [];
        for (const node of data.nodes || []) {
          const note = nodeToNote(node);
          if (note) {
            notes.push(note);
            seenNodesRef.current.add(note.id);
          }
        }
        setWhiteboardNotes(notes);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Live blackboard (read-only)
  useEffect(() => {
    const source = new EventSource('/api/events');
    source.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        const node = payload.node;
        if (!node) return;

        if (node.label === 'AgentStatus') {
          const agentId = String(node.properties?.agent_id || '');
          const status = String(node.properties?.status || '');
          const activity = String(node.properties?.activity || 'using_computer');
          if (agentId) {
            setAgentThinkingMap(prev => ({ ...prev, [agentId]: status === 'thinking' }));
            setAgents(prev =>
              prev.map(a =>
                a.id === agentId
                  ? {
                      ...a,
                      status: status === 'thinking' || status === 'speaking' ? status : 'idle',
                      currentActivity: (activity as Agent['currentActivity']) || a.currentActivity,
                    }
                  : a,
              ),
            );
          }
          return;
        }

        const note = nodeToNote(node);
        if (!note) {
          if (node.id && node.archived) {
            setWhiteboardNotes(prev => prev.filter(item => item.id !== node.id));
          }
          return;
        }

        setWhiteboardNotes(prev => {
          const idx = prev.findIndex(n => n.id === note.id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = note;
            return next;
          }
          return [note, ...prev];
        });

        if (node.label === 'Result' && !seenNodesRef.current.has(node.id)) {
          seenNodesRef.current.add(node.id);
          const agentId = String(node.properties?.assignee || '');
          const agentName = String(node.properties?.assignee_name || node.author || 'Agent');
          const content = String(node.properties?.result || node.content || '').slice(0, 400);
          if (agentId && content) {
            const report: AgentReport = {
              id: node.id,
              agentId,
              agentName,
              content,
              timestamp: Date.now(),
              needsFeedback: false,
            };
            setPendingReports(prev => [...prev.filter(r => r.agentId !== agentId), report]);
            setActiveReportingAgentId(agentId);
            setAgentTaskCounts(prev => ({
              ...prev,
              [agentId]: (prev[agentId] || 0) + 1,
            }));
            window.setTimeout(() => {
              setPendingReports(prev => prev.filter(r => r.id !== report.id));
              setActiveReportingAgentId(cur => (cur === agentId ? null : cur));
            }, BALLOON_AUTO_DISMISS_MS);
          }
        }

        if (node.label === 'Task' && node.properties?.status === 'completed') {
          const agentId = String(node.properties?.assignee || '');
          if (agentId) {
            setAgentTaskCounts(prev => ({
              ...prev,
              [agentId]: Math.max(prev[agentId] || 0, 1),
            }));
          }
        }
      } catch {
        /* ignore malformed SSE */
      }
    };
    return () => source.close();
  }, []);

  // Desk monitor frames
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      if (closed) return;
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      socket = new WebSocket(`${proto}://${window.location.host}/ws/monitors`);
      socket.onmessage = ev => {
        try {
          const msg = JSON.parse(ev.data);
          const agentId = String(msg.agentId || '');
          if (!agentId || !msg.jpeg) return;
          const frame: DeskMonitorFrame = {
            src: `data:image/jpeg;base64,${msg.jpeg}`,
            url: String(msg.url || ''),
            seq: Date.now(),
            updatedAt: Date.now(),
          };
          monitorsRef.current[agentId] = frame;
          if (monitorAgentId === agentId) setMonitorFrame(frame);
        } catch {
          /* ignore */
        }
      };
      socket.onclose = () => {
        if (!closed) window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      socket?.close();
    };
  }, [monitorAgentId]);

  const topPerformer = (() => {
    let best: { agent: Agent; tasksCompleted: number; score: number } | null = null;
    for (const agent of agents) {
      const tasksCompleted = agentTaskCounts[agent.id] || 0;
      const score = tasksCompleted;
      if (!best || score > best.score) best = { agent, tasksCompleted, score };
    }
    return best && best.score > 0 ? best : null;
  })();

  const onSelectAgent = useCallback((agentId: string) => {
    setSelectedAgentId(prev => (prev === agentId ? null : agentId));
  }, []);

  const rememberTranscript = useCallback((agentId: string, item: LiveTranscript) => {
    if (item.role === 'user' && item.text) setLastTranscript(item.text);
    setLiveLines(prev => applyLiveTranscript(prev, agentId, item));
  }, []);

  const markSpeaking = useCallback((agentId: string, status: 'speaking' | 'thinking' | 'idle') => {
    setAgents(prev => prev.map(agent => (
      agent.id === agentId ? { ...agent, status: status === 'idle' ? 'idle' : status } : agent
    )));
  }, []);

  const startTalking = useCallback((agentId: string) => {
    setLiveAgentId(agentId);
    void liveVoice.start({
      agentId,
      onTranscript: item => rememberTranscript(agentId, item),
      onStatus: status => {
        if (!liveVoice.activeAgentId) setLiveAgentId(null);
        markSpeaking(agentId, status);
      },
    }).catch(() => {
      setLiveAgentId(null);
      markSpeaking(agentId, 'idle');
    });
  }, [markSpeaking, rememberTranscript]);

  const stopTalking = useCallback((agentId?: string) => {
    const current = liveVoice.activeAgentId;
    liveVoice.stop();
    setLiveAgentId(null);
    if (current || agentId) markSpeaking(current || agentId || '', 'idle');
  }, [markSpeaking]);

  useEffect(() => {
    if (readOnly || !voiceAvailable) {
      if (liveVoice.activeAgentId) stopTalking();
      return;
    }
    if (viewMode === 'isometric' || !fpsMicEnabled) {
      if (viewMode === 'isometric' && liveVoice.activeAgentId && liveVoice.activeAgentId === selectedAgentId) {
        return;
      }
      if (liveVoice.activeAgentId) stopTalking();
      return;
    }
    const locked = selectedAgentId
      ? agentsRef.current.find(agent => agent.id === selectedAgentId) || null
      : null;
    const agent = locked || targetedAgent;
    if (!agent) {
      if (liveVoice.activeAgentId) stopTalking();
      return;
    }
    if (liveVoice.activeAgentId === agent.id) return;
    startTalking(agent.id);
  }, [readOnly, voiceAvailable, viewMode, fpsMicEnabled, targetedAgent, selectedAgentId, startTalking, stopTalking]);

  const assignTask = useCallback(async (agentId: string, prompt: string) => {
    if (!agentId) {
      const response = await fetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Nota', content: prompt }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || 'A redação recusou a nota.');
      return 'Nota fixada no quadro.';
    }
    const response = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: agentId, prompt }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || 'A redação recusou a tarefa.');
    }
    const name = agentsRef.current.find(agent => agent.id === agentId)?.name || agentId;
    return `${name} recebeu a tarefa (${body.status || 'open'}).`;
  }, []);

  const archiveNote = useCallback(async (id: string) => {
    const response = await fetch('/api/nodes/archive', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Não foi possível arquivar.');
    }
    setWhiteboardNotes(prev => prev.filter(note => note.id !== id));
  }, []);

  const selectedAgent = agents.find(agent => agent.id === selectedAgentId) || null;

  return (
    <div className="relative w-full h-screen overflow-hidden office-stage">
      <div className="office-atmosphere" aria-hidden />

      <AgentScene
        agents={agents}
        selectedAgentId={selectedAgentId}
        onSelectAgent={onSelectAgent}
        onOpenWhiteboard={() => setIsWhiteboardOpen(true)}
        viewMode={viewMode}
        onExitFirstPerson={() => setViewMode('isometric')}
        onTargetedAgentChange={(agent, distance) => {
          setTargetedAgent(agent);
          setTargetDistance(distance);
        }}
        topPerformer={topPerformer}
        monitorsRef={monitorsRef}
        onOpenMonitor={agentId => {
          setMonitorAgentId(agentId);
          setMonitorFrame(monitorsRef.current[agentId] || null);
        }}
        boardNotes={whiteboardNotes}
        pendingReports={pendingReports}
        thinkingAgents={agentThinkingMap}
      />

      <Navbar
        viewMode={viewMode}
        onChangeViewMode={setViewMode}
        onOpenWhiteboard={() => setIsWhiteboardOpen(true)}
        soundEnabled={soundEnabled}
        onToggleSound={() => {
          const next = !soundEnabled;
          soundFX.setEnabled(next);
          setSoundEnabled(next);
        }}
        agentCount={agents.length}
        officeName={officeName}
        articlesPerDay={articlesPerDay}
        blogUrl={blogUrl}
        topPerformer={topPerformer}
        readOnly={readOnly}
      />

      <div className="absolute top-3 left-3 z-40 pointer-events-none">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950/75 border border-amber-500/40 text-amber-200 text-xs font-mono backdrop-blur-md shadow-lg shadow-amber-900/20">
          <Eye className="w-3.5 h-3.5" />
          <span>{readOnly ? 'Modo espectador — somente observação' : 'Modo interativo — localhost'}</span>
        </div>
      </div>

      <FirstPersonHUD
        viewMode={viewMode}
        onExitMode={() => setViewMode('isometric')}
        targetedAgent={targetedAgent}
        targetDistance={targetDistance}
        pendingReports={pendingReports}
        activeReportingAgentId={activeReportingAgentId}
        interactive={!readOnly}
        isListening={!readOnly && voiceAvailable && fpsMicEnabled && liveAgentId !== null && viewMode !== 'isometric'}
        onToggleListening={voiceAvailable ? () => {
          setFpsMicEnabled(prev => {
            const next = !prev;
            if (!next) stopTalking();
            return next;
          });
        } : undefined}
        lastTranscript={lastTranscript}
      />

      <WhiteboardModal
        isOpen={isWhiteboardOpen}
        onClose={() => setIsWhiteboardOpen(false)}
        notes={whiteboardNotes}
        agents={agents.map(a => ({ id: a.id, name: a.name, role: a.role }))}
        readOnly={readOnly}
        onAssign={assignTask}
        onArchive={archiveNote}
      />

      {!readOnly && selectedAgent && (
        <form
          className="absolute bottom-4 left-4 z-40 w-[min(380px,92vw)] rounded-xl border border-slate-600/80 bg-slate-950/90 shadow-2xl backdrop-blur-md p-3 space-y-2"
          onSubmit={async event => {
            event.preventDefault();
            const field = event.currentTarget.elements.namedItem('prompt') as HTMLTextAreaElement;
            const text = field.value.trim();
            if (!text) return;
            if (liveVoice.activeAgentId === selectedAgent.id) {
              liveVoice.sendText(text);
              rememberTranscript(selectedAgent.id, { role: 'user', text, finished: true });
              field.value = '';
              return;
            }
            try {
              await assignTask(selectedAgent.id, text);
              field.value = '';
            } catch {
              /* the board stays the source of truth */
            }
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-mono text-amber-200">
              {selectedAgent.name}
              <span className="text-slate-400"> · {selectedAgent.role}</span>
            </p>
            {voiceAvailable && (
              <button
                type="button"
                onClick={() => {
                  if (liveAgentId === selectedAgent.id) stopTalking(selectedAgent.id);
                  else startTalking(selectedAgent.id);
                }}
                className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-semibold ${
                  liveAgentId === selectedAgent.id
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-800 text-slate-200 border border-slate-600'
                }`}
              >
                {liveAgentId === selectedAgent.id ? <Mic className="w-3.5 h-3.5" /> : <MicOff className="w-3.5 h-3.5" />}
                {liveAgentId === selectedAgent.id ? 'Ouvindo' : 'Falar'}
              </button>
            )}
          </div>
          {(liveLines[selectedAgent.id] || []).slice(-3).map((line, index) => (
            <p key={`${line.sender}-${index}`} className="text-[11px] text-slate-300 line-clamp-2">
              <span className="text-slate-500">{line.sender === 'user' ? 'Você' : selectedAgent.name}: </span>
              {line.text}
            </p>
          ))}
          <textarea
            name="prompt"
            rows={2}
            placeholder={liveAgentId === selectedAgent.id ? 'Fale, ou escreva um pedido' : 'Descreva o trabalho'}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100"
          />
          <button
            type="submit"
            className="px-3 py-1.5 rounded-lg bg-amber-500 text-slate-950 text-xs font-semibold"
          >
            {liveAgentId === selectedAgent.id ? 'Enviar na conversa' : 'Enviar tarefa'}
          </button>
        </form>
      )}

      {monitorAgentId && monitorFrame && (
        <div className="absolute bottom-4 right-4 z-40 w-[min(420px,92vw)] rounded-xl overflow-hidden border border-slate-600/80 bg-slate-950/90 shadow-2xl backdrop-blur-md">
          <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700/80 text-xs font-mono text-slate-300">
            <span className="truncate">
              Monitor · {agents.find(a => a.id === monitorAgentId)?.name || monitorAgentId}
            </span>
            <button
              className="text-slate-400 hover:text-white pointer-events-auto"
              onClick={() => {
                setMonitorAgentId(null);
                setMonitorFrame(null);
              }}
            >
              fechar
            </button>
          </div>
          <img src={monitorFrame.src} alt="Desk monitor" className="w-full aspect-video object-cover" />
          {monitorFrame.url && (
            <div className="px-3 py-1.5 text-[10px] text-slate-500 font-mono truncate">{monitorFrame.url}</div>
          )}
        </div>
      )}
    </div>
  );
}
