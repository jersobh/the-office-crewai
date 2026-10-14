export type AvatarType = 'executive_formal' | 'developer_suit' | 'researcher_coat' | 'analyst_casual' | 'cyber_specialist';

export type AgentActivity =
  | 'idle_standing'
  | 'using_computer'
  | 'writing_paper'
  | 'walking'
  | 'getting_coffee'
  | 'drinking_water'
  | 'looking_window'
  | 'at_whiteboard';

export type AgentStatus = 'idle' | 'thinking' | 'speaking';

export type ViewMode = 'isometric' | 'first-person' | 'vr-stereo';

export interface DataTransferBeam {
  id: string;
  fromAgentId: string;
  toAgentId: string;
  progress: number;
}

export interface Agent {
  id: string;
  name: string;
  role: string;
  avatarType: AvatarType;
  /** `<provider>/<model>`, or a bare Gemini name such as gemini-3.5-flash-lite. */
  model: string;
  color: string;
  systemPrompt: string;
  tools: string[];
  skills: string[];
  deskPosition?: [number, number, number];
  currentActivity: AgentActivity;
  status: AgentStatus;
  activeItem?: 'computer' | 'pencil_paper' | 'none';
  /** Local browser voice. Speech stays on this machine. */
  gender?: 'male' | 'female';
  voiceName?: string;
  voicePitch?: number;
  voiceRate?: number;
}
