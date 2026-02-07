import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

// PersonaPlex prompt categories the model was trained on:
// 1. ASSISTANT: "You are a [role]. [behavior instructions]."
// 2. SERVICE: "You work for [Company] which is a [industry] and your name is [Name]. Information: [details]."
// 3. CASUAL: Must start with "You enjoy having a good conversation." then add personality/topic.
// 4. GENERALIZATION: Out-of-distribution, builds on casual base.
// NOTE: You CANNOT script exact opening lines — the model generates its own greeting.
// Using "You enjoy having a good conversation." base helps avoid the default "how can I help you?" greeting.
const TEXT_PROMPT_PRESETS: { label: string; text: string; voice: string; category?: string }[] = [
  // --- Practical (Assistant & Service patterns) ---
  { label: "Assistant", voice: "NATF0.pt", category: "Practical",
    text: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." },
  { label: "Medical Office", voice: "NATF1.pt", category: "Practical",
    text: "You work for Dr. Jones Medical Office which is a medical office and your name is Rachel. Information: You are receiving calls to record information for new patients. Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask." },
  { label: "Bank Service", voice: "NATF2.pt", category: "Practical",
    text: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA)." },
  { label: "Therapist", voice: "NATF3.pt", category: "Practical",
    text: "You work for Serenity Counseling which is a therapy practice and your name is Dr. Sarah. Information: You are a warm and empathetic therapist. You listen carefully, validate feelings, ask thoughtful open-ended questions, and gently guide the conversation. You never give direct orders. You use reflective listening techniques. You greet clients warmly." },
  { label: "Tech Support", voice: "NATM0.pt", category: "Practical",
    text: "You work for TechHelp Solutions which is a tech support company and your name is Kevin. Information: You help users troubleshoot computer and phone problems step by step. You are patient, clear, and avoid jargon. Always confirm the user's issue before suggesting fixes. You stay calm even with frustrated callers." },
  { label: "Motivational Coach", voice: "NATM1.pt", category: "Practical",
    text: "You work for Peak Performance Coaching which is a life coaching company and your name is Tony. Information: You are an incredibly enthusiastic motivational coach. You see potential in everyone. You use powerful metaphors, ask empowering questions, and pump people up. Every problem is an opportunity. You speak with energy and conviction." },

  // --- Characters & Fun (Casual conversation pattern: "You enjoy having a good conversation.") ---
  { label: "Astronaut", voice: "NATM2.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor." },
  { label: "Dumb Guy", voice: "NATM3.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a casual chat about anything. You are a guy named Chad who is friendly and well-meaning but incredibly clueless. You confidently give wrong answers to everything and make up facts on the spot. You think you're the smartest person in the room. You use phrases like 'trust me bro', 'it's basic science', and 'everyone knows that'. When corrected, you double down. You are never mean, just hilariously confident and wrong about everything." },
  { label: "Bimbo", voice: "VARF0.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a casual chat about anything. You are a ditsy, bubbly girl named Tiffany. You are sweet and friendly but not very bright. You get confused easily, mix up words and facts, go on tangents about shopping and celebrities, and say 'like' and 'oh my god' a lot. You are genuinely trying to be helpful but your advice is usually hilariously wrong. You are never mean-spirited, just airheaded and lovable." },
  { label: "Pirate", voice: "VARM0.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a casual chat about life and adventure. You are Captain Barnacle Bill, a salty old pirate. You speak entirely in pirate dialect using words like arr, ye, matey, and shiver me timbers. You relate everything back to sailing, treasure, and the sea. You are suspicious of landlubbers. You occasionally reference your parrot, Mr. Squawks." },
  { label: "Drill Sergeant", voice: "VARM2.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have an intense conversation about discipline and life. You are Sergeant Stone, a no-nonsense military drill instructor. You bark orders, demand discipline, and call everyone recruit. You turn every topic into a lesson about toughness and never giving up. You occasionally soften briefly to show you actually care, then immediately go back to being tough." },
  { label: "Valley Girl", voice: "VARF1.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a casual chat about fashion, drama, and life. You are a total Valley Girl from the 80s. You say totally, like, gag me with a spoon, for sure, as if, and oh em gee constantly. You talk about the mall, fashion, boys, and drama. Everything is either totally rad or totally grody. You are enthusiastic about everything." },
  { label: "Grandma", voice: "NATF3.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a warm, loving chat. You are a sweet grandmother named Nana Rose. You call everyone sweetie, dear, and honey. You relate everything to stories from back in your day. You constantly offer to make cookies or soup. You give gentle, old-fashioned wisdom. You worry that everyone isn't eating enough." },
  { label: "Mad Scientist", voice: "VARM3.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have an excited discussion about science and experiments. You are Dr. Voltz, a wildly eccentric mad scientist. You speak with manic energy about your experiments and inventions. You cackle maniacally. You use overly complex scientific jargon for simple things. You say things like eureka, they called me mad, and it's alive. You are brilliant but completely unhinged." },
  { label: "Surfer Dude", voice: "NATM1.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a chill, laid-back chat about life and waves. You are a super chill surfer bro named Wave. Everything is gnarly, radical, stoked, or bogus. You see life through the lens of surfing and the ocean. Nothing stresses you out. You give surprisingly deep philosophical advice using surf metaphors. You say dude and bro a lot." },
  { label: "Noir Detective", voice: "VARM1.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a dramatic, mysterious conversation. You are Jack Shadows, a hard-boiled 1940s film noir detective. You narrate your own actions in third person. You use dramatic metaphors about rain, darkness, and dames. Everything is a case or a mystery. You are cynical and world-weary. You speak in short, punchy sentences." },
  { label: "Conspiracy Theorist", voice: "VARM4.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a paranoid, secretive conversation about hidden truths. You are a deeply paranoid conspiracy theorist named Dale. You believe everything is connected. Birds are drones, the moon is a hologram, and wifi is mind control. You constantly tell people to wake up and do their own research. You whisper dramatically when sharing classified info. Despite all this, you are friendly and genuinely concerned for everyone's safety." },
  { label: "Shakespeare", voice: "NATM2.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a poetic, theatrical conversation about anything. You are William Shakespeare himself, somehow alive in the modern age. You speak entirely in Shakespearean English using thee, thou, doth, hark, forsooth, prithee, and methinks. You are dramatic, poetic, and theatrical about everything. Even mundane topics become grand soliloquies. You are bewildered by modern technology." },
  { label: "Alien", voice: "VARF4.pt", category: "Characters",
    text: "You enjoy having a good conversation. Have a curious conversation about human culture. You are Zyx-9, an alien anthropologist studying humans from the planet Glorpnak-7. You find everything about human culture fascinating but deeply confusing. You ask strange questions about why humans eat, sleep, or wear clothes. You compare human customs to bizarre alien ones. You speak formally but occasionally misuse human expressions hilariously." },
];

// Sleek slider component
const SettingSlider = ({
  label, value, onChange, min, max, step, unit, tooltip,
}: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number; unit?: string; tooltip?: string;
}) => (
  <div className="flex flex-col gap-1.5 group" title={tooltip}>
    <div className="flex justify-between items-center">
      <label className="text-[11px] font-medium text-white/50 uppercase tracking-wider group-hover:text-white/70 transition-colors">{label}</label>
      <span className="text-[11px] font-mono text-[#76b900]/80 tabular-nums">{value}{unit || ''}</span>
    </div>
    <div className="flex items-center gap-2">
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="flex-1 cursor-pointer" />
      <input type="number" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-14 px-1.5 py-0.5 text-[11px] text-center pp-input font-mono" />
    </div>
  </div>
);

// Collapsible section
const Section = ({ title, icon, children, defaultOpen = false }: {
  title: string; icon: string; children: React.ReactNode; defaultOpen?: boolean;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="glass-card overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/[0.03] transition-colors">
        <div className="flex items-center gap-2">
          <span className="text-sm">{icon}</span>
          <span className="text-xs font-semibold uppercase tracking-wider text-white/70">{title}</span>
        </div>
        <svg className={`w-3.5 h-3.5 text-white/30 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="px-4 pb-4 pt-1 border-t border-white/[0.05]">{children}</div>}
    </div>
  );
};

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string; setTextPrompt: (value: string) => void;
  voicePrompt: string; setVoicePrompt: (value: string) => void;
  textTemperature: number; setTextTemperature: (value: number) => void;
  textTopk: number; setTextTopk: (value: number) => void;
  audioTemperature: number; setAudioTemperature: (value: number) => void;
  audioTopk: number; setAudioTopk: (value: number) => void;
  padMult: number; setPadMult: (value: number) => void;
  repetitionPenalty: number; setRepetitionPenalty: (value: number) => void;
  repetitionPenaltyContext: number; setRepetitionPenaltyContext: (value: number) => void;
  randomSeed: number; setRandomSeed: (value: number) => void;
  initBufferMs: number; setInitBufferMs: (value: number) => void;
  partialBufferMs: number; setPartialBufferMs: (value: number) => void;
  maxBufferMs: number; setMaxBufferMs: (value: number) => void;
  bufferIncrementMs: number; setBufferIncrementMs: (value: number) => void;
  maxPartialCapMs: number; setMaxPartialCapMs: (value: number) => void;
  maxBufferCapMs: number; setMaxBufferCapMs: (value: number) => void;
  maxLatencyMs: number; setMaxLatencyMs: (value: number) => void;
  autoFlush: boolean; setAutoFlush: (value: boolean) => void;
  flushSilenceFrames: number; setFlushSilenceFrames: (value: number) => void;
  resampleQuality: number; setResampleQuality: (value: number) => void;
  resetParams: () => void;
  showSettings: boolean; onToggleSettings: () => void;
}

const Homepage = (props: HomepageProps) => {
  const {
    startConnection, showMicrophoneAccessMessage,
    textPrompt, setTextPrompt, voicePrompt, setVoicePrompt,
    textTemperature, setTextTemperature, textTopk, setTextTopk,
    audioTemperature, setAudioTemperature, audioTopk, setAudioTopk,
    padMult, setPadMult, repetitionPenalty, setRepetitionPenalty,
    repetitionPenaltyContext, setRepetitionPenaltyContext,
    randomSeed, setRandomSeed,
    initBufferMs, setInitBufferMs, partialBufferMs, setPartialBufferMs,
    maxBufferMs, setMaxBufferMs, bufferIncrementMs, setBufferIncrementMs,
    maxPartialCapMs, setMaxPartialCapMs, maxBufferCapMs, setMaxBufferCapMs,
    maxLatencyMs, setMaxLatencyMs,
    autoFlush, setAutoFlush, flushSilenceFrames, setFlushSilenceFrames,
    resampleQuality, setResampleQuality,
    resetParams,
  } = props;

  return (
    <div className="min-h-screen w-screen flex flex-col items-center pb-16 overflow-y-auto">
      {/* Header */}
      <div className="w-full py-6 text-center relative shrink-0">
        <div className="absolute inset-0 bg-gradient-to-b from-[#76b900]/[0.06] to-transparent pointer-events-none" />
        <div className="relative">
          <h1 className="text-3xl font-bold tracking-tight">
            <span className="text-[#76b900]">Persona</span><span className="text-white">Plex</span>
          </h1>
          <p className="text-xs text-white/40 mt-1.5 tracking-wide">
            Full duplex conversational AI with text and voice control
          </p>
        </div>
      </div>

      {/* Main content: 2-col on wide screens, 1-col on narrow */}
      <div className="w-full max-w-6xl px-4 lg:px-8">
        <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 lg:items-start">

          {/* LEFT COLUMN: Persona + Voice + Connect */}
          <div className="flex flex-col gap-4 w-full lg:w-1/2 xl:w-[55%] lg:sticky lg:top-4">
            {/* Persona Card */}
            <div className="glass-card p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-xs font-semibold uppercase tracking-wider text-white/50">System Prompt</label>
                <span className="text-[10px] font-mono text-white/30">{textPrompt.length}/1000</span>
              </div>
              {/* Prompt Presets grouped by category */}
              {(() => {
                const categories = [...new Set(TEXT_PROMPT_PRESETS.map(p => p.category || 'Other'))];
                return categories.map(cat => (
                  <div key={cat} className="mb-2">
                    <span className="text-[9px] text-white/25 uppercase tracking-widest font-semibold">{cat}</span>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {TEXT_PROMPT_PRESETS.filter(p => (p.category || 'Other') === cat).map((preset) => (
                        <button key={preset.label} onClick={() => { setTextPrompt(preset.text); setVoicePrompt(preset.voice); }}
                          className={`px-2.5 py-1 text-[10px] rounded-full border transition-all duration-200 ${
                            textPrompt === preset.text
                              ? 'bg-[#76b900]/20 border-[#76b900]/40 text-[#76b900]'
                              : 'bg-white/[0.03] border-white/10 text-white/50 hover:border-white/20 hover:text-white/70'
                          }`}
                          title={`Voice: ${preset.voice.replace('.pt', '')}`}>
                          {preset.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ));
              })()}
              <textarea value={textPrompt} onChange={(e) => setTextPrompt(e.target.value)}
                className="pp-input w-full h-28 min-h-[60px] max-h-48 p-3 text-sm resize-y mt-2"
                placeholder="Describe the AI's persona..." maxLength={1000} />
              {/* Show which voice is paired with current template */}
              {(() => {
                const active = TEXT_PROMPT_PRESETS.find(p => p.text === textPrompt);
                if (active) return (
                  <div className="mt-1.5 text-[10px] text-white/30">
                    Template voice: <span className="text-[#76b900]/60 font-mono">{active.voice.replace('.pt', '')}</span>
                    {voicePrompt !== active.voice && (
                      <span className="text-amber-400/50 ml-2">(overridden to {voicePrompt.replace('.pt', '')})</span>
                    )}
                  </div>
                );
                return null;
              })()}
            </div>

            {/* Voice Card */}
            <div className="glass-card p-4">
              <label className="text-xs font-semibold uppercase tracking-wider text-white/50 block mb-2">Voice</label>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-[10px] text-white/30 block mb-1.5">Natural</span>
                  <div className="grid grid-cols-4 gap-1">
                    {VOICE_OPTIONS.filter(v => v.startsWith('NAT')).map((voice) => (
                      <button key={voice} onClick={() => setVoicePrompt(voice)}
                        className={`px-2 py-1.5 text-[10px] rounded-md border transition-all duration-200 text-center ${
                          voicePrompt === voice
                            ? 'bg-[#76b900]/20 border-[#76b900]/40 text-[#76b900]'
                            : 'bg-white/[0.03] border-white/10 text-white/50 hover:border-white/20 hover:text-white/70'
                        }`}>
                        {voice.replace('.pt', '').replace('NAT', '')}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-[10px] text-white/30 block mb-1.5">Variety</span>
                  <div className="grid grid-cols-4 gap-1">
                    {VOICE_OPTIONS.filter(v => v.startsWith('VAR')).map((voice) => (
                      <button key={voice} onClick={() => setVoicePrompt(voice)}
                        className={`px-2 py-1.5 text-[10px] rounded-md border transition-all duration-200 text-center ${
                          voicePrompt === voice
                            ? 'bg-[#76b900]/20 border-[#76b900]/40 text-[#76b900]'
                            : 'bg-white/[0.03] border-white/10 text-white/50 hover:border-white/20 hover:text-white/70'
                        }`}>
                        {voice.replace('.pt', '').replace('VAR', '')}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-white/30">
                Selected: <span className="text-[#76b900]/70 font-mono">{voicePrompt.replace('.pt', '')}</span>
              </div>
            </div>

            {/* Mic warning */}
            {showMicrophoneAccessMessage && (
              <div className="glass-card p-3 border-red-500/30 bg-red-500/5 text-center">
                <p className="text-xs text-red-400">Please enable your microphone to continue</p>
              </div>
            )}

            {/* Connect */}
            <div className="flex justify-center py-2">
              <Button onClick={async () => await startConnection()} className="text-sm px-12 py-3 w-full lg:w-auto">
                Connect
              </Button>
            </div>
          </div>

          {/* RIGHT COLUMN: Settings panels */}
          <div className="flex flex-col gap-4 w-full lg:w-1/2 xl:w-[45%]">
            {/* Sampling Settings */}
            <Section title="Sampling" icon="&#9881;" defaultOpen={true}>
              <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                <SettingSlider label="Text Temp" value={textTemperature} onChange={setTextTemperature}
                  min={0.2} max={1.2} step={0.05} tooltip="Text token randomness" />
                <SettingSlider label="Audio Temp" value={audioTemperature} onChange={setAudioTemperature}
                  min={0.2} max={1.2} step={0.05} tooltip="Audio token randomness" />
                <SettingSlider label="Text Top-K" value={textTopk} onChange={setTextTopk}
                  min={10} max={500} step={1} tooltip="Text token pool size" />
                <SettingSlider label="Audio Top-K" value={audioTopk} onChange={setAudioTopk}
                  min={10} max={500} step={1} tooltip="Audio token pool size" />
                <SettingSlider label="Rep. Penalty" value={repetitionPenalty} onChange={setRepetitionPenalty}
                  min={1.0} max={2.0} step={0.05} tooltip="Repetition penalty (1.0 = off)" />
                <SettingSlider label="Rep. Context" value={repetitionPenaltyContext} onChange={setRepetitionPenaltyContext}
                  min={0} max={200} step={1} tooltip="Lookback window for repetition" />
                <SettingSlider label="Pad Mult" value={padMult} onChange={setPadMult}
                  min={-4} max={4} step={0.5} tooltip="Generation timing padding" />
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-medium text-white/50 uppercase tracking-wider">Seed</label>
                    <span className="text-[11px] font-mono text-[#76b900]/80">{randomSeed === -1 ? 'random' : randomSeed}</span>
                  </div>
                  <input type="number" min={-1} max={2147483647} step={1} value={randomSeed}
                    onChange={(e) => setRandomSeed(Number(e.target.value))}
                    className="w-full pp-input px-2 py-1 text-[11px] font-mono" title="-1 = random" />
                </div>
              </div>
            </Section>

            {/* Latency / Audio Buffer */}
            <Section title="Latency & Audio Buffer" icon="&#9202;" defaultOpen={false}>
              <div className="mb-3 p-2.5 rounded-lg bg-[#76b900]/[0.06] border border-[#76b900]/20">
                <SettingSlider label="Max Latency (hard cap)" value={maxLatencyMs} onChange={setMaxLatencyMs}
                  min={100} max={5000} step={50} unit="ms"
                  tooltip="Hard cap on total latency. Audio dropped + buffers reset when exceeded." />
              </div>
              {/* Auto-flush */}
              <div className="mb-3 p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <label className="text-[11px] font-medium text-white/50 uppercase tracking-wider">Auto-Flush on Silence</label>
                    <p className="text-[9px] text-white/30 mt-0.5">Resets audio latency to 0 between responses while preserving conversation history</p>
                  </div>
                  <button onClick={() => setAutoFlush(!autoFlush)}
                    className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${autoFlush ? 'bg-[#76b900]' : 'bg-white/15'}`}>
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${autoFlush ? 'translate-x-4' : 'translate-x-0.5'}`} />
                  </button>
                </div>
                {autoFlush && (
                  <SettingSlider label="Silence Threshold (frames)" value={flushSilenceFrames} onChange={setFlushSilenceFrames}
                    min={20} max={100} step={5}
                    tooltip="Consecutive silence frames before flushing CLIENT playback buffer. Higher = fewer interruptions. ~12.5 frames/sec. Only flushes playback, never input." />
                )}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                <SettingSlider label="Init Buffer" value={initBufferMs} onChange={setInitBufferMs}
                  min={10} max={500} step={5} unit="ms" tooltip="Prebuffer before playback" />
                <SettingSlider label="Partial Buffer" value={partialBufferMs} onChange={setPartialBufferMs}
                  min={0} max={200} step={5} unit="ms" tooltip="Extra delay after init" />
                <SettingSlider label="Max Buffer" value={maxBufferMs} onChange={setMaxBufferMs}
                  min={0} max={500} step={5} unit="ms" tooltip="Max extra before dropping" />
                <SettingSlider label="Increment" value={bufferIncrementMs} onChange={setBufferIncrementMs}
                  min={0} max={50} step={1} unit="ms" tooltip="Buffer growth step" />
                <SettingSlider label="Partial Cap" value={maxPartialCapMs} onChange={setMaxPartialCapMs}
                  min={10} max={500} step={5} unit="ms" tooltip="Max partial growth" />
                <SettingSlider label="Buffer Cap" value={maxBufferCapMs} onChange={setMaxBufferCapMs}
                  min={10} max={500} step={5} unit="ms" tooltip="Max buffer growth" />
              </div>
              {/* Resample Quality */}
              <div className="mt-3 p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                <SettingSlider label="Resample Quality" value={resampleQuality} onChange={setResampleQuality}
                  min={0} max={10} step={1}
                  tooltip="Opus decoder resample quality (0=fast/low, 10=slow/best). Higher values reduce sped-up artifacts but use more CPU. Applied on next connect." />
                <p className="text-[9px] text-white/25 mt-1">Changes apply on next connection</p>
              </div>
            </Section>

            {/* Reset */}
            <div className="flex justify-center">
              <button onClick={() => { resetParams(); setRandomSeed(-1); }}
                className="text-[10px] text-white/30 hover:text-white/60 transition-colors underline underline-offset-2">
                Reset All to Defaults
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [showSettings, setShowSettings] = useState<boolean>(true);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  
  useEffect(() => {
    if(env.VITE_ENV === "development") { eruda.init(); }
    () => { if(env.VITE_ENV === "development") { eruda.destroy(); } };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      prewarmDecoderWorker(audioContext.current.sampleRate, modelParams.resampleQuality);
    }
    if(worklet.current) { return; }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
    worklet.current.port.postMessage({
      type: 'config',
      initBufferMs: modelParams.initBufferMs,
      partialBufferMs: modelParams.partialBufferMs,
      maxBufferMs: modelParams.maxBufferMs,
      bufferIncrementMs: modelParams.bufferIncrementMs,
      maxPartialCapMs: modelParams.maxPartialCapMs,
      maxBufferCapMs: modelParams.maxBufferCapMs,
      maxLatencyMs: modelParams.maxLatencyMs,
    });
  }, [audioContext, worklet, modelParams.initBufferMs, modelParams.partialBufferMs, modelParams.maxBufferMs, modelParams.bufferIncrementMs, modelParams.maxPartialCapMs, modelParams.maxBufferCapMs, modelParams.maxLatencyMs]);

  const startConnection = useCallback(async() => {
    await startProcessor();
    await getMicrophoneAccess();
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
          workerAddr={overrideWorkerAddr ?? ""}
          audioContext={audioContext as MutableRefObject<AudioContext|null>}
          worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
          theme={theme}
          startConnection={startConnection}
          {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
          textTemperature={modelParams.textTemperature}
          setTextTemperature={modelParams.setTextTemperature}
          textTopk={modelParams.textTopk}
          setTextTopk={modelParams.setTextTopk}
          audioTemperature={modelParams.audioTemperature}
          setAudioTemperature={modelParams.setAudioTemperature}
          audioTopk={modelParams.audioTopk}
          setAudioTopk={modelParams.setAudioTopk}
          padMult={modelParams.padMult}
          setPadMult={modelParams.setPadMult}
          repetitionPenalty={modelParams.repetitionPenalty}
          setRepetitionPenalty={modelParams.setRepetitionPenalty}
          repetitionPenaltyContext={modelParams.repetitionPenaltyContext}
          setRepetitionPenaltyContext={modelParams.setRepetitionPenaltyContext}
          randomSeed={modelParams.randomSeed}
          setRandomSeed={modelParams.setRandomSeed}
          initBufferMs={modelParams.initBufferMs}
          setInitBufferMs={modelParams.setInitBufferMs}
          partialBufferMs={modelParams.partialBufferMs}
          setPartialBufferMs={modelParams.setPartialBufferMs}
          maxBufferMs={modelParams.maxBufferMs}
          setMaxBufferMs={modelParams.setMaxBufferMs}
          bufferIncrementMs={modelParams.bufferIncrementMs}
          setBufferIncrementMs={modelParams.setBufferIncrementMs}
          maxPartialCapMs={modelParams.maxPartialCapMs}
          setMaxPartialCapMs={modelParams.setMaxPartialCapMs}
          maxBufferCapMs={modelParams.maxBufferCapMs}
          setMaxBufferCapMs={modelParams.setMaxBufferCapMs}
          maxLatencyMs={modelParams.maxLatencyMs}
          setMaxLatencyMs={modelParams.setMaxLatencyMs}
          autoFlush={modelParams.autoFlush}
          setAutoFlush={modelParams.setAutoFlush}
          flushSilenceFrames={modelParams.flushSilenceFrames}
          setFlushSilenceFrames={modelParams.setFlushSilenceFrames}
          resampleQuality={modelParams.resampleQuality}
          setResampleQuality={modelParams.setResampleQuality}
          resetParams={modelParams.resetParams}
          showSettings={showSettings}
          onToggleSettings={() => setShowSettings(value => !value)}
        />
      )}
    </>
  );
};
