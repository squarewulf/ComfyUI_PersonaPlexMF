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

const TEXT_PROMPT_PRESETS = [
  {
    label: "Assistant (default)",
    text: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
  },
  {
    label: "Medical office (service)",
    text: "You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
  },
  {
    label: "Bank (service)",
    text: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
  },
  {
    label: "Astronaut (fun)",
    text: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
  },
];

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
  // Sampling params
  textTemperature: number;
  setTextTemperature: (value: number) => void;
  audioTemperature: number;
  setAudioTemperature: (value: number) => void;
  textTopk: number;
  setTextTopk: (value: number) => void;
  // Audio buffer params
  initBufferMs: number;
  setInitBufferMs: (value: number) => void;
  partialBufferMs: number;
  setPartialBufferMs: (value: number) => void;
  decoderBufferSamples: number;
  setDecoderBufferSamples: (value: number) => void;
  resampleQuality: number;
  setResampleQuality: (value: number) => void;
  silenceDelayS: number;
  setSilenceDelayS: (value: number) => void;
  // Reset
  resetParams: () => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  textPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
  textTemperature,
  setTextTemperature,
  audioTemperature,
  setAudioTemperature,
  textTopk,
  setTextTopk,
  initBufferMs,
  setInitBufferMs,
  partialBufferMs,
  setPartialBufferMs,
  decoderBufferSamples,
  setDecoderBufferSamples,
  resampleQuality,
  setResampleQuality,
  silenceDelayS,
  setSilenceDelayS,
  resetParams,
}: HomepageProps) => {
  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <h1 className="text-4xl text-black">PersonaPlex</h1>
        <p className="text-sm text-gray-600 mt-2">
          Full duplex conversational AI with text and voice control.
        </p>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Text Prompt:
          </label>
          <div className="border border-gray-300 rounded p-3 mb-3 bg-gray-50">
            <span className="text-xs font-medium text-gray-500 block mb-2">Examples:</span>
            <div className="flex flex-wrap gap-2 justify-center">
              {TEXT_PROMPT_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => setTextPrompt(preset.text)}
                  className="px-3 py-1 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder="Enter your text prompt..."
            maxLength={1000}
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length}/1000
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Voice:
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {VOICE_OPTIONS.map((voice) => (
              <option key={voice} value={voice}>
                {voice
                  .replace('.pt', '')
                  .replace(/^NAT/, 'NATURAL_')
                  .replace(/^VAR/, 'VARIETY_')}
              </option>
            ))}
          </select>
      </div>

        {/* Settings Panel */}
        <div className="w-full border border-gray-300 rounded p-4 bg-gray-50">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Settings</h3>
          <div className="grid grid-cols-2 gap-4">
            {/* Text Temperature */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Text temperature</label>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={textTemperature}
                onChange={(e) => setTextTemperature(parseFloat(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="0.1"
                max="1.5"
                step="0.05"
                value={textTemperature}
                onChange={(e) => setTextTemperature(parseFloat(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Audio Temperature */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Audio temperature</label>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={audioTemperature}
                onChange={(e) => setAudioTemperature(parseFloat(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="0.1"
                max="1.5"
                step="0.05"
                value={audioTemperature}
                onChange={(e) => setAudioTemperature(parseFloat(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Top-k */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Top-k</label>
              <input
                type="range"
                min="1"
                max="500"
                step="1"
                value={textTopk}
                onChange={(e) => setTextTopk(parseInt(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="1"
                max="500"
                value={textTopk}
                onChange={(e) => setTextTopk(parseInt(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Init buffer */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Init buffer (ms)</label>
              <input
                type="range"
                min="50"
                max="1000"
                step="10"
                value={initBufferMs}
                onChange={(e) => setInitBufferMs(parseInt(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="50"
                max="1000"
                value={initBufferMs}
                onChange={(e) => setInitBufferMs(parseInt(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Partial buffer */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Partial buffer (ms)</label>
              <input
                type="range"
                min="50"
                max="500"
                step="10"
                value={partialBufferMs}
                onChange={(e) => setPartialBufferMs(parseInt(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="50"
                max="500"
                value={partialBufferMs}
                onChange={(e) => setPartialBufferMs(parseInt(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Decoder buffer */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Decoder buffer (samples @24k)</label>
              <input
                type="range"
                min="480"
                max="9600"
                step="480"
                value={decoderBufferSamples}
                onChange={(e) => setDecoderBufferSamples(parseInt(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="480"
                max="9600"
                value={decoderBufferSamples}
                onChange={(e) => setDecoderBufferSamples(parseInt(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Resample quality */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Resample quality</label>
              <input
                type="range"
                min="0"
                max="10"
                step="1"
                value={resampleQuality}
                onChange={(e) => setResampleQuality(parseInt(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="0"
                max="10"
                value={resampleQuality}
                onChange={(e) => setResampleQuality(parseInt(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
            {/* Silence delay */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 w-24">Silence delay (s)</label>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={silenceDelayS}
                onChange={(e) => setSilenceDelayS(parseFloat(e.target.value))}
                className="flex-grow"
              />
              <input
                type="number"
                min="0.01"
                max="0.5"
                step="0.01"
                value={silenceDelayS}
                onChange={(e) => setSilenceDelayS(parseFloat(e.target.value))}
                className="w-16 p-1 text-sm border rounded"
              />
            </div>
          </div>
          <div className="mt-4 flex justify-center">
            <button
              onClick={resetParams}
              className="px-4 py-2 text-sm bg-gray-200 hover:bg-gray-300 text-gray-700 rounded border border-gray-300"
            >
              Set Defaults
            </button>
          </div>
        </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">Please enable your microphone before proceeding</p>
        )}
        
        <Button onClick={async () => await startConnection()}>Connect</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  
  // Fetch server defaults on mount
  useEffect(() => {
    const fetchDefaults = async () => {
      try {
        const response = await fetch('/api/defaults');
        if (response.ok) {
          const defaults = await response.json();
          console.log('Loaded server defaults:', defaults);
          if (defaults.voicePrompt) {
            modelParams.setVoicePrompt(defaults.voicePrompt);
          }
          if (defaults.textPrompt) {
            modelParams.setTextPrompt(defaults.textPrompt);
          }
          if (defaults.textTemperature !== undefined) {
            modelParams.setTextTemperature(defaults.textTemperature);
          }
          if (defaults.audioTemperature !== undefined) {
            modelParams.setAudioTemperature(defaults.audioTemperature);
          }
          if (defaults.textTopk !== undefined) {
            modelParams.setTextTopk(defaults.textTopk);
          }
          if (defaults.audioTopk !== undefined) {
            modelParams.setAudioTopk(defaults.audioTopk);
          }
          // Audio buffer settings
          if (defaults.initBufferMs !== undefined) {
            modelParams.setInitBufferMs(defaults.initBufferMs);
          }
          if (defaults.partialBufferMs !== undefined) {
            modelParams.setPartialBufferMs(defaults.partialBufferMs);
          }
          if (defaults.decoderBufferSamples !== undefined) {
            modelParams.setDecoderBufferSamples(defaults.decoderBufferSamples);
          }
          if (defaults.resampleQuality !== undefined) {
            modelParams.setResampleQuality(defaults.resampleQuality);
          }
          if (defaults.silenceDelayS !== undefined) {
            modelParams.setSilenceDelayS(defaults.silenceDelayS);
          }
          // Set audio buffer settings globally for decoder/processor
          (window as any).audioSettings = {
            initBuf: defaults.initBufferMs || 400,
            partialBuf: defaults.partialBufferMs || 210,
            decBuf: defaults.decoderBufferSamples || 3840,
            resampleQ: defaults.resampleQuality || 5,
            silenceDelay: defaults.silenceDelayS || 0.07,
          };
          console.log('Audio settings applied:', (window as any).audioSettings);
        }
      } catch (e) {
        console.log('Could not fetch server defaults, using client defaults');
      }
    };
    fetchDefaults();
  }, []);  // Only run once on mount
  
  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
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
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
    
    // Send buffer config to worklet from global audio settings
    const audioSettings = (window as any).audioSettings || {};
    worklet.current.port.postMessage({
      type: 'config',
      initBufferMs: audioSettings.initBuf || 400,
      partialBufferMs: audioSettings.partialBuf || 210,
    });
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
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
          audioTemperature={modelParams.audioTemperature}
          setAudioTemperature={modelParams.setAudioTemperature}
          textTopk={modelParams.textTopk}
          setTextTopk={modelParams.setTextTopk}
          initBufferMs={modelParams.initBufferMs}
          setInitBufferMs={modelParams.setInitBufferMs}
          partialBufferMs={modelParams.partialBufferMs}
          setPartialBufferMs={modelParams.setPartialBufferMs}
          decoderBufferSamples={modelParams.decoderBufferSamples}
          setDecoderBufferSamples={modelParams.setDecoderBufferSamples}
          resampleQuality={modelParams.resampleQuality}
          setResampleQuality={modelParams.setResampleQuality}
          silenceDelayS={modelParams.silenceDelayS}
          setSilenceDelayS={modelParams.setSilenceDelayS}
          resetParams={modelParams.resetParams}
        />
      )}
    </>
  );
};
