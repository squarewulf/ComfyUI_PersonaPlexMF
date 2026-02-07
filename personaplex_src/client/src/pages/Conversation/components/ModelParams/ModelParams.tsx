import { FC, RefObject, useState } from "react";
import { useModelParams, DEFAULT_INIT_BUFFER_MS, DEFAULT_PARTIAL_BUFFER_MS, DEFAULT_MAX_BUFFER_MS, DEFAULT_BUFFER_INCREMENT_MS, DEFAULT_MAX_PARTIAL_CAP_MS, DEFAULT_MAX_BUFFER_CAP_MS, DEFAULT_MAX_LATENCY_MS, DEFAULT_AUTO_FLUSH, DEFAULT_FLUSH_SILENCE_FRAMES, DEFAULT_RESAMPLE_QUALITY } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";

type ModelParamsProps = {
  isConnected: boolean;
  modal?: RefObject<HTMLDialogElement>,
} &  ReturnType<typeof useModelParams>;
export const ModelParams:FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  initBufferMs,
  partialBufferMs,
  maxBufferMs,
  bufferIncrementMs,
  maxPartialCapMs,
  maxBufferCapMs,
  maxLatencyMs,
  autoFlush,
  flushSilenceFrames,
  resampleQuality,
  setParams,
  resetParams,
  isConnected,
  textPrompt,
  voicePrompt,
  randomSeed,
  modal,
}) => {
  const [modalVoicePrompt, setModalVoicePrompt] = useState<string>(voicePrompt);
  const [modalTextPrompt, setModalTextPrompt] = useState<string>(textPrompt);
  return (
    <div className="p-4 mt-4 self-center flex flex-col items-center text-center glass-card max-w-md w-full">
        <div className="space-y-3 w-full">
          <div>
            <label className="text-[11px] font-medium text-white/50 uppercase tracking-wider block text-left mb-1">Text Prompt</label>
            <input className="pp-input w-full px-3 py-1.5 text-sm" disabled={isConnected} type="text" value={modalTextPrompt} onChange={e => setModalTextPrompt(e.target.value)} />
          </div>
          <div>
            <label className="text-[11px] font-medium text-white/50 uppercase tracking-wider block text-left mb-1">Voice</label>
            <select className="pp-input w-full px-3 py-1.5 text-sm" disabled={isConnected} value={modalVoicePrompt} onChange={e => setModalVoicePrompt(e.target.value)}>
              {["NATF0","NATF1","NATF2","NATF3","NATM0","NATM1","NATM2","NATM3","VARF0","VARF1","VARF2","VARF3","VARF4","VARM0","VARM1","VARM2","VARM3","VARM4"].map(v => (
                <option key={v} value={`${v}.pt`}>{v}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <Button variant="ghost" onClick={resetParams}>Reset</Button>
          <Button variant="primary" onClick={() => {
            setParams({
            textTemperature,
            textTopk,
            audioTemperature,
            audioTopk,
            padMult,
            repetitionPenalty,
            repetitionPenaltyContext,
            textPrompt: modalTextPrompt,
            voicePrompt: modalVoicePrompt,
            randomSeed,
            initBufferMs: initBufferMs ?? DEFAULT_INIT_BUFFER_MS,
            partialBufferMs: partialBufferMs ?? DEFAULT_PARTIAL_BUFFER_MS,
            maxBufferMs: maxBufferMs ?? DEFAULT_MAX_BUFFER_MS,
            bufferIncrementMs: bufferIncrementMs ?? DEFAULT_BUFFER_INCREMENT_MS,
            maxPartialCapMs: maxPartialCapMs ?? DEFAULT_MAX_PARTIAL_CAP_MS,
            maxBufferCapMs: maxBufferCapMs ?? DEFAULT_MAX_BUFFER_CAP_MS,
            maxLatencyMs: maxLatencyMs ?? DEFAULT_MAX_LATENCY_MS,
            autoFlush: autoFlush ?? DEFAULT_AUTO_FLUSH,
            flushSilenceFrames: flushSilenceFrames ?? DEFAULT_FLUSH_SILENCE_FRAMES,
            resampleQuality: resampleQuality ?? DEFAULT_RESAMPLE_QUALITY,
          });
          modal?.current?.close()
        }}>Apply</Button>
        </div>
    </div>
  )
};
