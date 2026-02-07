import { useCallback } from "react";
import {useLocalStorage} from './useLocalStorage';

export const DEFAULT_TEXT_TEMPERATURE = 0.7;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.8;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.0;
export const DEFAULT_TEXT_PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.";
export const DEFAULT_VOICE_PROMPT = "NATF0.pt";
export const DEFAULT_RANDOM_SEED = -1;

// Audio buffer defaults (in ms)
export const DEFAULT_INIT_BUFFER_MS = 80;
export const DEFAULT_PARTIAL_BUFFER_MS = 10;
export const DEFAULT_MAX_BUFFER_MS = 10;
export const DEFAULT_BUFFER_INCREMENT_MS = 5;
export const DEFAULT_MAX_PARTIAL_CAP_MS = 80;
export const DEFAULT_MAX_BUFFER_CAP_MS = 80;
export const DEFAULT_MAX_LATENCY_MS = 500;
export const DEFAULT_AUTO_FLUSH = true;
export const DEFAULT_FLUSH_SILENCE_FRAMES = 50;
export const DEFAULT_RESAMPLE_QUALITY = 5;

export type ModelParamsValues = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padMult: number;
  repetitionPenaltyContext: number,
  repetitionPenalty: number,
  textPrompt: string;
  voicePrompt: string;
  randomSeed: number;
  // Audio buffer settings
  initBufferMs: number;
  partialBufferMs: number;
  maxBufferMs: number;
  bufferIncrementMs: number;
  maxPartialCapMs: number;
  maxBufferCapMs: number;
  maxLatencyMs: number;
  autoFlush: boolean;
  flushSilenceFrames: number;
  resampleQuality: number;
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {

  const [textTemperature, setTextTemperatureBase] = useLocalStorage('pp_textTemperature', params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase] = useLocalStorage('pp_textTopk', params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useLocalStorage('pp_audioTemperature', params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useLocalStorage('pp_audioTopk', params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useLocalStorage('pp_padMult', params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useLocalStorage('pp_repetitionPenalty', params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useLocalStorage('pp_repetitionPenaltyContext', params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [textPrompt, setTextPromptBase] = useLocalStorage('pp_textPrompt', params?.textPrompt || DEFAULT_TEXT_PROMPT);
  const [voicePrompt, setVoicePromptBase] = useLocalStorage('pp_voicePrompt', params?.voicePrompt || DEFAULT_VOICE_PROMPT);
  const [randomSeed, setRandomSeedBase] = useLocalStorage('randomSeed', params?.randomSeed || DEFAULT_RANDOM_SEED);
  // Audio buffer settings
  const [initBufferMs, setInitBufferMsBase] = useLocalStorage('pp_initBufferMs', params?.initBufferMs || DEFAULT_INIT_BUFFER_MS);
  const [partialBufferMs, setPartialBufferMsBase] = useLocalStorage('pp_partialBufferMs', params?.partialBufferMs || DEFAULT_PARTIAL_BUFFER_MS);
  const [maxBufferMs, setMaxBufferMsBase] = useLocalStorage('pp_maxBufferMs', params?.maxBufferMs || DEFAULT_MAX_BUFFER_MS);
  const [bufferIncrementMs, setBufferIncrementMsBase] = useLocalStorage('pp_bufferIncrementMs', params?.bufferIncrementMs || DEFAULT_BUFFER_INCREMENT_MS);
  const [maxPartialCapMs, setMaxPartialCapMsBase] = useLocalStorage('pp_maxPartialCapMs', params?.maxPartialCapMs || DEFAULT_MAX_PARTIAL_CAP_MS);
  const [maxBufferCapMs, setMaxBufferCapMsBase] = useLocalStorage('pp_maxBufferCapMs', params?.maxBufferCapMs || DEFAULT_MAX_BUFFER_CAP_MS);
  const [maxLatencyMs, setMaxLatencyMsBase] = useLocalStorage('pp_maxLatencyMs', params?.maxLatencyMs || DEFAULT_MAX_LATENCY_MS);
  const [autoFlush, setAutoFlushBase] = useLocalStorage('pp_autoFlush', params?.autoFlush ?? DEFAULT_AUTO_FLUSH);
  const [flushSilenceFrames, setFlushSilenceFramesBase] = useLocalStorage('pp_flushSilenceFrames', params?.flushSilenceFrames || DEFAULT_FLUSH_SILENCE_FRAMES);
  const [resampleQuality, setResampleQualityBase] = useLocalStorage('pp_resampleQuality', params?.resampleQuality ?? DEFAULT_RESAMPLE_QUALITY);

  const resetParams = useCallback(() => {
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContext(DEFAULT_REPETITION_PENALTY_CONTEXT);
    setInitBufferMsBase(DEFAULT_INIT_BUFFER_MS);
    setPartialBufferMsBase(DEFAULT_PARTIAL_BUFFER_MS);
    setMaxBufferMsBase(DEFAULT_MAX_BUFFER_MS);
    setBufferIncrementMsBase(DEFAULT_BUFFER_INCREMENT_MS);
    setMaxPartialCapMsBase(DEFAULT_MAX_PARTIAL_CAP_MS);
    setMaxBufferCapMsBase(DEFAULT_MAX_BUFFER_CAP_MS);
    setMaxLatencyMsBase(DEFAULT_MAX_LATENCY_MS);
    setAutoFlushBase(DEFAULT_AUTO_FLUSH);
    setFlushSilenceFramesBase(DEFAULT_FLUSH_SILENCE_FRAMES);
    setResampleQualityBase(DEFAULT_RESAMPLE_QUALITY);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
  ]);

  const setParams = useCallback((params: ModelParamsValues) => {
    setTextTemperatureBase(params.textTemperature);
    setTextTopkBase(params.textTopk);
    setAudioTemperatureBase(params.audioTemperature);
    setAudioTopkBase(params.audioTopk);
    setPadMultBase(params.padMult);
    setRepetitionPenaltyBase(params.repetitionPenalty);
    setRepetitionPenaltyContextBase(params.repetitionPenaltyContext);
    setTextPromptBase(params.textPrompt);
    setVoicePromptBase(params.voicePrompt);
    setRandomSeedBase(params.randomSeed);
    if (params.initBufferMs !== undefined) setInitBufferMsBase(params.initBufferMs);
    if (params.partialBufferMs !== undefined) setPartialBufferMsBase(params.partialBufferMs);
    if (params.maxBufferMs !== undefined) setMaxBufferMsBase(params.maxBufferMs);
    if (params.bufferIncrementMs !== undefined) setBufferIncrementMsBase(params.bufferIncrementMs);
    if (params.maxPartialCapMs !== undefined) setMaxPartialCapMsBase(params.maxPartialCapMs);
    if (params.maxBufferCapMs !== undefined) setMaxBufferCapMsBase(params.maxBufferCapMs);
    if (params.maxLatencyMs !== undefined) setMaxLatencyMsBase(params.maxLatencyMs);
    if (params.autoFlush !== undefined) setAutoFlushBase(params.autoFlush);
    if (params.flushSilenceFrames !== undefined) setFlushSilenceFramesBase(params.flushSilenceFrames);
    if (params.resampleQuality !== undefined) setResampleQualityBase(params.resampleQuality);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    setTextPromptBase,
    setVoicePromptBase,
    setRandomSeedBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if(value <= 1.2 && value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 && value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 && value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 && value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 && value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 && value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200 && value >= 0) {
      setRepetitionPenaltyContextBase(value);
    }
  }, []);
  const setTextPrompt = useCallback((value: string) => {
    setTextPromptBase(value);
  }, []);
  const setVoicePrompt = useCallback((value: string) => {
    setVoicePromptBase(value);
  }, []);
  const setRandomSeed = useCallback((value: number) => {
    if (value >= -1 && value <= 2147483647) {
      setRandomSeedBase(value);
    }
  }, []);
  const setInitBufferMs = useCallback((value: number) => {
    if (value >= 10 && value <= 500) setInitBufferMsBase(value);
  }, []);
  const setPartialBufferMs = useCallback((value: number) => {
    if (value >= 0 && value <= 200) setPartialBufferMsBase(value);
  }, []);
  const setMaxBufferMs = useCallback((value: number) => {
    if (value >= 0 && value <= 500) setMaxBufferMsBase(value);
  }, []);
  const setBufferIncrementMs = useCallback((value: number) => {
    if (value >= 0 && value <= 50) setBufferIncrementMsBase(value);
  }, []);
  const setMaxPartialCapMs = useCallback((value: number) => {
    if (value >= 10 && value <= 500) setMaxPartialCapMsBase(value);
  }, []);
  const setMaxBufferCapMs = useCallback((value: number) => {
    if (value >= 10 && value <= 500) setMaxBufferCapMsBase(value);
  }, []);
  const setMaxLatencyMs = useCallback((value: number) => {
    if (value >= 100 && value <= 5000) setMaxLatencyMsBase(value);
  }, []);
  const setAutoFlush = useCallback((value: boolean) => {
    setAutoFlushBase(value);
  }, []);
  const setFlushSilenceFrames = useCallback((value: number) => {
    if (value >= 20 && value <= 100) setFlushSilenceFramesBase(value);
  }, []);
  const setResampleQuality = useCallback((value: number) => {
    if (value >= 0 && value <= 10) setResampleQualityBase(value);
  }, []);

  return {
    textTemperature,
    textTopk,
    audioTemperature,
    audioTopk,
    padMult,
    repetitionPenalty,
    repetitionPenaltyContext,
    setTextTemperature,
    setTextTopk,
    setAudioTemperature,
    setAudioTopk,
    setPadMult,
    setRepetitionPenalty,
    setRepetitionPenaltyContext,
    setTextPrompt,
    textPrompt,
    setVoicePrompt,
    voicePrompt,
    resetParams,
    setParams,
    randomSeed,
    setRandomSeed,
    // Audio buffer settings
    initBufferMs,
    setInitBufferMs,
    partialBufferMs,
    setPartialBufferMs,
    maxBufferMs,
    setMaxBufferMs,
    bufferIncrementMs,
    setBufferIncrementMs,
    maxPartialCapMs,
    setMaxPartialCapMs,
    maxBufferCapMs,
    setMaxBufferCapMs,
    maxLatencyMs,
    setMaxLatencyMs,
    autoFlush,
    setAutoFlush,
    flushSilenceFrames,
    setFlushSilenceFrames,
    resampleQuality,
    setResampleQuality,
  }
}
