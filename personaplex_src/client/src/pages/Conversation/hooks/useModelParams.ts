import { useCallback, useState } from "react";
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
// Audio buffer defaults
export const DEFAULT_INIT_BUFFER_MS = 400;
export const DEFAULT_PARTIAL_BUFFER_MS = 210;
export const DEFAULT_DECODER_BUFFER_SAMPLES = 3840;
export const DEFAULT_RESAMPLE_QUALITY = 5;
export const DEFAULT_SILENCE_DELAY_S = 0.07;

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
  decoderBufferSamples: number;
  resampleQuality: number;
  silenceDelayS: number;
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {

  const [textTemperature, setTextTemperatureBase] = useState(params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase]= useState(params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useState(params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useState(params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [textPrompt, setTextPromptBase] = useState(params?.textPrompt || DEFAULT_TEXT_PROMPT);
  const [voicePrompt, setVoicePromptBase] = useState(params?.voicePrompt || DEFAULT_VOICE_PROMPT);
  const [randomSeed, setRandomSeedBase] = useLocalStorage('randomSeed', params?.randomSeed || DEFAULT_RANDOM_SEED);
  // Audio buffer settings
  const [initBufferMs, setInitBufferMsBase] = useState(params?.initBufferMs || DEFAULT_INIT_BUFFER_MS);
  const [partialBufferMs, setPartialBufferMsBase] = useState(params?.partialBufferMs || DEFAULT_PARTIAL_BUFFER_MS);
  const [decoderBufferSamples, setDecoderBufferSamplesBase] = useState(params?.decoderBufferSamples || DEFAULT_DECODER_BUFFER_SAMPLES);
  const [resampleQuality, setResampleQualityBase] = useState(params?.resampleQuality || DEFAULT_RESAMPLE_QUALITY);
  const [silenceDelayS, setSilenceDelaySBase] = useState(params?.silenceDelayS || DEFAULT_SILENCE_DELAY_S);

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
    setDecoderBufferSamplesBase(DEFAULT_DECODER_BUFFER_SAMPLES);
    setResampleQualityBase(DEFAULT_RESAMPLE_QUALITY);
    setSilenceDelaySBase(DEFAULT_SILENCE_DELAY_S);
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
    setInitBufferMsBase(params.initBufferMs);
    setPartialBufferMsBase(params.partialBufferMs);
    setDecoderBufferSamplesBase(params.decoderBufferSamples);
    setResampleQualityBase(params.resampleQuality);
    setSilenceDelaySBase(params.silenceDelayS);
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
    if(value <= 1.2 || value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 || value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 || value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200|| value >= 0) {
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
    setRandomSeedBase(value);
  }, []);
  const setInitBufferMs = useCallback((value: number) => {
    setInitBufferMsBase(value);
    // Update global settings for audio processor
    (window as any).audioSettings = { ...(window as any).audioSettings, initBuf: value };
  }, []);
  const setPartialBufferMs = useCallback((value: number) => {
    setPartialBufferMsBase(value);
    (window as any).audioSettings = { ...(window as any).audioSettings, partialBuf: value };
  }, []);
  const setDecoderBufferSamples = useCallback((value: number) => {
    setDecoderBufferSamplesBase(value);
    (window as any).audioSettings = { ...(window as any).audioSettings, decBuf: value };
  }, []);
  const setResampleQuality = useCallback((value: number) => {
    setResampleQualityBase(value);
    (window as any).audioSettings = { ...(window as any).audioSettings, resampleQ: value };
  }, []);
  const setSilenceDelayS = useCallback((value: number) => {
    setSilenceDelaySBase(value);
    (window as any).audioSettings = { ...(window as any).audioSettings, silenceDelay: value };
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
    decoderBufferSamples,
    setDecoderBufferSamples,
    resampleQuality,
    setResampleQuality,
    silenceDelayS,
    setSilenceDelayS,
  }
}
