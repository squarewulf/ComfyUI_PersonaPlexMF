import { FC, useRef } from "react";
import { AudioStats, useServerAudio } from "../../hooks/useServerAudio";
import { ServerVisualizer } from "../AudioVisualizer/ServerVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";

type ServerAudioProps = {
  setGetAudioStats: (getAudioStats: () => AudioStats) => void;
  theme: ThemeType;
};
export const ServerAudio: FC<ServerAudioProps> = ({ setGetAudioStats, theme }) => {
  const { analyser, hasCriticalDelay, setHasCriticalDelay } = useServerAudio({
    setGetAudioStats,
  });
  const containerRef = useRef<HTMLDivElement>(null);
  return (
    <>
      {hasCriticalDelay && (
        <div className="fixed left-0 top-0 flex w-screen justify-between items-center bg-red-500/20 border-b border-red-500/30 backdrop-blur-sm p-2.5 text-center z-50">
          <p className="text-xs text-red-300">A connection issue has been detected, you've been reconnected</p>
          <button
            onClick={async () => {
              setHasCriticalDelay(false);
            }}
            className="text-xs px-3 py-1 rounded bg-white/10 hover:bg-white/20 text-white/80 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}
      <div className="server-audio h-4/6 aspect-square" ref={containerRef}>
        <ServerVisualizer analyser={analyser.current} parent={containerRef} theme={theme}/>
      </div>
    </>
  );
};
