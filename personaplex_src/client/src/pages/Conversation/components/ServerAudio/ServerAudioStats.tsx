import { useState, useEffect, useRef } from "react";

type ServerAudioStatsProps = {
  getAudioStats: React.MutableRefObject<
    () => {
      playedAudioDuration: number;
      missedAudioDuration: number;
      totalAudioMessages: number;
      delay: number;
      minPlaybackDelay: number;
      maxPlaybackDelay: number;
    }
  >;
};

export const ServerAudioStats = ({ getAudioStats }: ServerAudioStatsProps) => {
  const [audioStats, setAudioStats] = useState(getAudioStats.current());

  const movingAverageSum = useRef<number>(0.);
  const movingAverageCount = useRef<number>(0.);
  const movingBeta = 0.85;

  let convertMinSecs = (total_secs: number) => {
    // convert secs to the format mm:ss.cc
    let mins = (Math.floor(total_secs / 60)).toString();
    let secs = (Math.floor(total_secs) % 60).toString();
    let cents = (Math.floor(100 * (total_secs - Math.floor(total_secs)))).toString();
    if (secs.length < 2) {
      secs = "0" + secs;
    }
    if (cents.length < 2) {
      cents = "0" + cents;
    }
    return mins + ":" + secs + "." + cents;
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const newAudioStats = getAudioStats.current();
      setAudioStats(newAudioStats);
      movingAverageCount.current *= movingBeta;
      movingAverageCount.current += (1 - movingBeta) * 1;
      movingAverageSum.current *= movingBeta;
      movingAverageSum.current += (1 - movingBeta) * newAudioStats.delay;

    }, 141);
    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="w-full rounded-lg p-3 glass-card">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-white/50 pb-2">Audio Stats</h2>
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-white/40">Played</span>
          <span className="font-mono text-white/70">{convertMinSecs(audioStats.playedAudioDuration)}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-white/40">Missed</span>
          <span className="font-mono text-red-400/70">{convertMinSecs(audioStats.missedAudioDuration)}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-white/40">Latency</span>
          <span className="font-mono text-[#76b900]/80">{(movingAverageSum.current / movingAverageCount.current).toFixed(3)}s</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-white/40">Buffer</span>
          <span className="font-mono text-white/70">{audioStats.minPlaybackDelay.toFixed(3)} / {audioStats.maxPlaybackDelay.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
};
