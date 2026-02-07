import { useServerInfo } from "../../hooks/useServerInfo";
import { useSocketContext } from "../../SocketContext";

export const ServerInfo = () => {
  const { serverInfo } = useServerInfo();
  const { socketStatus } = useSocketContext();
  const connectionStatus =
    socketStatus === "connected"
      ? "Connected"
      : socketStatus === "connecting"
        ? "Warming up"
        : "Disconnected";

  const rows = [
    { label: "Server status", value: serverInfo?.status },
    { label: "Text temperature", value: serverInfo?.text_temperature },
    { label: "Text topk", value: serverInfo?.text_topk },
    { label: "Audio temperature", value: serverInfo?.audio_temperature },
    { label: "Audio topk", value: serverInfo?.audio_topk },
    { label: "Seed", value: serverInfo?.seed },
    { label: "Voice prompt", value: serverInfo?.voice_prompt },
    { label: "Model device", value: serverInfo?.model_device },
    { label: "LM model file", value: serverInfo?.lm_model_file },
    { label: "Instance name", value: serverInfo?.instance_name },
  ].filter((row) => row.value !== undefined && row.value !== null && row.value !== "");

  return (
    <div className="p-3 self-center flex flex-col break-words glass-card text-sm max-w-lg w-full">
      <div className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
        Connection: <span className={
          connectionStatus === "Connected" ? "text-[#76b900]" :
          connectionStatus === "Warming up" ? "text-amber-400" : "text-red-400"
        }>{connectionStatus}</span>
      </div>
      {rows.length > 0 && (
        <div className="flex flex-col gap-0.5">
          {rows.map((row) => (
            <div key={row.label} className="flex justify-between text-xs">
              <span className="text-white/40">{row.label}</span>
              <span className="font-mono text-white/60 text-right max-w-[60%] truncate">{row.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
