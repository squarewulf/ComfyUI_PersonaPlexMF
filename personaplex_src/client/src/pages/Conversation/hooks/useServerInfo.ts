import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { z } from "zod";

const ServersInfoSchema = z.object({
  status: z.string().optional(),
  text_temperature: z.number().optional(),
  text_topk: z.number().optional(),
  audio_temperature: z.number().optional(),
  audio_topk: z.number().optional(),
  pad_mult: z.number().optional(),
  repetition_penalty_context: z.number().optional(),
  repetition_penalty: z.number().optional(),
  lm_model_file: z.string().optional(),
  instance_name: z.string().optional(),
  voice_prompt: z.string().nullable().optional(),
  model_device: z.string().optional(),
  seed: z.number().nullable().optional(),
  build_info: z.object({
    build_timestamp: z.string().optional(),
    build_date: z.string().optional(),
    git_branch: z.string().optional(),
    git_timestamp: z.string().optional(),
    git_date: z.string().optional(),
    git_hash: z.string().optional(),
    git_describe: z.string().optional(),
    rustc_host_triple: z.string().optional(),
    rustc_version: z.string().optional(),
    cargo_target_triple: z.string().optional(),
  }).optional(),
}).passthrough();

const parseInfo = (infos: any) => {
  const serverInfo =  ServersInfoSchema.safeParse(infos);
  if (!serverInfo.success) {
    console.error(serverInfo.error);
    return null;
  }
  return serverInfo.data;
};

type ServerInfo = z.infer<typeof ServersInfoSchema>;

export const useServerInfo = () => {
  const [serverInfo, setServerInfo] = useState<ServerInfo|null>(null);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "metadata") {
      const infos = parseInfo(message.data);
      if (infos) {
        setServerInfo(infos);
        console.log("received metadata", infos);
      }
    }
  }, [setServerInfo]);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setServerInfo(null);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { serverInfo };
};
