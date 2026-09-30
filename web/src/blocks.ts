// 时间线条目 → 界面上的块。一段连续的「思考 + 工具调用」合成一个 activity 块（界面上是一行淡灰小字），
// 回答正文、用户的话、提问、提示各自成块（学 Claude desktop 的「Used 2 tools ›」）。

import type { Item } from "./types";

type Assistant = Extract<Item, { kind: "assistant" }>;
export type Tool = Extract<Item, { kind: "tool" }>;

export type Step = { kind: "thinking"; id: number; text: string; live: boolean } | { kind: "tool"; id: number; tool: Tool };

export type Block =
  | { kind: "user"; id: number; text: string }
  | { kind: "text"; id: number; item: Assistant }
  | { kind: "activity"; id: number; steps: Step[]; figures: string[] }
  | { kind: "ask"; id: number; tool: Tool }
  | { kind: "notice"; id: number; text: string; level: "info" | "warn" | "error" };

export function toBlocks(items: Item[]): Block[] {
  const blocks: Block[] = [];
  let group: Extract<Block, { kind: "activity" }> | null = null;
  const activity = () => {
    if (!group) {
      group = { kind: "activity", id: 0, steps: [], figures: [] };
      blocks.push(group);
    }
    return group;
  };

  for (const it of items) {
    if (it.kind === "assistant") {
      if (it.thinking) {
        const g = activity();
        if (!g.id) g.id = it.id;
        g.steps.push({ kind: "thinking", id: it.id, text: it.thinking, live: it.streaming && !it.text });
      }
      if (it.text) {
        group = null;
        blocks.push({ kind: "text", id: it.id, item: it });
      }
    } else if (it.kind === "tool") {
      if (it.name === "ask_user") {
        group = null;
        blocks.push({ kind: "ask", id: it.id, tool: it });
        continue;
      }
      const g = activity();
      if (!g.id) g.id = it.id;
      g.steps.push({ kind: "tool", id: it.id, tool: it });
      if (it.details?.kind === "execution") g.figures.push(...it.details.figures);
    } else {
      group = null;
      blocks.push(it.kind === "user" ? { kind: "user", id: it.id, text: it.text }
                                     : { kind: "notice", id: it.id, text: it.text, level: it.level });
    }
  }
  // 同一张图重画了好几次（文件名一样），只留最后一版
  for (const b of blocks) {
    if (b.kind === "activity") {
      const latest = new Map(b.figures.map((u) => [u.split("?")[0], u]));
      b.figures = [...latest.values()];
    }
  }
  return blocks;
}
