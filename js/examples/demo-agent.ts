// The pretend support agent both TypeScript examples run: a scripted "model" that sleeps a little per token, an
// order lookup, and a carrier call that times out once and is retried. The same agent as examples/demo_agent.py.
import { setTimeout as sleep } from "node:timers/promises";
import { agent, modelCall, tool } from "../src/index.ts";

export class TimeoutError extends Error {
  override name = "TimeoutError";
}

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

async function ask(model: string, msPerToken: number, prompt: string, reply: string): Promise<string> {
  return modelCall(model, async (m) => {
    const inputTokens = Math.floor(words(prompt) * 1.3) + 400; // plus a system prompt
    const outputTokens = Math.floor(words(reply) * 1.3);
    await sleep(20 + outputTokens * msPerToken);
    m.record({ inputTokens, outputTokens, finishReasons: ["stop"] });
    return reply;
  }, { provider: "example", maxTokens: 512 });
}

let trackCalls = 0;
/** 0: the next carrier call times out once; 1: it answers. */
export function setTrackCalls(n: number): void {
  trackCalls = n;
}

const findOrder = tool("find_order", async (_email: string) => {
  await sleep(35);
  return { order: "A-1042", status: "in transit", carrier: "parcel co" };
});
const trackParcel = tool("track_parcel", async (_order: string) => {
  await sleep(90);
  trackCalls += 1;
  if (trackCalls === 1) throw new TimeoutError("carrier API did not answer in 80 ms");
  return { eta: "Thursday" };
});

export function handle(question: string, conversation: string): Promise<string> {
  return agent("support-agent", async () => {
    await ask("local-qwen-7b", 4, question, "intent: order status. need email lookup then tracking");
    const order = await findOrder("someone@example.org");
    let eta: { eta: string } | undefined;
    for (let attempt = 0; attempt < 2 && !eta; attempt++) {
      try {
        eta = await trackParcel(order.order);
      } catch (e) {
        if (!(e instanceof TimeoutError) || attempt === 1) throw e;
      }
    }
    return ask("example-hosted-small", 1.5, `${question} ${order.order} ${eta?.eta}`,
      "Your order A-1042 is on its way with Parcel Co and should arrive on Thursday. " +
      "I will send you the tracking link by email as well.");
  }, { conversationId: conversation });
}
