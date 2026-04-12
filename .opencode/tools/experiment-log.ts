import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Read and summarize experiment log results concisely",
  args: {
    experiment: tool.schema.string().describe("Experiment name"),
    lines: tool.schema.number().optional().default(10).describe("Number of recent lines to read"),
  },
  async execute(args) {
    const fs = await import("fs")
    const path = await import("path")
    
    const logPath = path.join(process.cwd(), "results", args.experiment, "log", "experiments.log")
    const bestPath = path.join(process.cwd(), "results", args.experiment, "log", "best_strategy.json")
    
    let output = ""
    
    if (fs.existsSync(logPath)) {
      const content = fs.readFileSync(logPath, "utf-8")
      const allLines = content.trim().split("\n")
      const recent = allLines.slice(-args.lines)
      output += `=== ${args.experiment} last ${recent.length} attempts ===\n`
      output += recent.join("\n") + "\n"
    }
    
    if (fs.existsSync(bestPath)) {
      const best = JSON.parse(fs.readFileSync(bestPath, "utf-8"))
      output += `\n=== Current best ===\n`
      output += `Strategy: ${best.strategy}\n`
      output += `F1: ${best.f1?.toFixed(4) || "N/A"}\n`
      if (best.threshold) output += `Threshold: ${best.threshold.toFixed(4)}\n`
    }
    
    return output || "No log data found"
  }
})