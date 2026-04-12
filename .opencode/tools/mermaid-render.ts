import { tool } from "@opencode-ai/plugin"
import { execSync } from "child_process"
import fs from "fs"
import path from "path"

export default tool({
  description: "Render mermaid diagram to PNG image",
  args: {
    input: tool.schema.string().describe("Path to .mmd mermaid file or file containing mermaid blocks"),
    output: tool.schema.string().describe("Path for output PNG image"),
    format: tool.schema.string().optional().describe("Output format: png, svg, pdf").default("png"),
  },
  async execute(args, context) {
    const inputPath = path.resolve(context.worktree, args.input)
    const outputPath = path.resolve(context.worktree, args.output)
    const outputDir = path.dirname(outputPath)

    if (!fs.existsSync(inputPath)) {
      return `Error: Input file not found: ${inputPath}`
    }

    try {
      execSync(`mkdir -p "${outputDir}"`, { stdio: "pipe" })
      execSync(`mmdc -i "${inputPath}" -o "${outputPath}" -f ${args.format}`, { stdio: "pipe" })
      return `Rendered: ${outputPath}`
    } catch (err) {
      return `Error rendering mermaid: ${err.message}`
    }
  }
})

export const render_mermaid = tool({
  description: "Render mermaid diagram from markdown content to PNG",
  args: {
    markdown: tool.schema.string().describe("Markdown content with mermaid block"),
    output: tool.schema.string().describe("Output PNG path"),
  },
  async execute(args, context) {
    const fs = await import("fs")
    const path = await import("path")
    const os = await import("os")

    const mmdPath = path.join(os.tmpdir(), "diagram.mmd")

    const mermaidMatch = args.markdown.match(/```mermaid\n([\s\S]*?)```/)
    if (!mermaidMatch) {
      return "No mermaid block found in markdown"
    }

    fs.writeFileSync(mmdPath, mermaidMatch[1].trim())
    const outputPath = path.resolve(context.worktree, args.output)
    const outputDir = path.dirname(outputPath)

    try {
      execSync(`mkdir -p "${outputDir}"`, { stdio: "pipe" })
      execSync(`mmdc -i "${mmdPath}" -o "${outputPath}"`, { stdio: "pipe" })
      return `Rendered: ${outputPath}`
    } catch (err) {
      return `Error: ${err.message}`
    }
  }
})