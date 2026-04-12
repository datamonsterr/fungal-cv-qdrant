import { tool } from "@opencode-ai/plugin"
import { execSync } from "child_process"
import fs from "fs"
import path from "path"

export default tool({
  description: "Compile LaTeX main.tex to PDF in the same folder",
  args: {
    reportDir: tool.schema.string().describe("Path to report directory (e.g. report/threshold/001)"),
    texFile: tool.schema.string().optional().describe("TeX filename").default("main.tex"),
    passes: tool.schema.number().optional().describe("Number of pdflatex passes").default(1),
  },
  async execute(args, context) {
    const reportDir = path.resolve(context.worktree, args.reportDir)
    const texPath = path.join(reportDir, args.texFile)

    if (!fs.existsSync(texPath)) {
      return `Error: TeX file not found: ${texPath}`
    }

    try {
      for (let i = 0; i < args.passes; i++) {
        execSync(`pdflatex -interaction=nonstopmode -halt-on-error "${texPath}"`, {
          cwd: reportDir,
          stdio: "pipe",
        })
      }

      const pdfPath = texPath.replace(".tex", ".pdf")
      if (fs.existsSync(pdfPath)) {
        return `Compiled: ${pdfPath}`
      }
      return "PDF not found after compilation"
    } catch (err) {
      return `Compilation error: ${err.message}`
    }
  }
})

export const latex_clean = tool({
  description: "Clean LaTeX auxiliary files from a report directory",
  args: {
    reportDir: tool.schema.string().describe("Path to report directory"),
  },
  async execute(args, context) {
    const reportDir = path.resolve(context.worktree, args.reportDir)
    const auxExts = [".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".fls", ".fdb_latexmk"]

    let cleaned = 0
    for (const ext of auxExts) {
      const files = fs.readdirSync(reportDir).filter(f => f.endsWith(ext))
      for (const file of files) {
        fs.unlinkSync(path.join(reportDir, file))
        cleaned++
      }
    }
    return `Cleaned ${cleaned} aux files`
  }
})