import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'
import type { AnswersReport, EvaluationResults, RetrievalReport, SearchMode } from '../types'
import { MODE_LABELS } from '../types'
import { percent } from '../styles'
import { Card, ErrorNote } from './ui'

const SERIES_COLORS = ['#6366f1', '#14b8a6', '#f59e0b']

export function EvaluationPage() {
  const [results, setResults] = useState<EvaluationResults | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .evaluation()
      .then(setResults)
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load results'))
  }, [])

  if (error) return <ErrorNote message={error} />
  if (!results) return <p className="text-sm text-slate-500">Loading results…</p>
  return <EvaluationView results={results} />
}

export function EvaluationView({ results }: { results: EvaluationResults }) {
  const { retrieval, answers } = results
  if (!retrieval) {
    return (
      <Card>
        <p className="text-sm">
          No results yet. Run <code>python -m evaluation.run retrieval</code> to benchmark search.
        </p>
      </Card>
    )
  }
  return (
    <div className="space-y-5">
      <RetrievalSection report={retrieval} />
      <AnswersSection report={answers} />
    </div>
  )
}

function RetrievalSection({ report }: { report: RetrievalReport }) {
  const modes = Object.keys(report.modes) as SearchMode[]
  const qualityData = modes.map((mode) => ({
    mode: MODE_LABELS[mode],
    'Recall@1': +(report.modes[mode].recall_at_1 * 100).toFixed(1),
    'Recall@5': +(report.modes[mode].recall_at_5 * 100).toFixed(1),
    'MRR@5': +(report.modes[mode].mrr_at_5 * 100).toFixed(1),
  }))
  const latencyData = modes.map((mode) => ({
    mode: MODE_LABELS[mode],
    p50: report.modes[mode].latency_p50_ms,
    p95: report.modes[mode].latency_p95_ms,
  }))
  const best = modes.reduce((a, b) => (report.modes[b].mrr_at_5 > report.modes[a].mrr_at_5 ? b : a))

  return (
    <>
      <Card>
        <h2 className="text-lg font-semibold">Search quality</h2>
        <p className="mt-1 text-sm text-slate-500">
          {report.questions} questions ({report.answerable} answerable, {report.unanswerable} with no
          answer in the documents) · generated {new Date(report.generated_at).toLocaleString()}
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <Stat label={`Best mode: ${MODE_LABELS[best]}`} value={percent(report.modes[best].recall_at_1)} hint="right passage ranked first (Recall@1)" />
          <Stat label="Recall@5" value={percent(report.modes[best].recall_at_5)} hint="right passage in the top 5" />
          <Stat label="Off-topic rejected" value={percent(report.modes[best].unanswerable_rejected)} hint="refused before calling the model" />
        </div>
        <div className="mt-6 h-72" aria-label="Search quality by mode chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={qualityData} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
              <XAxis dataKey="mode" fontSize={12} />
              <YAxis domain={[0, 100]} unit="%" fontSize={12} />
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
              {['Recall@1', 'Recall@5', 'MRR@5'].map((key, i) => (
                <Bar key={key} dataKey={key} fill={SERIES_COLORS[i]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="font-semibold">Search latency</h3>
          <p className="text-sm text-slate-500">Milliseconds per question on this machine (CPU).</p>
          <div className="mt-4 h-60" aria-label="Latency by mode chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData} margin={{ left: -5 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
                <XAxis dataKey="mode" fontSize={12} />
                <YAxis unit=" ms" fontSize={12} />
                <Tooltip formatter={(value) => `${value} ms`} />
                <Legend />
                <Bar dataKey="p50" fill={SERIES_COLORS[0]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="p95" fill={SERIES_COLORS[2]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <h3 className="font-semibold">Results table</h3>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-slate-500">
                <tr>
                  <th className="py-2 pr-3">Mode</th>
                  <th className="py-2 pr-3">R@1</th>
                  <th className="py-2 pr-3">R@5</th>
                  <th className="py-2 pr-3">MRR@5</th>
                  <th className="py-2">p50</th>
                </tr>
              </thead>
              <tbody>
                {modes.map((mode) => {
                  const m = report.modes[mode]
                  return (
                    <tr key={mode} className="border-t border-slate-200 dark:border-slate-800">
                      <td className="py-2 pr-3 font-medium">{MODE_LABELS[mode]}</td>
                      <td className="py-2 pr-3">{percent(m.recall_at_1)}</td>
                      <td className="py-2 pr-3">{percent(m.recall_at_5)}</td>
                      <td className="py-2 pr-3">{m.mrr_at_5.toFixed(3)}</td>
                      <td className="py-2">{Math.round(m.latency_p50_ms)} ms</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs text-slate-500">
            Settings: {String(report.config.embedding_model)}, reranker {String(report.config.rerank_model)},
            chunks of {report.config.chunk_size} characters, top {report.config.top_k}.
          </p>
        </Card>
      </div>
    </>
  )
}

function AnswersSection({ report }: { report: AnswersReport | null }) {
  if (!report) {
    return (
      <Card>
        <h2 className="text-lg font-semibold">Answer quality</h2>
        <p className="mt-1 text-sm text-slate-500">
          Not measured yet. With Ollama running, run <code>python -m evaluation.run answers</code> to
          grade every answer for correctness and faithfulness.
        </p>
      </Card>
    )
  }
  const m = report.metrics
  return (
    <Card>
      <h2 className="text-lg font-semibold">Answer quality</h2>
      <p className="mt-1 text-sm text-slate-500">
        {String(report.config.llm_model)} · {MODE_LABELS[report.mode]} · graded by the same model as judge
      </p>
      <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Correct" value={percent(m.correctness)} hint="of answerable questions" />
        <Stat label="Faithful" value={percent(m.faithfulness)} hint="claims backed by sources" />
        <Stat label="Right citation" value={percent(m.citation_accuracy)} hint="cites the evidence" />
        <Stat label="Refused correctly" value={percent(m.refusal_accuracy)} hint="unanswerable → I don't know" />
        <Stat label="Latency p50" value={`${(m.latency_p50_ms / 1000).toFixed(1)} s`} hint={`p95 ${(m.latency_p95_ms / 1000).toFixed(1)} s`} />
        <Stat label="Cost / question" value={`$${m.cost_per_question_usd.toFixed(4)}`} hint={`${Math.round(m.avg_input_tokens + m.avg_output_tokens)} tokens`} />
      </div>
    </Card>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3 dark:bg-slate-800/50">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      <p className="text-xs text-slate-500">{hint}</p>
    </div>
  )
}
