import { useState } from 'react'
import type { Answer, Citation } from '../types'
import { MODE_LABELS } from '../types'

export type AnswerView = Pick<Answer, 'answer' | 'citations' | 'grounded' | 'mode'> &
  Partial<Pick<Answer, 'retrieval_ms' | 'generation_ms'>>

const IDK = "I don't know."

export function AnswerCard({ answer }: { answer: AnswerView }) {
  const refused = answer.answer === IDK
  // History items have no timings, and refused questions never reach the model.
  let timing = ''
  if (answer.retrieval_ms !== undefined) timing += ` · search ${Math.round(answer.retrieval_ms)} ms`
  if (answer.generation_ms) timing += ` · answer ${Math.round(answer.generation_ms)} ms`

  return (
    <div
      className={`rounded-xl border p-4 ${
        refused
          ? 'border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/40'
          : 'border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'
      }`}
    >
      <p className="whitespace-pre-wrap text-sm leading-relaxed">
        {refused ? "I don't know. The documents don't contain the answer to this question." : answer.answer}
      </p>
      {!refused && !answer.grounded && (
        <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">
          Warning: this answer did not cite any source.
        </p>
      )}
      {answer.citations.length > 0 && (
        <ul className="mt-3 space-y-2" aria-label="Sources">
          {answer.citations.map((c) => (
            <CitationItem key={c.number} citation={c} />
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs text-slate-500">
        {MODE_LABELS[answer.mode]}
        {timing}
      </p>
    </div>
  )
}

function CitationItem({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="rounded-lg bg-slate-50 text-xs dark:bg-slate-800/60">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <span className="rounded bg-indigo-100 px-1.5 font-mono text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200">
          [{citation.number}]
        </span>
        <span className="font-medium">{citation.filename}</span>
        <span className="text-slate-500">page {citation.page}</span>
        <span className="ml-auto text-slate-400">{open ? 'Hide' : 'Show'} text</span>
      </button>
      {open && <p className="px-3 pb-3 leading-relaxed text-slate-600 dark:text-slate-300">{citation.text}</p>}
    </li>
  )
}
